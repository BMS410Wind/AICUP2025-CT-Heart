import os
import torch
import glob
import numpy as np
from monai.networks.nets import DynUNet
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, Orientation,
    Spacing, ScaleIntensityRangePercentiles, CropForeground, SpatialPad
)
from monai.inferers import sliding_window_inference
import nibabel as nib
from tqdm import tqdm
import logging
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from scipy.ndimage import binary_closing, binary_opening

# ==================== 配置 ====================
CONFIG = {
    # 模型路徑
    "model_path": r"D:\aicup\experiments\nnunet_all_classes\best_model_nnunet.pth",
    
    # 測試資料
    "test_dir": r"D:\aicup\test\41_testing_image",
    "output_dir": r"D:\aicup\predictions\nnunet_all_classes",
    
    # 模型配置(需與訓練時一致)
    "in_channels": 1,
    "out_channels": 4,  # 0=背景, 1=心臟, 2=心包膜, 3=鈣化
    "spatial_dims": 3,
    "kernels": [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]],
    "deep_supervision": False,  # 與 trainall.py 一致
    "deep_supr_num": 2,
    
    # 推理設定
    "target_spacing": (1.5, 1.5, 2.0),
    "roi_size": (128, 128, 128),
    "overlap": 0.6,  # 與訓練時 sliding window 一致
    "sw_batch_size": 4,
    
    # 後處理設定
    "use_postprocess": True,
    "postprocess_mode": "aggressive",  # "minimal" / "moderate" / "aggressive"
    "min_size_class1": 500,   # 心臟最小體素數
    "min_size_class2": 5,    # 心包膜最小體素數
    "min_size_class3": 0,    # 鈣化最小體素數
    "fill_holes": True,
    "use_morphology": True,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== 日誌設定 ====================
log_file = os.path.join(CONFIG["output_dir"], 'prediction.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# ==================== 後處理函數 ====================
def remove_small_components(prediction, min_sizes):
    """移除小碎片"""
    result = prediction.copy()
    structure = generate_binary_structure(3, 3)
    
    for class_id, min_size in min_sizes.items():
        mask = (prediction == class_id)
        if np.any(mask):
            labeled, num_features = label(mask, structure=structure)
            component_sizes = np.bincount(labeled.ravel())
            
            for i in range(1, num_features + 1):
                if component_sizes[i] < min_size:
                    result[labeled == i] = 0
    
    return result

def fill_holes_all_classes(prediction):
    """填補所有類別的內部孔洞"""
    result = prediction.copy()
    
    for class_id in [1, 2, 3]:
        mask = (prediction == class_id)
        if np.any(mask):
            filled = binary_fill_holes(mask)
            result[filled & ~mask] = class_id
    
    return result

# def morphological_smooth(prediction):
#     """形態學平滑"""
#     result = prediction.copy()
#     structure = generate_binary_structure(3, 1)
    
#     for class_id in [1, 2, 3]:
#         mask = (prediction == class_id)
#         if np.any(mask):
#             # 閉運算:連接小間隙
#             mask = binary_closing(mask, structure=structure, iterations=1)
#             # 開運算:移除小突起
#             mask = binary_opening(mask, structure=structure, iterations=1)
            
#             result[mask] = class_id
#             result[~mask & (result == class_id)] = 0
    
#     return result
def morphological_smooth(prediction):
    """形態學平滑"""
    result = prediction.copy()
    
    for class_id in [1, 2, 3]:
        mask = (prediction == class_id)
        if np.any(mask):
            if class_id == 2:  # 心包膜需要更多迭代
                # 使用更大的結構元素
                structure = generate_binary_structure(3, 2)
                # 閉運算:連接小間隙
                mask = binary_closing(mask, structure=structure, iterations=2)
                # 開運算:移除小突起
                mask = binary_opening(mask, structure=structure, iterations=1)
            else:
                structure = generate_binary_structure(3, 1)
                mask = binary_closing(mask, structure=structure, iterations=1)
                mask = binary_opening(mask, structure=structure, iterations=1)
            
            result[mask] = class_id
            result[~mask & (result == class_id)] = 0
    
    return result
def keep_largest_component_per_class(prediction, apply_to_classes):
    """保留每個類別的最大連通域"""
    result = prediction.copy()
    structure = generate_binary_structure(3, 3)
    
    for class_id in apply_to_classes:
        mask = (prediction == class_id)
        if np.any(mask):
            labeled, num_features = label(mask, structure=structure)
            if num_features > 1:
                component_sizes = np.bincount(labeled.ravel())
                largest_component = component_sizes[1:].argmax() + 1
                result[(labeled != largest_component) & (result == class_id)] = 0
    
    return result

def comprehensive_postprocess(prediction):
    """完整的後處理流程"""
    mode = CONFIG.get("postprocess_mode", "moderate")
    logger.info(f"    開始後處理 (模式: {mode})...")
    
    if mode == "minimal":
        # 最小後處理
        prediction = remove_small_components(
            prediction,
            {
                1: CONFIG["min_size_class1"],
                2: CONFIG["min_size_class2"],
                3: CONFIG["min_size_class3"]
            }
        )
        logger.info("      ✅ 移除小碎片")
        
        if CONFIG.get("fill_holes", True):
            prediction = fill_holes_all_classes(prediction)
            logger.info("      ✅ 填補孔洞")
    
    elif mode == "moderate":
        # 中等後處理
        prediction = remove_small_components(
            prediction,
            {
                1: CONFIG["min_size_class1"],
                2: CONFIG["min_size_class2"],
                3: CONFIG["min_size_class3"]
            }
        )
        logger.info("      ✅ 移除小碎片")
        
        prediction = fill_holes_all_classes(prediction)
        logger.info("      ✅ 填補孔洞")
        
        if CONFIG.get("use_morphology", True):
            prediction = morphological_smooth(prediction)
            logger.info("      ✅ 形態學平滑")
        
        # 只對心臟保留最大連通域
        prediction = keep_largest_component_per_class(prediction, [1])
        logger.info("      ✅ 保留心臟最大連通域")
    
    elif mode == "aggressive":
        # 激進後處理
        prediction = remove_small_components(
            prediction,
            {
                1: CONFIG["min_size_class1"],
                2: CONFIG["min_size_class2"],
                3: CONFIG["min_size_class3"]
            }
        )
        logger.info("      ✅ 移除小碎片")
        
        prediction = fill_holes_all_classes(prediction)
        logger.info("      ✅ 填補孔洞")
        
        prediction = morphological_smooth(prediction)
        logger.info("      ✅ 形態學平滑")
        
        # 對心臟和心包膜保留最大連通域
        prediction = keep_largest_component_per_class(prediction, [1, 2])
        logger.info("      ✅ 保留最大連通域")
        
        # 再次清理小碎片
        prediction = remove_small_components(
            prediction,
            {
                1: CONFIG["min_size_class1"],
                2: CONFIG["min_size_class2"],
                3: CONFIG["min_size_class3"]
            }
        )
        logger.info("      ✅ 最終清理")
    
    return prediction

# ==================== 載入模型 ====================
logger.info("="*80)
logger.info("載入 nnU-Net 模型...")
logger.info("="*80)

model = DynUNet(
    spatial_dims=CONFIG["spatial_dims"],
    in_channels=CONFIG["in_channels"],
    out_channels=CONFIG["out_channels"],
    kernel_size=CONFIG["kernels"],
    strides=CONFIG["strides"],
    upsample_kernel_size=CONFIG["strides"][1:],
    norm_name="instance",
    deep_supervision=CONFIG["deep_supervision"],
    deep_supr_num=CONFIG["deep_supr_num"],
    res_block=True,
).to(CONFIG["device"])

checkpoint = torch.load(CONFIG["model_path"], map_location=CONFIG["device"], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

total_params = sum(p.numel() for p in model.parameters())

logger.info(f"✅ 模型載入成功")
logger.info(f"   模型參數量: {total_params:,}")
if 'val_dice_mean' in checkpoint:
    logger.info(f"   驗證 Mean Dice: {checkpoint['val_dice_mean']:.4f}")
    logger.info(f"     - Class 1 (Heart): {checkpoint.get('val_dice_class1', 0):.4f}")
    logger.info(f"     - Class 2 (Pericardium): {checkpoint.get('val_dice_class2', 0):.4f}")
    logger.info(f"     - Class 3 (Calcium): {checkpoint.get('val_dice_class3', 0):.4f}")
    logger.info(f"   訓練 Epoch: {checkpoint.get('epoch', 'Unknown')}")

# ==================== 資料預處理 ====================
# 移除這部分,我們會在 predict_case 中定義

# ==================== 取得測試檔案 ====================
test_files = sorted(glob.glob(os.path.join(CONFIG["test_dir"], "patient*.nii.gz")))
logger.info(f"\n找到 {len(test_files)} 個測試檔案")
# 驗證檔案完整性
logger.info("\n驗證檔案完整性...")
valid_files = []
corrupted_files = []

for test_file in test_files:
    try:
        # 嘗試讀取檔案頭
        img = nib.load(test_file)
        valid_files.append(test_file)
    except Exception as e:
        patient_id = os.path.basename(test_file)
        logger.warning(f"⚠️ 檔案損壞或無法讀取: {patient_id} - {str(e)}")
        corrupted_files.append(test_file)

logger.info(f"✅ 有效檔案: {len(valid_files)}")
if corrupted_files:
    logger.warning(f"❌ 損壞檔案: {len(corrupted_files)}")
    for cf in corrupted_files:
        logger.warning(f"   - {os.path.basename(cf)}")

# 使用有效檔案進行預測
test_files = valid_files

if len(test_files) > 0:
    logger.info(f"測試檔案範圍: {os.path.basename(test_files[0])} 到 {os.path.basename(test_files[-1])}")
else:
    logger.error("沒有有效的測試檔案!")
    exit(1)

# ==================== 預測函數 ====================
def predict_case(image_path):
    """對單一案例進行預測,確保正確對齊原始影像空間"""
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
        ScaleIntensityRangePercentilesd, CropForegroundd, SpatialPadd,
        Invertd
    )
    from monai.data import MetaTensor
    
    # 載入原始影像資訊
    original_img = nib.load(image_path)
    original_shape = original_img.shape
    original_affine = original_img.affine
    original_header = original_img.header
    
    logger.info(f"  原始尺寸: {original_shape}")
    
    # 建立資料字典
    data_dict = {"image": image_path}
    
    # 定義可逆的 transforms
    pre_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="LPS"),
        Spacingd(keys=["image"], pixdim=CONFIG["target_spacing"], mode="bilinear"),
        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=0.5,
            upper=99.5,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=CONFIG["roi_size"]),
    ])
    
    logger.info("  預處理影像...")
    data_dict = pre_transforms(data_dict)
    image = data_dict["image"]
    
    logger.info(f"  預處理後尺寸: {image.shape[1:]}")
    
    # 預測
    logger.info("  執行推理...")
    image_tensor = image.unsqueeze(0).to(CONFIG["device"])
    
    with torch.no_grad():
        output = sliding_window_inference(
            image_tensor,
            CONFIG["roi_size"],
            sw_batch_size=CONFIG["sw_batch_size"],
            predictor=model,
            overlap=CONFIG["overlap"],
            mode="gaussian"
        )
        
        if isinstance(output, list):
            output = output[0]
    
    # 取得預測結果
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    logger.info(f"  預測尺寸: {prediction.shape}")
    
    # 後處理
    if CONFIG["use_postprocess"]:
        prediction = comprehensive_postprocess(prediction)
    
    # 將預測結果轉換為 MetaTensor 以使用 inverse transform
    pred_metatensor = MetaTensor(
        torch.from_numpy(prediction).unsqueeze(0).float(),
        meta=image.meta
    )
    
    # 反向轉換
    logger.info("  反向轉換到原始空間...")
    post_transforms = Compose([
        Invertd(
            keys=["pred"],
            transform=pre_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=True,
            to_tensor=True,
        )
    ])
    
    # 準備反向轉換的資料
    data_dict["pred"] = pred_metatensor
    data_dict["pred_meta_dict"] = image.meta
    
    try:
        # 嘗試使用 Invert
        data_dict = post_transforms(data_dict)
        prediction_original_space = data_dict["pred"].squeeze().numpy().astype(np.uint8)
    except Exception as e:
        logger.warning(f"  Invert 失敗: {e}, 使用簡單 resize")
        # 如果 invert 失敗,使用簡單的 resize
        from scipy.ndimage import zoom
        
        zoom_factors = [
            orig / pred 
            for orig, pred in zip(original_shape, prediction.shape)
        ]
        prediction_original_space = zoom(
            prediction, 
            zoom_factors, 
            order=0  # nearest neighbor
        ).astype(np.uint8)
    
    logger.info(f"  最終尺寸: {prediction_original_space.shape}")
    
    # 統計資訊
    class_counts = {
        0: np.sum(prediction_original_space == 0),
        1: np.sum(prediction_original_space == 1),
        2: np.sum(prediction_original_space == 2),
        3: np.sum(prediction_original_space == 3)
    }
    
    return prediction_original_space, original_affine, original_header, class_counts

# ==================== 批次預測 ====================
logger.info("\n" + "="*80)
logger.info("開始預測...")
logger.info("="*80)
logger.info(f"輸出格式: 0=背景, 1=心臟肌肉, 2=主動脈心包膜, 3=鈣化")
logger.info(f"後處理: {'✅ 開啟' if CONFIG['use_postprocess'] else '❌ 關閉'} (模式: {CONFIG.get('postprocess_mode', 'minimal')})")
logger.info(f"Device: {CONFIG['device']}")
logger.info("="*80)

success_count = 0
failed_cases = []

for test_file in tqdm(test_files, desc="預測中"):
    patient_id = os.path.basename(test_file).replace('.nii.gz', '')
    
    try:
        logger.info(f"\n處理 {patient_id}...")
        
        # 清理 GPU 緩存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 預測
        prediction, affine, header, class_counts = predict_case(test_file)
        
        # 顯示統計
        total_pixels = prediction.size
        logger.info(f"  預測統計:")
        logger.info(f"    背景: {class_counts[0]:,} pixels ({class_counts[0]/total_pixels*100:.2f}%)")
        logger.info(f"    心臟: {class_counts[1]:,} pixels ({class_counts[1]/total_pixels*100:.2f}%)")
        logger.info(f"    心包膜: {class_counts[2]:,} pixels ({class_counts[2]/total_pixels*100:.2f}%)")
        logger.info(f"    鈣化: {class_counts[3]:,} pixels ({class_counts[3]/total_pixels*100:.2f}%)")
        
        # 儲存預測結果
        output_path = os.path.join(CONFIG["output_dir"], f"{patient_id}_predict.nii.gz")
        pred_img = nib.Nifti1Image(prediction.astype(np.uint8), affine, header)
        nib.save(pred_img, output_path)
        
        success_count += 1
        logger.info(f"  ✅ 儲存成功: {output_path}")
        
    except Exception as e:
        logger.error(f"  ❌ 預測失敗 {patient_id}: {str(e)}")
        # 只記錄簡短錯誤訊息,避免 logging 問題
        failed_cases.append(patient_id)

# ==================== 總結 ====================
logger.info("\n" + "="*80)
logger.info(f"預測完成!")
logger.info("="*80)
logger.info(f"成功: {success_count}/{len(test_files)}")
if failed_cases:
    logger.warning(f"失敗案例: {failed_cases}")
else:
    logger.info("✅ 所有案例預測成功!")

# ==================== 建立提交檔案 ====================
import zipfile

zip_path = os.path.join(os.path.dirname(CONFIG["output_dir"]), "submission_nnunet_all_classes.zip")

logger.info(f"\n建立提交檔案...")
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    pred_files = sorted(glob.glob(os.path.join(CONFIG["output_dir"], "*_predict.nii.gz")))
    for pred_file in tqdm(pred_files, desc="壓縮中"):
        zipf.write(pred_file, os.path.basename(pred_file))

logger.info(f"✅ 提交檔案已建立: {zip_path}")
logger.info(f"📦 包含 {len(pred_files)} 個預測檔案")

# 檢查提交檔案
logger.info(f"\n檢查提交檔案內容...")
with zipfile.ZipFile(zip_path, 'r') as zipf:
    file_list = sorted(zipf.namelist())
    logger.info(f"壓縮檔中有 {len(file_list)} 個檔案")
    
    # 顯示前5個和後5個檔案
    if len(file_list) > 10:
        logger.info(f"前5個檔案: {file_list[:5]}")
        logger.info(f"後5個檔案: {file_list[-5:]}")
    else:
        logger.info(f"所有檔案: {file_list}")
    
    # 檢查是否有缺少的檔案(假設從 patient0051 開始)
    # 根據實際測試集範圍調整
    first_patient = int(os.path.basename(test_files[0]).replace('patient', '').replace('.nii.gz', ''))
    last_patient = int(os.path.basename(test_files[-1]).replace('patient', '').replace('.nii.gz', ''))
    
    expected_files = [f"patient{i:04d}_predict.nii.gz" for i in range(first_patient, last_patient + 1)]
    missing_in_zip = set(expected_files) - set(file_list)
    
    if missing_in_zip:
        logger.warning(f"⚠️ 壓縮檔中缺少 {len(missing_in_zip)} 個檔案:")
        for mf in sorted(missing_in_zip)[:10]:  # 只顯示前10個
            logger.warning(f"   - {mf}")
        if len(missing_in_zip) > 10:
            logger.warning(f"   ... 還有 {len(missing_in_zip) - 10} 個")
    else:
        logger.info(f"✅ 壓縮檔內容完整 (patient{first_patient:04d} 到 patient{last_patient:04d})")

logger.info(f"\n" + "="*80)
logger.info("✨ 全部完成!")
logger.info("="*80)
logger.info(f"\n模型策略:")
logger.info(f"  - nnU-Net (DynUNet): 直接預測所有類別 (1, 2, 3)")
logger.info(f"  - Deep Supervision: {CONFIG['deep_supervision']}")
logger.info(f"  - 與訓練配置完全一致")
logger.info(f"\n推理設定:")
logger.info(f"  - ROI Size: {CONFIG['roi_size']}")
logger.info(f"  - Overlap: {CONFIG['overlap']}")
logger.info(f"  - Mode: Gaussian")
logger.info(f"  - SW Batch Size: {CONFIG['sw_batch_size']}")
logger.info(f"\n後處理策略:")
if CONFIG["use_postprocess"]:
    logger.info(f"  ✅ 後處理已啟用 (模式: {CONFIG.get('postprocess_mode', 'minimal')})")
    logger.info(f"    - 移除小碎片 (Class1≥{CONFIG['min_size_class1']}, Class2≥{CONFIG['min_size_class2']}, Class3≥{CONFIG['min_size_class3']} voxels)")
    if CONFIG.get("fill_holes", True):
        logger.info(f"    - 填補孔洞")
    if CONFIG.get("use_morphology", False):
        logger.info(f"    - 形態學平滑")
else:
    logger.info(f"  ❌ 後處理未啟用")

logger.info(f"\n輸出位置:")
logger.info(f"  - 預測資料夾: {CONFIG['output_dir']}")
logger.info(f"  - 提交壓縮檔: {zip_path}")
logger.info("="*80)