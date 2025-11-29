import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from monai.networks.nets import DynUNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRangePercentilesd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    ToTensord, RandAffined, RandGaussianNoised, 
    RandGaussianSmoothd, RandAdjustContrastd, RandScaleIntensityd,
    SpatialPadd,DivisiblePadd
)
from monai.data import DataLoader, CacheDataset, decollate_batch, pad_list_data_collate
from monai.inferers import sliding_window_inference
from tqdm import tqdm
import logging
import multiprocessing

# ==================== CONFIG ====================
CONFIG = {
    "data_json": r"D:\\aicup\\data_split\\AICUP_training.json",
    "output_dir": r"D:\\aicup\\experiments\\nnunet_all_classes",
    "in_channels": 1,
    "out_channels": 4,
    "spatial_dims": 3,
    "kernels": [[3,3,3]]*6,
    "strides": [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,1]],
    "deep_supervision": False,
    "deep_supr_num": 0,
    "batch_size": 2,
    "num_epochs": 1000,
    "learning_rate": 1e-2,
    "weight_decay": 3e-5,
    "val_interval": 5,
    "target_spacing": (1.5,1.5,2.0),
    "roi_size": (128,128,128),
    "num_samples": 2,
    "class_weights": {0:1.0,1:1.0,2:275.4,3:1377.1},
    "patience": 100,
    "num_workers": 4,
    "cache_num_workers": 0,
    "cache_rate": 1.0,
    "use_amp": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "enable_dataset_analysis": False,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# ==================== logger ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(CONFIG["output_dir"], 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== utilities ====================

def analyze_dataset(data_files):
    from monai.transforms import LoadImage
    loader = LoadImage(image_only=True)
    class_stats = {0:[],1:[],2:[],3:[]}
    for item in tqdm(data_files, desc="分析數據集"):
        try:
            label = loader(item['label'])
            if isinstance(label, torch.Tensor):
                label = label.numpy()
            for class_id in range(4):
                count = np.sum(label==class_id)
                class_stats[class_id].append(count)
        except Exception as e:
            logger.warning(f"無法讀取 {item['label']}: {e}")
    return class_stats


def get_transforms(mode="train"):

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="LPS"),

        Spacingd(
            keys=["image", "label"],
            pixdim=CONFIG["target_spacing"],
            mode=("bilinear", "nearest")
        ),

        # 確保空間尺寸可整除（DynUNet 要求）
        DivisiblePadd(keys=["image", "label"], k=16),
    ]

    # ---- Train transforms ----
    if mode == "train":
        train_tf = base + [

            # 只保留 1 次 RandCrop（絕對不要重複）
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=CONFIG["roi_size"],
                pos=1, neg=1,
                num_samples=CONFIG["num_samples"],
                allow_smaller=False
            ),

            # 幾何增強
            RandAffined(
                keys=["image", "label"],
                prob=0.9,
                rotate_range=(0.52, 0.52, 0.52),
                scale_range=(0.3, 0.3, 0.3),
                mode=("bilinear", "nearest"),
                padding_mode="border",
            ),

            # 強度增強
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.1),
            RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5,1.5), sigma_y=(0.5,1.5), sigma_z=(0.5,1.5)),
            RandAdjustContrastd(keys=["image"], prob=0.15, gamma=(0.7,1.5)),
            RandScaleIntensityd(keys=["image"], factors=0.25, prob=0.15),

            # 空間增強
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[1]),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=[2]),
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),

            ToTensord(keys=["image", "label"]),
        ]
        return Compose(train_tf)

    # ---- Validation transforms ----
    else:
        return Compose(
            base + [
                ToTensord(keys=["image", "label"])
            ]
        )


def build_model(device):
    model = DynUNet(spatial_dims=CONFIG['spatial_dims'], in_channels=CONFIG['in_channels'], out_channels=CONFIG['out_channels'], kernel_size=CONFIG['kernels'], strides=CONFIG['strides'], upsample_kernel_size=CONFIG['kernels'][1:], norm_name='instance', deep_supervision=CONFIG['deep_supervision'], deep_supr_num=CONFIG['deep_supr_num'], res_block=True).to(device)
    return model


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(2,2, figsize=(16,12))
    axes[0,0].plot(history['epochs'], history['train_loss'], 'b-'); axes[0,0].set_title('Training Loss'); axes[0,0].grid(True,alpha=0.3)
    axes[0,1].plot(history['epochs'], history['train_dice'], 'b-', label='Train')
    if len(history['val_dice_mean'])>0:
        val_epochs = history['epochs'][::CONFIG['val_interval']][:len(history['val_dice_mean'])]
        axes[0,1].plot(val_epochs, history['val_dice_class1'], marker='o', label='C1')
        axes[0,1].plot(val_epochs, history['val_dice_class2'], marker='s', label='C2')
        axes[0,1].plot(val_epochs, history['val_dice_class3'], marker='^', label='C3')
        axes[0,1].plot(val_epochs, history['val_dice_mean'], linestyle='--', label='Mean')
        axes[0,1].legend()
    axes[1,0].plot(history['epochs'], history['learning_rate'], 'g-'); axes[1,0].set_yscale('log')
    if len(history['val_dice_mean'])>0:
        val_epochs = history['epochs'][::CONFIG['val_interval']][:len(history['val_dice_mean'])]
        axes[1,1].plot(val_epochs, history['val_dice_class1'], marker='o'); axes[1,1].plot(val_epochs, history['val_dice_class2'], marker='s'); axes[1,1].plot(val_epochs, history['val_dice_class3'], marker='^')
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()


# ==================== train_main ====================

def train_main():
    logger.info('='*80)
    logger.info('nnU-Net 風格訓練 - 預測所有類別 (1, 2, 3)')
    logger.info('='*80)

    with open(CONFIG['data_json'], 'r') as f:
        data_dict = json.load(f)
    train_files = data_dict['training']; val_files = data_dict['validation']
    logger.info(f"訓練樣本數: {len(train_files)}"); logger.info(f"驗證樣本數: {len(val_files)}")

    if CONFIG.get('enable_dataset_analysis', False):
        logger.info('\n分析數據集...')
        train_stats = analyze_dataset(train_files)
        logger.info('\n訓練集類別分布:')
        for class_id in range(4):
            class_names = ['背景','心臟','心包膜','鈣化']
            avg = np.mean(train_stats[class_id]); samples_with_class = sum(1 for x in train_stats[class_id] if x>0)
            logger.info(f"  Class {class_id} ({class_names[class_id]}): 平均 {avg:.0f} 像素/樣本, {samples_with_class}/{len(train_files)} 樣本含有此類別")

    train_transforms = get_transforms('train'); val_transforms = get_transforms('val')

    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=CONFIG['cache_rate'], num_workers=CONFIG['cache_num_workers'])
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=CONFIG['cache_rate'], num_workers=CONFIG['cache_num_workers'])

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=(CONFIG['device']=='cuda'), collate_fn=pad_list_data_collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=(CONFIG['device']=='cuda'), collate_fn=pad_list_data_collate)

    device = CONFIG['device']; model = build_model(device)
    logger.info(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")

    class_weights = torch.tensor([float(CONFIG['class_weights'][i]) for i in range(4)], dtype=torch.float32).to(device)
    loss_function = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=1.0, lambda_ce=1.0, weight=class_weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.99, weight_decay=CONFIG['weight_decay'], nesterov=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: CONFIG['learning_rate'] * (1 - epoch/CONFIG['num_epochs'])**0.9)

    scaler = GradScaler() if CONFIG['use_amp'] else None
    dice_metric = DiceMetric(include_background=False, reduction='mean_batch')

    history = {'train_loss':[], 'train_dice':[], 'val_dice_class1':[], 'val_dice_class2':[], 'val_dice_class3':[], 'val_dice_mean':[], 'learning_rate':[], 'epochs':[]}

    def train_epoch(epoch):
        model.train(); epoch_loss=0; step=0; dice_metric.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['num_epochs']}")
        for batch_data in pbar:
            step+=1
            inputs = batch_data['image'].to(device); labels = batch_data['label'].to(device)
            optimizer.zero_grad()
            if CONFIG['use_amp']:
                with autocast(enabled=True):
                    outputs = model(inputs)
                    if CONFIG['deep_supervision'] and isinstance(outputs, list):
                        loss = sum([0.5**i * loss_function(out, labels) for i,out in enumerate(outputs)])
                        outputs = outputs[0]
                    else:
                        loss = loss_function(outputs, labels)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 12); scaler.step(optimizer); scaler.update()
            else:
                outputs = model(inputs)
                if CONFIG['deep_supervision'] and isinstance(outputs, list):
                    loss = sum([0.5**i * loss_function(out, labels) for i,out in enumerate(outputs)])
                    outputs = outputs[0]
                else:
                    loss = loss_function(outputs, labels)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 12); optimizer.step()
            epoch_loss += loss.item()
            with torch.no_grad():
                pred = torch.argmax(outputs, dim=1, keepdim=True)
                pred_onehot = torch.zeros_like(outputs); pred_onehot.scatter_(1, pred.long(), 1)
                label_onehot = torch.zeros_like(outputs); label_onehot.scatter_(1, labels.long(), 1)
                dice_metric(y_pred=pred_onehot[:,1:], y=label_onehot[:,1:])
            current_dice = dice_metric.aggregate().item()
            pbar.set_postfix({'loss':f"{loss.item():.4f}", 'dice':f"{current_dice:.4f}", 'lr':f"{optimizer.param_groups[0]['lr']:.6f}"})
        epoch_loss /= step; train_dice = dice_metric.aggregate().item(); dice_metric.reset(); return epoch_loss, train_dice

    def validate(epoch):
        model.eval(); dice_metric.reset(); dice_scores_per_class={1:[],2:[],3:[]}
        with torch.no_grad():
            for val_data in tqdm(val_loader, desc='Validating'):
                val_inputs = val_data['image'].to(device); val_labels = val_data['label'].to(device)
                val_outputs = sliding_window_inference(val_inputs, CONFIG['roi_size'], sw_batch_size=4, predictor=model, overlap=0.6, mode='gaussian')
                if isinstance(val_outputs, list): val_outputs = val_outputs[0]
                val_pred = torch.argmax(val_outputs, dim=1, keepdim=True)
                pred_onehot = torch.zeros_like(val_outputs); pred_onehot.scatter_(1, val_pred.long(), 1)
                label_onehot = torch.zeros_like(val_outputs); label_onehot.scatter_(1, val_labels.long(), 1)
                for class_id in [1,2,3]:
                    pred_class = pred_onehot[:, class_id:class_id+1]; label_class = label_onehot[:, class_id:class_id+1]
                    if label_class.sum()>0:
                        intersection = (pred_class * label_class).sum(); union = pred_class.sum() + label_class.sum(); dice = (2.0*intersection)/(union+1e-8)
                        dice_scores_per_class[class_id].append(dice.item())
        val_dice_class1 = np.mean(dice_scores_per_class[1]) if dice_scores_per_class[1] else 0.0
        val_dice_class2 = np.mean(dice_scores_per_class[2]) if dice_scores_per_class[2] else 0.0
        val_dice_class3 = np.mean(dice_scores_per_class[3]) if dice_scores_per_class[3] else 0.0
        val_dice_mean = np.mean([val_dice_class1, val_dice_class2, val_dice_class3])
        return val_dice_class1, val_dice_class2, val_dice_class3, val_dice_mean

    logger.info('='*80); logger.info('開始訓練 - nnU-Net All Classes'); logger.info('='*80)
    logger.info(f"Device: {device}"); logger.info(f"Training samples: {len(train_files)}"); logger.info(f"Validation samples: {len(val_files)}"); logger.info(f"Effective patches per epoch: {len(train_files)*CONFIG['num_samples']}")

    best_dice=0.0; best_epoch=0; patience_counter=0
    for epoch in range(1, CONFIG['num_epochs']+1):
        train_loss, train_dice = train_epoch(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss); history['train_dice'].append(train_dice); history['learning_rate'].append(current_lr); history['epochs'].append(epoch)
        logger.info(f"Epoch {epoch} - Loss: {train_loss:.4f} - Train Dice: {train_dice:.4f} - LR: {current_lr:.6f}")
        if epoch % CONFIG['val_interval'] == 0:
            val_dice_c1, val_dice_c2, val_dice_c3, val_dice_mean = validate(epoch)
            history['val_dice_class1'].append(val_dice_c1); history['val_dice_class2'].append(val_dice_c2); history['val_dice_class3'].append(val_dice_c3); history['val_dice_mean'].append(val_dice_mean)
            logger.info(f"Epoch {epoch} - Validation Dice:"); logger.info(f"  Class 1 (Heart): {val_dice_c1:.4f}"); logger.info(f"  Class 2 (Pericardium): {val_dice_c2:.4f}"); logger.info(f"  Class 3 (Calcium): {val_dice_c3:.4f}"); logger.info(f"  Mean: {val_dice_mean:.4f}")
            if val_dice_mean > best_dice:
                best_dice = val_dice_mean; best_epoch = epoch; patience_counter=0
                torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'val_dice_mean':val_dice_mean,'val_dice_class1':val_dice_c1,'val_dice_class2':val_dice_c2,'val_dice_class3':val_dice_c3,'config':CONFIG,'history':history}, os.path.join(CONFIG['output_dir'], 'best_model_nnunet.pth'))
                logger.info(f"✅ 保存最佳模型 (Mean Dice: {val_dice_mean:.4f})")
            else:
                patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"Early stopping at epoch {epoch}"); break
            plot_training_history(history, os.path.join(CONFIG['output_dir'], 'training_curves.png'))
        scheduler.step()

    logger.info('='*80); logger.info(f"訓練完成！最佳 Mean Dice: {best_dice:.4f} (Epoch {best_epoch})"); logger.info('='*80)
    with open(os.path.join(CONFIG['output_dir'], 'final_history.json'), 'w') as f: json.dump(history, f, indent=4)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_main()
