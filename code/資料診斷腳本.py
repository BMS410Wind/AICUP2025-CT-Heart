import os
import json
import numpy as np
import matplotlib.pyplot as plt
from monai.transforms import LoadImage, EnsureChannelFirst
import nibabel as nib
from tqdm import tqdm

# ==================== 設定 ====================
DATA_JSON = r"D:\aicup\data_split\AICUP_training.json"
OUTPUT_DIR = r"D:\aicup\experiments\baseline\data_diagnosis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 載入資料列表 ====================
with open(DATA_JSON, 'r') as f:
    data_dict = json.load(f)

train_files = data_dict["training"]
val_files = data_dict["validation"]

print(f"訓練樣本數: {len(train_files)}")
print(f"驗證樣本數: {len(val_files)}")

# ==================== 診斷函數 ====================
def diagnose_single_file(image_path, label_path):
    """診斷單個檔案"""
    try:
        # 載入影像
        img = nib.load(image_path)
        img_data = img.get_fdata()
        
        # 載入標籤
        label = nib.load(label_path)
        label_data = label.get_fdata()
        
        # 基本資訊
        info = {
            'image_path': image_path,
            'label_path': label_path,
            'image_shape': img_data.shape,
            'label_shape': label_data.shape,
            'image_dtype': img_data.dtype,
            'label_dtype': label_data.dtype,
            'image_range': (img_data.min(), img_data.max()),
            'label_unique': np.unique(label_data),
        }
        
        # 標籤分布
        label_dist = {}
        for val in info['label_unique']:
            count = np.sum(label_data == val)
            percentage = count / label_data.size * 100
            label_dist[int(val)] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        info['label_distribution'] = label_dist
        
        return info, img_data, label_data
    
    except Exception as e:
        print(f"錯誤: {e}")
        return None, None, None

def visualize_sample(img_data, label_data, save_path, title="Sample"):
    """視覺化單個樣本"""
    if img_data is None or label_data is None:
        return
    
    # 選擇中間切片
    z_slice = img_data.shape[2] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 影像
    axes[0, 0].imshow(img_data[:, :, z_slice], cmap='gray')
    axes[0, 0].set_title(f'Image - Axial (Z={z_slice})', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_data[:, img_data.shape[1]//2, :], cmap='gray')
    axes[0, 1].set_title('Image - Coronal', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(img_data[img_data.shape[0]//2, :, :], cmap='gray')
    axes[0, 2].set_title('Image - Sagittal', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # 標籤
    axes[1, 0].imshow(label_data[:, :, z_slice], cmap='tab10', vmin=0, vmax=3)
    axes[1, 0].set_title(f'Label - Axial (Z={z_slice})', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(label_data[:, label_data.shape[1]//2, :], cmap='tab10', vmin=0, vmax=3)
    axes[1, 1].set_title('Label - Coronal', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(label_data[label_data.shape[0]//2, :, :], cmap='tab10', vmin=0, vmax=3)
    axes[1, 2].set_title('Label - Sagittal', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def comprehensive_diagnosis(files_list, name="Dataset", max_samples=10):
    """全面診斷資料集"""
    print(f"\n{'='*80}")
    print(f"診斷 {name}")
    print(f"{'='*80}\n")
    
    all_info = []
    all_label_distributions = []
    
    for i, file_info in enumerate(tqdm(files_list[:max_samples], desc=f"檢查 {name}")):
        image_path = file_info['image']
        label_path = file_info['label']
        
        info, img_data, label_data = diagnose_single_file(image_path, label_path)
        
        if info is None:
            print(f"⚠️  樣本 {i} 載入失敗")
            continue
        
        all_info.append(info)
        all_label_distributions.append(info['label_distribution'])
        
        # 輸出詳細資訊
        print(f"\n樣本 {i}:")
        print(f"  影像路徑: {image_path}")
        print(f"  標籤路徑: {label_path}")
        print(f"  影像形狀: {info['image_shape']}")
        print(f"  標籤形狀: {info['label_shape']}")
        print(f"  影像值範圍: [{info['image_range'][0]:.2f}, {info['image_range'][1]:.2f}]")
        print(f"  標籤唯一值: {info['label_unique']}")
        print(f"  標籤分布:")
        for cls, dist in info['label_distribution'].items():
            print(f"    Class {cls}: {dist['count']:,} pixels ({dist['percentage']:.2f}%)")
        
        # 視覺化前3個樣本
        if i < 3:
            save_path = os.path.join(OUTPUT_DIR, f"{name}_sample_{i}.png")
            visualize_sample(img_data, label_data, save_path, 
                           title=f"{name} - Sample {i}")
            print(f"  ✅ 視覺化已儲存: {save_path}")
    
    # 統計分析
    print(f"\n{'='*80}")
    print(f"{name} 整體統計")
    print(f"{'='*80}")
    
    # 檢查標籤是否一致
    all_unique_labels = set()
    for info in all_info:
        all_unique_labels.update(info['label_unique'])
    
    print(f"\n所有樣本中出現的標籤值: {sorted(all_unique_labels)}")
    
    # 檢查是否有問題
    expected_labels = {0, 1, 2, 3}
    if all_unique_labels != expected_labels:
        print(f"⚠️  警告: 標籤值不符合預期！")
        print(f"   預期: {expected_labels}")
        print(f"   實際: {all_unique_labels}")
        print(f"   差異: {all_unique_labels.symmetric_difference(expected_labels)}")
    else:
        print(f"✅ 標籤值正確")
    
    # 計算平均類別分布
    avg_distribution = {0: [], 1: [], 2: [], 3: []}
    for dist in all_label_distributions:
        for cls in range(4):
            if cls in dist:
                avg_distribution[cls].append(dist[cls]['percentage'])
            else:
                avg_distribution[cls].append(0.0)
    
    print(f"\n平均類別分布:")
    for cls in range(4):
        if avg_distribution[cls]:
            mean_pct = np.mean(avg_distribution[cls])
            std_pct = np.std(avg_distribution[cls])
            print(f"  Class {cls}: {mean_pct:.2f}% (±{std_pct:.2f}%)")
        else:
            print(f"  Class {cls}: 0.00% (不存在)")
    
    # 檢查類別不平衡
    print(f"\n類別平衡檢查:")
    class_percentages = [np.mean(avg_distribution[cls]) for cls in range(4) if avg_distribution[cls]]
    if class_percentages:
        max_pct = max(class_percentages)
        min_pct = min([p for p in class_percentages if p > 0])
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        print(f"  最大類別占比: {max_pct:.2f}%")
        print(f"  最小類別占比: {min_pct:.2f}%")
        print(f"  不平衡比例: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 100:
            print(f"  ⚠️  警告: 類別極度不平衡！建議使用加權損失函數")
        elif imbalance_ratio > 10:
            print(f"  ⚠️  注意: 類別不平衡較嚴重")
        else:
            print(f"  ✅ 類別平衡度尚可")
    
    # 視覺化類別分布
    plot_class_distribution(avg_distribution, name)
    
    return all_info

def plot_class_distribution(avg_distribution, name):
    """繪製類別分布圖"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 箱線圖
    ax = axes[0]
    data_to_plot = [avg_distribution[cls] for cls in range(4)]
    bp = ax.boxplot(data_to_plot, labels=[f'Class {i}' for i in range(4)],
                    patch_artist=True, showmeans=True)
    
    for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} - Class Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # 平均值條形圖
    ax = axes[1]
    means = [np.mean(avg_distribution[cls]) if avg_distribution[cls] else 0 for cls in range(4)]
    stds = [np.std(avg_distribution[cls]) if avg_distribution[cls] else 0 for cls in range(4)]
    
    bars = ax.bar(range(4), means, yerr=stds, capsize=5, 
                  color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'],
                  edgecolor='black', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{name} - Average Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(4))
    ax.set_xticklabels([f'Class {i}' for i in range(4)])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    
    # 標註數值
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(i, mean + std, f'{mean:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 類別分布圖已儲存: {name}_distribution.png")

# ==================== 執行診斷 ====================
print("開始診斷資料集...")
print(f"輸出目錄: {OUTPUT_DIR}")

# 診斷訓練集
train_info = comprehensive_diagnosis(train_files, "Training", max_samples=10)

# 診斷驗證集
val_info = comprehensive_diagnosis(val_files, "Validation", max_samples=10)

# 生成診斷報告
report_path = os.path.join(OUTPUT_DIR, "diagnosis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("資料診斷報告\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"訓練樣本數: {len(train_files)}\n")
    f.write(f"驗證樣本數: {len(val_files)}\n\n")
    
    f.write("請檢查以下項目:\n")
    f.write("1. 標籤值是否為 0, 1, 2, 3\n")
    f.write("2. 是否有極度的類別不平衡\n")
    f.write("3. 前景類別（1, 2, 3）的占比是否過低\n")
    f.write("4. 影像和標籤的形狀是否匹配\n\n")
    
    f.write("如果發現問題，可能的解決方案:\n")
    f.write("- 類別不平衡 → 使用加權損失函數\n")
    f.write("- 前景過少 → 增加 pos 權重在 RandCropByPosNegLabeld\n")
    f.write("- 標籤值錯誤 → 檢查資料預處理流程\n")

print(f"\n✅ 診斷完成！報告已儲存至: {report_path}")
print(f"請檢查 {OUTPUT_DIR} 目錄下的所有圖片和報告")