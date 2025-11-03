import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体和科研配色
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 科研配色方案
COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8', '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']
sns.set_palette(COLORS)

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class ContainerDamageAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.images_path = os.path.join(data_path, 'images')
        self.labels_path = os.path.join(data_path, 'labels')
        self.classes = self._load_classes()
        self.data_info = None

    def _load_classes(self):
        """加载类别信息"""
        classes_file = os.path.join(self.data_path, 'classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines()]
            return classes
        else:
            # 根据问题描述，类别是dent、hole、rusty
            return ['dent', 'hole', 'rusty']

    def analyze_dataset(self):
        """进行全面的数据探索性分析"""
        print("开始数据探索性分析...")

        # 收集数据集信息 - 只处理存在的目录
        train_images = os.listdir(os.path.join(self.images_path, 'train'))
        test_images = os.listdir(os.path.join(self.images_path, 'test'))

        print(f"训练集图像数量: {len(train_images)}")
        print(f"测试集图像数量: {len(test_images)}")
        print(f"总图像数量: {len(train_images) + len(test_images)}")
        print(f"类别: {self.classes}")

        # 分析标签分布
        self._analyze_label_distribution()

        # 分析图像尺寸和特征
        self._analyze_image_characteristics()

        # 可视化分析结果
        self._create_visualizations()

        return self.data_info

    def _analyze_label_distribution(self):
        """分析标签分布"""
        print("\n分析标签分布...")

        damage_data = []

        # 分析训练集标签
        train_labels_path = os.path.join(self.labels_path, 'train')
        for label_file in os.listdir(train_labels_path):
            if label_file.endswith('.txt'):
                file_path = os.path.join(train_labels_path, label_file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                if lines:  # 有损伤
                    for line in lines:
                        class_id = int(line.strip().split()[0])
                        damage_data.append({
                            'image_id': label_file.replace('.txt', ''),
                            'class_id': class_id,
                            'class_name': self.classes[class_id],
                            'has_damage': 1
                        })
                else:  # 无损伤
                    damage_data.append({
                        'image_id': label_file.replace('.txt', ''),
                        'class_id': -1,
                        'class_name': 'no_damage',
                        'has_damage': 0
                    })

        self.data_info = pd.DataFrame(damage_data)

        # 打印统计信息
        damage_count = len(self.data_info[self.data_info['has_damage'] == 1])
        no_damage_count = len(self.data_info[self.data_info['has_damage'] == 0])
        total_count = damage_count + no_damage_count

        print(f"有损伤图像数量: {damage_count}")
        print(f"无损伤图像数量: {no_damage_count}")
        print(f"损伤比例: {damage_count / total_count * 100:.2f}%" if total_count > 0 else "损伤比例: 0.00%")

        if damage_count > 0:
            class_dist = self.data_info[self.data_info['has_damage'] == 1]['class_name'].value_counts()
            print("\n各类别损伤分布:")
            for class_name, count in class_dist.items():
                print(f"  {class_name}: {count} ({count / damage_count * 100:.2f}%)")

        # 输出部分表格数据
        print("\n数据表格预览:")
        print(self.data_info.head(10))

        # 保存数据统计表格
        stats_table = pd.DataFrame({
            '统计项': ['总图像数', '有损伤图像', '无损伤图像', '损伤比例'],
            '数值': [total_count, damage_count, no_damage_count, f"{damage_count / total_count * 100:.2f}%"]
        })
        stats_table.to_csv('dataset_statistics.csv', index=False, encoding='utf-8-sig')
        print("数据统计已保存到 dataset_statistics.csv")

    def _analyze_image_characteristics(self):
        """分析图像特征"""
        print("\n分析图像特征...")

        sample_images = []
        train_images_dir = os.path.join(self.images_path, 'train')

        # 随机选择一些图像进行分析
        image_files = [f for f in os.listdir(train_images_dir) if f.endswith('.jpg')][:100]

        widths, heights = [], []
        mean_intensities, std_intensities = [], []

        for img_file in tqdm(image_files, desc="分析图像特征"):
            img_path = os.path.join(train_images_dir, img_file)
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)

                # 转换为numpy数组分析强度
                img_array = np.array(img)
                mean_intensities.append(np.mean(img_array))
                std_intensities.append(np.std(img_array))

            except Exception as e:
                print(f"处理图像 {img_file} 时出错: {e}")

        self.image_stats = {
            'widths': widths,
            'heights': heights,
            'mean_intensity': mean_intensities,
            'std_intensity': std_intensities
        }

        print(f"图像平均尺寸: {np.mean(widths):.1f} x {np.mean(heights):.1f}")
        print(f"图像尺寸范围: {np.min(widths)}-{np.max(widths)} x {np.min(heights)}-{np.max(heights)}")
        print(f"平均亮度: {np.mean(mean_intensities):.2f}")
        print(f"平均对比度: {np.mean(std_intensities):.2f}")

        # 保存图像特征统计
        image_features_df = pd.DataFrame({
            '宽度': widths,
            '高度': heights,
            '平均亮度': mean_intensities,
            '对比度': std_intensities
        })
        image_features_df.to_csv('image_features_statistics.csv', index=False, encoding='utf-8-sig')
        print("图像特征统计已保存到 image_features_statistics.csv")

    def _create_visualizations(self):
        """创建数据可视化 - 每个子图单独保存"""
        print("\n创建可视化图表...")

        # 1. 损伤分布饼图
        plt.figure(figsize=(10, 8))
        damage_counts = self.data_info['has_damage'].value_counts()

        # 动态生成标签和颜色
        if len(damage_counts) == 1:
            # 只有有损伤的情况
            labels = ['有损伤']
            colors = [COLORS[1]]
        else:
            # 有损伤和无损伤都有
            labels = ['无损伤', '有损伤']
            colors = COLORS[:2]

        plt.pie(damage_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('损伤图像分布', fontsize=16, fontweight='bold')
        plt.savefig('damage_distribution_pie.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. 损伤类别分布
        plt.figure(figsize=(10, 8))
        if len(self.data_info[self.data_info['has_damage'] == 1]) > 0:
            damage_class_dist = self.data_info[self.data_info['has_damage'] == 1]['class_name'].value_counts()
            x_pos = np.arange(len(damage_class_dist))
            plt.bar(x_pos, damage_class_dist.values,
                    color=COLORS[2:2 + len(damage_class_dist)], alpha=0.8)
            plt.xticks(x_pos, damage_class_dist.index, rotation=45)
            plt.xlabel('损伤类别')
            plt.ylabel('数量')
            plt.title('损伤类别分布', fontsize=16, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig('damage_class_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 3. 图像尺寸分布
        plt.figure(figsize=(10, 8))
        plt.scatter(self.image_stats['widths'], self.image_stats['heights'],
                    alpha=0.6, color=COLORS[3], s=50)
        plt.xlabel('宽度 (像素)')
        plt.ylabel('高度 (像素)')
        plt.title('图像尺寸分布', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('image_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. 图像亮度分布
        plt.figure(figsize=(10, 8))
        plt.hist(self.image_stats['mean_intensity'], bins=30, color=COLORS[4], alpha=0.7, edgecolor='black')
        plt.xlabel('平均亮度')
        plt.ylabel('频数')
        plt.title('图像亮度分布', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('image_brightness_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 5. 图像对比度分布
        plt.figure(figsize=(10, 8))
        plt.hist(self.image_stats['std_intensity'], bins=30, color=COLORS[5], alpha=0.7, edgecolor='black')
        plt.xlabel('对比度 (标准差)')
        plt.ylabel('频数')
        plt.title('图像对比度分布', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.savefig('image_contrast_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 6. 每图像损伤数量分布
        plt.figure(figsize=(10, 8))
        if len(self.data_info) > 0:
            damage_per_image = self.data_info.groupby('image_id')['class_id'].count()
            damage_count_dist = damage_per_image.value_counts().sort_index()
            plt.bar(damage_count_dist.index, damage_count_dist.values,
                    color=COLORS[6], alpha=0.8, edgecolor='black')
            plt.xlabel('每图像损伤数量')
            plt.ylabel('图像数量')
            plt.title('每图像损伤数量分布', fontsize=16, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig('damage_per_image_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()

        # 创建综合可视化（可选）
        self._create_comprehensive_visualization()

        # 创建样本图像展示
        self._display_sample_images()

    def _create_comprehensive_visualization(self):
        """创建综合可视化图表（保留原综合图）"""
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        damage_counts = self.data_info['has_damage'].value_counts()
        if len(damage_counts) == 1:
            labels = ['有损伤']
            colors = [COLORS[1]]
        else:
            labels = ['无损伤', '有损伤']
            colors = COLORS[:2]
        plt.pie(damage_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title('损伤图像分布')

        plt.subplot(2, 3, 2)
        if len(self.data_info[self.data_info['has_damage'] == 1]) > 0:
            damage_class_dist = self.data_info[self.data_info['has_damage'] == 1]['class_name'].value_counts()
            x_pos = np.arange(len(damage_class_dist))
            plt.bar(x_pos, damage_class_dist.values,
                    color=COLORS[2:2 + len(damage_class_dist)])
            plt.xticks(x_pos, damage_class_dist.index, rotation=45)
            plt.title('损伤类别分布')

        plt.subplot(2, 3, 3)
        plt.scatter(self.image_stats['widths'], self.image_stats['heights'], alpha=0.6, color=COLORS[3])
        plt.xlabel('宽度 (像素)')
        plt.ylabel('高度 (像素)')
        plt.title('图像尺寸分布')

        plt.subplot(2, 3, 4)
        plt.hist(self.image_stats['mean_intensity'], bins=30, color=COLORS[4], alpha=0.7)
        plt.xlabel('平均亮度')
        plt.ylabel('频数')
        plt.title('图像亮度分布')

        plt.subplot(2, 3, 5)
        plt.hist(self.image_stats['std_intensity'], bins=30, color=COLORS[5], alpha=0.7)
        plt.xlabel('对比度 (标准差)')
        plt.ylabel('频数')
        plt.title('图像对比度分布')

        plt.subplot(2, 3, 6)
        if len(self.data_info) > 0:
            damage_per_image = self.data_info.groupby('image_id')['class_id'].count()
            damage_count_dist = damage_per_image.value_counts().sort_index()
            plt.bar(damage_count_dist.index, damage_count_dist.values, color=COLORS[6])
            plt.xlabel('每图像损伤数量')
            plt.ylabel('图像数量')
            plt.title('每图像损伤数量分布')

        plt.tight_layout()
        plt.savefig('data_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _display_sample_images(self):
        """显示样本图像"""
        print("\n显示样本图像...")

        train_images_dir = os.path.join(self.images_path, 'train')
        train_labels_dir = os.path.join(self.labels_path, 'train')

        # 获取有损伤的样本
        damage_samples = self.data_info['image_id'].unique()[:8]

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for i, img_id in enumerate(damage_samples[:8]):
            img_path = os.path.join(train_images_dir, f"{img_id}.jpg")
            label_path = os.path.join(train_labels_dir, f"{img_id}.txt")

            try:
                img = Image.open(img_path)
                axes[i].imshow(img)

                # 读取并绘制边界框
                with open(label_path, 'r') as f:
                    lines = f.readlines()

                img_width, img_height = img.size
                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())

                    # 转换为像素坐标
                    x_center_px = x_center * img_width
                    y_center_px = y_center * img_height
                    width_px = width * img_width
                    height_px = height * img_height

                    # 计算边界框坐标
                    x1 = x_center_px - width_px / 2
                    y1 = y_center_px - height_px / 2
                    x2 = x_center_px + width_px / 2
                    y2 = y_center_px + height_px / 2

                    # 绘制边界框
                    rect = plt.Rectangle((x1, y1), width_px, height_px,
                                         fill=False, edgecolor='red', linewidth=2)
                    axes[i].add_patch(rect)

                    # 添加类别标签
                    axes[i].text(x1, y1 - 10, self.classes[int(class_id)],
                                 color='red', fontsize=10, weight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

                damage_types = self.data_info[self.data_info['image_id'] == img_id]['class_name'].unique()
                axes[i].set_title(f'图像 {img_id}\n{", ".join(damage_types)}', fontsize=12)
                axes[i].axis('off')

            except Exception as e:
                print(f"显示图像 {img_id} 时出错: {e}")
                axes[i].axis('off')

        # 如果不足8个图像，隐藏多余的子图
        for j in range(len(damage_samples), 8):
            axes[j].axis('off')

        plt.suptitle('样本图像展示 (红色框表示损伤区域)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('sample_images_display.png', dpi=300, bbox_inches='tight')
        plt.show()


# 定义创新的分类模型 - 修改为多类别分类
class DamageClassificationModel(nn.Module):
    def __init__(self, num_classes=3, use_attention=True):  # 默认为3个类别
        super(DamageClassificationModel, self).__init__()

        # 使用预训练的EfficientNet作为基础网络
        self.backbone = models.efficientnet_b0(pretrained=True)

        # 获取特征维度
        in_features = self.backbone.classifier[1].in_features

        # 添加注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(in_features, in_features // 4),
                nn.ReLU(inplace=True),
                nn.Linear(in_features // 4, in_features),
                nn.Sigmoid()
            )

        # 分类器 - 输出3个类别
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)

        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights

        output = self.classifier(features)
        return output


# 自定义数据集类 - 修改为多类别分类
class ContainerDamageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, data_info, transform=None, num_classes=3):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.data_info = data_info
        self.transform = transform
        self.num_classes = num_classes

        # 获取唯一的图像ID和它们的主要类别
        self.image_data = self._prepare_image_data()

    def _prepare_image_data(self):
        """为每个图像准备主要类别标签"""
        image_data = []
        for image_id in self.data_info['image_id'].unique():
            image_damages = self.data_info[self.data_info['image_id'] == image_id]

            # 如果图像有多个损伤，选择最频繁的类别作为主要类别
            if len(image_damages) > 0:
                # 过滤掉无损伤的情况（如果有）
                damages = image_damages[image_damages['has_damage'] == 1]
                if len(damages) > 0:
                    # 选择最频繁的损伤类别
                    main_class = damages['class_id'].mode()
                    if len(main_class) > 0:
                        class_label = main_class.iloc[0]
                    else:
                        class_label = damages['class_id'].iloc[0]
                else:
                    # 如果没有损伤，标记为0（第一个类别）
                    class_label = 0
            else:
                class_label = 0  # 默认类别

            image_data.append({
                'image_id': image_id,
                'class_label': class_label
            })

        return pd.DataFrame(image_data)

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_id = self.image_data.iloc[idx]['image_id']
        class_label = self.image_data.iloc[idx]['class_label']
        img_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 应用数据增强
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(class_label, dtype=torch.long)


def train_and_evaluate_model():
    """训练和评估分类模型"""
    print("开始训练分类模型...")

    # 数据路径
    data_path = r"D:\桌面\2025年MathorCup大数据挑战赛-赛道A初赛\数据集3713"

    # 初始化分析器
    analyzer = ContainerDamageAnalyzer(data_path)
    data_info = analyzer.analyze_dataset()

    # 检查数据分布
    print(f"\n数据集分布检查:")
    print(f"总图像数量: {len(data_info['image_id'].unique())}")
    print(f"有损伤图像数量: {len(data_info[data_info['has_damage'] == 1]['image_id'].unique())}")
    print(f"无损伤图像数量: {len(data_info[data_info['has_damage'] == 0]['image_id'].unique())}")

    # 检查类别平衡性
    unique_classes = data_info[data_info['has_damage'] == 1]['class_id'].unique()
    print(f"存在的损伤类别: {sorted(unique_classes)}")
    print(f"类别数量: {len(unique_classes)}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 准备数据
    train_images_dir = os.path.join(data_path, 'images', 'train')
    train_labels_dir = os.path.join(data_path, 'labels', 'train')

    # 划分训练集和验证集
    train_ids, val_ids = train_test_split(
        data_info['image_id'].unique(),
        test_size=0.2,
        random_state=42,
        stratify=data_info.groupby('image_id')['has_damage'].first()
    )

    train_info = data_info[data_info['image_id'].isin(train_ids)]
    val_info = data_info[data_info['image_id'].isin(val_ids)]

    # 使用多类别分类
    num_classes = len(analyzer.classes)  # 3个类别
    print(f"使用多类别分类，类别数量: {num_classes}")

    train_dataset = ContainerDamageDataset(train_images_dir, train_labels_dir, train_info, train_transform, num_classes)
    val_dataset = ContainerDamageDataset(train_images_dir, train_labels_dir, val_info, val_transform, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = DamageClassificationModel(num_classes=num_classes, use_attention=True)
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # 训练模型 - 增加到30轮
    num_epochs = 30
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                probabilities = torch.softmax(outputs, dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })

        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_damage_classification_model.pth')
            print(f'  保存最佳模型，验证准确率: {best_val_acc:.2f}%')

    # 绘制训练曲线
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color=COLORS[0], linewidth=2, marker='o')
    plt.plot(val_losses, label='验证损失', color=COLORS[1], linewidth=2, marker='s')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率', color=COLORS[2], linewidth=2, marker='o')
    plt.plot(val_accs, label='验证准确率', color=COLORS[3], linewidth=2, marker='s')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.title('训练和验证准确率曲线', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_accuracy_curve.png', dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.savefig('training_curves_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 最终评估
    if os.path.exists('best_damage_classification_model.pth'):
        model.load_state_dict(torch.load('best_damage_classification_model.pth'))
    model.eval()

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    # 生成类别名称
    class_names = analyzer.classes

    print("\n模型评估结果:")
    print(classification_report(all_labels, all_predictions,
                                target_names=class_names))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 创建结果表格
    accuracy = 100 * (all_predictions == all_labels).mean()

    # 计算每个类别的精确率、召回率和F1分数
    class_report = classification_report(all_labels, all_predictions,
                                         target_names=class_names, output_dict=True)

    results_data = []
    for i, class_name in enumerate(class_names):
        precision = class_report[class_name]['precision'] * 100
        recall = class_report[class_name]['recall'] * 100
        f1 = class_report[class_name]['f1-score'] * 100
        results_data.append({
            '类别': class_name,
            '精确率': f"{precision:.2f}%",
            '召回率': f"{recall:.2f}%",
            'F1分数': f"{f1:.2f}%"
        })

    # 添加总体指标
    results_data.append({
        '类别': '总体',
        '精确率': f"{class_report['macro avg']['precision'] * 100:.2f}%",
        '召回率': f"{class_report['macro avg']['recall'] * 100:.2f}%",
        'F1分数': f"{class_report['macro avg']['f1-score'] * 100:.2f}%"
    })

    results_df = pd.DataFrame(results_data)

    print("\n详细评估指标:")
    print(results_df.to_string(index=False))

    # 保存训练过程数据
    training_history = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    })
    training_history.to_csv('training_history.csv', index=False, encoding='utf-8-sig')
    print("训练历史已保存到 training_history.csv")

    # 保存结果表格
    results_df.to_csv('classification_results.csv', index=False, encoding='utf-8-sig')

    return model, results_df, class_names


def generate_test_predictions(model, class_names):
    """生成测试集预测结果"""
    print("\n生成测试集预测结果...")

    data_path = r"D:\桌面\2025年MathorCup大数据挑战赛-赛道A初赛\数据集3713"
    test_images_dir = os.path.join(data_path, 'images', 'test')

    # 测试集预处理
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # 获取测试图像
    test_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]

    predictions = []

    with torch.no_grad():
        for img_file in tqdm(test_images, desc="处理测试图像"):
            img_path = os.path.join(test_images_dir, img_file)
            image_id = img_file.replace('.jpg', '')

            try:
                image = Image.open(img_path).convert('RGB')
                image = test_transform(image).unsqueeze(0).to(device)

                output = model(image)
                probability = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = probability[0, predicted_class].item()

                predictions.append({
                    'image_id': image_id,
                    'predicted_class': predicted_class,
                    'class_name': class_names[predicted_class],
                    'confidence': confidence
                })

            except Exception as e:
                print(f"处理测试图像 {img_file} 时出错: {e}")
                predictions.append({
                    'image_id': image_id,
                    'predicted_class': 0,  # 默认类别
                    'class_name': class_names[0],
                    'confidence': 0.5
                })

    # 创建结果DataFrame
    results_df = pd.DataFrame(predictions)

    # 保存预测结果详情
    results_df.to_csv('test_predictions_details.csv', index=False, encoding='utf-8-sig')
    print("测试集预测详情已保存到 test_predictions_details.csv")

    # 保存为要求的格式 - 修改为多类别输出
    submission_df = pd.DataFrame({
        'image_id': results_df['image_id'],
        'class_id': results_df['predicted_class'],  # 多类别ID
        'x_center': 0.5,  # 问题1不需要定位信息，设为默认值
        'y_center': 0.5,
        'width': 0.1,
        'height': 0.1
    })

    submission_df.to_csv('test_result.csv', index=False)
    print(f"测试集预测结果已保存到 test_result.csv")

    # 输出预测统计
    pred_stats = results_df['class_name'].value_counts().reset_index()
    pred_stats.columns = ['预测类别', '数量']
    print("\n预测结果统计:")
    print(pred_stats.to_string(index=False))
    pred_stats.to_csv('prediction_statistics.csv', index=False, encoding='utf-8-sig')

    return submission_df, results_df


# 执行完整流程
if __name__ == "__main__":
    try:
        # 训练和评估模型
        trained_model, results, class_names = train_and_evaluate_model()

        # 生成测试集预测
        test_predictions, test_details = generate_test_predictions(trained_model, class_names)

        print("\n=== 任务完成 ===")
        print("已生成以下文件:")
        print("1. 数据探索性分析:")
        print("   - damage_distribution_pie.png - 损伤分布饼图")
        print("   - damage_class_distribution.png - 损伤类别分布")
        print("   - image_size_distribution.png - 图像尺寸分布")
        print("   - image_brightness_distribution.png - 图像亮度分布")
        print("   - image_contrast_distribution.png - 图像对比度分布")
        print("   - damage_per_image_distribution.png - 每图像损伤数量分布")
        print("   - data_analysis_comprehensive.png - 综合分析图")
        print("   - sample_images_display.png - 样本图像展示")
        print("   - dataset_statistics.csv - 数据集统计表格")
        print("   - image_features_statistics.csv - 图像特征统计")

        print("\n2. 模型训练:")
        print("   - training_loss_curve.png - 训练损失曲线")
        print("   - training_accuracy_curve.png - 训练准确率曲线")
        print("   - training_curves_comprehensive.png - 综合训练曲线")
        print("   - confusion_matrix.png - 混淆矩阵")
        print("   - training_history.csv - 训练历史数据")

        print("\n3. 结果文件:")
        print("   - classification_results.csv - 分类结果表格")
        print("   - best_damage_classification_model.pth - 最佳模型权重")
        print("   - test_result.csv - 测试集预测结果")
        print("   - test_predictions_details.csv - 测试集预测详情")
        print("   - prediction_statistics.csv - 预测统计")

    except Exception as e:
        print(f"程序执行过程中出现错误: {e}")
        import traceback

        traceback.print_exc()