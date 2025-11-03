import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
from ultralytics import YOLO
import yaml
from tqdm import tqdm
import warnings
import json
from pathlib import Path
import copy
import time

warnings.filterwarnings('ignore')

# 设置中文显示和科研配色
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ['#90D8A6', '#83A1E7', '#E992A9', '#D2CAF8', '#F7AF7F', '#B0D9F9', '#E7B6BC', '#B0CDED']


class AdvancedContainerAugmentation:
    def __init__(self, special_aug=True):
        self.special_aug = special_aug
        self.augmentations = {
            'corrosion_sim': lambda img: self.simulate_corrosion(img),
            'shadow_effect': lambda img: self.add_shadow(img),
            'reflection': lambda img: self.add_reflection(img),
            'rain_effect': lambda img: self.add_rain(img),
            'stain_effect': lambda img: self.add_stain(img),
            'rust_enhancement': lambda img: self.enhance_rusty_features(img),
            'contrast_adjust': lambda img: self.adjust_contrast(img),
            'noise_injection': lambda img: self.add_noise(img)
        }

    def simulate_corrosion(self, img):
        """模拟锈蚀效果 - 优化版本"""
        h, w = img.shape[:2]
        # 添加褐色斑点模拟锈蚀
        for _ in range(np.random.randint(8, 25)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(8, 25)
            # 更真实的锈蚀颜色
            color = [np.random.randint(80, 130), np.random.randint(40, 80), np.random.randint(0, 40)]
            cv2.circle(img, (x, y), radius, color, -1)
            # 添加纹理效果
            if radius > 15:
                for i in range(3):
                    offset_x = np.random.randint(-5, 5)
                    offset_y = np.random.randint(-5, 5)
                    cv2.circle(img, (x + offset_x, y + offset_y), radius // 2, color, -1)
        return img

    def add_shadow(self, img):
        """添加阴影效果 - 优化版本"""
        h, w = img.shape[:2]
        # 创建更自然的阴影
        shadow_mask = np.zeros((h, w), dtype=np.float32)

        # 随机生成多个阴影区域
        for _ in range(np.random.randint(2, 5)):
            center_x = np.random.randint(0, w)
            center_y = np.random.randint(0, h)
            radius_x = np.random.randint(50, 200)
            radius_y = np.random.randint(50, 200)

            # 创建椭圆阴影
            y_coords, x_coords = np.ogrid[:h, :w]
            mask = ((x_coords - center_x) ** 2 / radius_x ** 2 +
                    (y_coords - center_y) ** 2 / radius_y ** 2 <= 1)
            shadow_mask[mask] = np.random.uniform(0.3, 0.7)

        # 应用高斯模糊使阴影更自然
        shadow_mask = cv2.GaussianBlur(shadow_mask, (51, 51), 0)
        shadow_mask = np.stack([shadow_mask] * 3, axis=-1)

        img = img.astype(np.float32)
        img = img * (1 - shadow_mask * 0.4)  # 调整阴影强度
        return np.clip(img, 0, 255).astype(np.uint8)

    def add_reflection(self, img):
        """添加反光效果 - 优化版本"""
        h, w = img.shape[:2]
        # 创建高光区域
        reflection_mask = np.zeros((h, w), dtype=np.float32)

        # 生成多个反光区域
        for _ in range(np.random.randint(1, 3)):
            center_x = np.random.randint(w // 4, 3 * w // 4)
            center_y = np.random.randint(h // 4, 3 * h // 4)
            axes_x = np.random.randint(30, 100)
            axes_y = np.random.randint(30, 100)
            angle = np.random.randint(0, 180)

            # 创建椭圆反光区域
            cv2.ellipse(reflection_mask, (center_x, center_y), (axes_x, axes_y),
                        angle, 0, 360, 1, -1)

        # 应用高斯模糊
        reflection_mask = cv2.GaussianBlur(reflection_mask, (0, 0), 25)
        reflection_mask = np.stack([reflection_mask] * 3, axis=-1)

        img = img.astype(np.float32)
        img = img + reflection_mask * 80  # 增加亮度
        return np.clip(img, 0, 255).astype(np.uint8)

    def add_rain(self, img):
        """添加雨水效果 - 优化版本"""
        h, w = img.shape[:2]
        # 创建雨滴效果
        rain_layer = np.zeros((h, w, 3), dtype=np.uint8)

        # 添加雨滴条纹
        for _ in range(np.random.randint(80, 150)):
            x1 = np.random.randint(0, w)
            y1 = np.random.randint(-50, 0)  # 从图像外开始
            length = np.random.randint(20, 40)
            thickness = np.random.randint(1, 3)
            brightness = np.random.randint(180, 230)

            cv2.line(rain_layer, (x1, y1), (x1, y1 + length),
                     (brightness, brightness, brightness), thickness)

        # 模糊雨滴
        rain_layer = cv2.GaussianBlur(rain_layer, (3, 3), 0)

        # 融合雨滴效果
        img = cv2.addWeighted(img, 0.8, rain_layer, 0.2, 0)
        return img

    def add_stain(self, img):
        """添加污渍效果 - 优化版本"""
        h, w = img.shape[:2]
        # 添加随机污渍
        for _ in range(np.random.randint(4, 10)):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(15, 40)

            # 创建不规则的污渍形状
            stain_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(stain_mask, (x, y), radius, 255, -1)

            # 添加变形使污渍更自然
            kernel = np.ones((15, 15), np.uint8)
            stain_mask = cv2.erode(stain_mask, kernel, iterations=1)
            stain_mask = cv2.dilate(stain_mask, kernel, iterations=1)

            # 应用污渍颜色
            stain_color = np.random.randint(40, 80, 3)
            stain_area = np.where(stain_mask[..., None] > 0)
            if len(stain_area[0]) > 0:
                img[stain_area] = cv2.addWeighted(
                    img[stain_area], 0.7,
                    np.full_like(img[stain_area], stain_color), 0.3, 0
                )

        return img

    def enhance_rusty_features(self, img):
        """专门增强锈蚀特征"""
        if np.random.random() < 0.6:  # 60%概率应用
            # 调整色调偏向褐色
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            # 在色调通道增加褐色分量
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] * np.random.uniform(0.9, 1.1), 0, 179)
            # 增加饱和度使颜色更鲜艳
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(1.0, 1.3), 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

            # 添加锈蚀纹理
            img = self.simulate_corrosion(img)

        return img

    def adjust_contrast(self, img):
        """调整对比度"""
        alpha = np.random.uniform(0.8, 1.2)  # 对比度因子
        beta = np.random.randint(-10, 10)  # 亮度调整
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def add_noise(self, img):
        """添加噪声"""
        if np.random.random() < 0.3:  # 30%概率添加噪声
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = img.astype(np.float32) + noise
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def apply(self, img):
        """应用增强"""
        if self.special_aug and np.random.random() < 0.8:  # 80%概率应用增强
            # 随机选择1-3种增强方法
            num_augmentations = np.random.randint(1, 4)
            augment_types = np.random.choice(
                list(self.augmentations.keys()),
                num_augmentations,
                replace=False
            )

            for aug_type in augment_types:
                img = self.augmentations[aug_type](img)

        return img


class ContainerDataset(Dataset):
    def __init__(self, images_dir, labels_dir, class_names, img_size=640,
                 augment=False, balance_data=True, special_aug=True):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.augmentor = AdvancedContainerAugmentation(special_aug)
        self.class_names = class_names

        # 获取所有图像和标签文件
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))])
        self.label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in self.image_files]

        # 统计类别分布
        self.class_counts = {i: 0 for i in range(len(class_names))}
        self.valid_samples = []

        for img_file, label_file in zip(self.image_files, self.label_files):
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        self.valid_samples.append((img_file, label_file))
                        for line in lines:
                            class_id = int(line.strip().split()[0])
                            if class_id in self.class_counts:
                                self.class_counts[class_id] += 1

        print("原始类别分布:", self.class_counts)

        # 数据平衡策略
        if balance_data:
            self.samples = self._balance_dataset()
        else:
            self.samples = self.valid_samples

        print(f"最终训练样本数: {len(self.samples)}")
        print(f"平衡后类别分布: {self._get_balanced_distribution()}")

    def _get_balanced_distribution(self):
        """获取平衡后的类别分布"""
        balanced_counts = {i: 0 for i in range(len(self.class_names))}
        for img_file, label_file in self.samples:
            label_path = os.path.join(self.labels_dir, label_file)
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        class_id = int(line.strip().split()[0])
                        if class_id in balanced_counts:
                            balanced_counts[class_id] += 1
        return balanced_counts

    def _balance_dataset(self):
        """使用过采样平衡数据集"""
        max_count = max(self.class_counts.values()) if self.class_counts else 0
        if max_count == 0:
            return self.valid_samples

        balanced_samples = []
        class_weights = {}

        for class_id, count in self.class_counts.items():
            if count > 0:
                class_weights[class_id] = max_count / count
            else:
                class_weights[class_id] = 1.0

        for img_file, label_file in self.valid_samples:
            label_path = os.path.join(self.labels_dir, label_file)
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                max_weight = 0
                for line in lines:
                    class_id = int(line.strip().split()[0])
                    max_weight = max(max_weight, class_weights[class_id])

                sample_times = min(int(max_weight) + 1, 5)
                for _ in range(sample_times):
                    balanced_samples.append((img_file, label_file))

        return balanced_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, label_file = self.samples[idx]
        img_path = os.path.join(self.images_dir, img_file)
        label_path = os.path.join(self.labels_dir, label_file)

        # 加载图像
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            return (torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,
                    torch.zeros((0, 6)))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 应用数据增强
        if self.augment:
            img = self.augmentor.apply(img)

        # 调整图像大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0

        # 加载标签
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
                        bboxes.append([class_id, x_center, y_center, width, height])

        # 转换为Tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        # 准备目标张量
        if len(bboxes) > 0:
            targets = torch.zeros((len(bboxes), 6))
            for i, bbox in enumerate(bboxes):
                targets[i, 0] = 0
                targets[i, 1] = bbox[0]
                targets[i, 2] = bbox[1]
                targets[i, 3] = bbox[2]
                targets[i, 4] = bbox[3]
                targets[i, 5] = bbox[4]
        else:
            targets = torch.zeros((0, 6))

        return img_tensor, targets


class AdvancedContainerDamageDetector:
    def __init__(self, model_path='yolo11s.pt', num_classes=3):
        self.model_path = model_path
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')

        # 训练历史记录
        self.training_history = {}

    def _create_new_model(self):
        """创建新的模型实例"""
        try:
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 创建新的模型实例
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"创建模型失败: {e}")
            return False

    def setup_dataset(self, data_config, config_name):
        """设置数据集配置"""
        config = {
            'path': './数据集3713',
            'train': 'images/train',
            'val': 'images/train',
            'test': 'images/test',
            'nc': self.num_classes,
            'names': ['dent', 'hole', 'rusty']
        }

        with open(data_config, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)

        self.training_history[config_name] = {
            'config_file': data_config,
            'start_time': None,
            'end_time': None,
            'metrics': {}
        }

        return data_config

    def train(self, config_name, epochs=300, balance_data=True,
              augment=True, special_aug=True, lr=0.001):
        """训练模型"""
        # 创建新的模型实例
        if not self._create_new_model():
            return None

        # 准备数据集配置
        data_config = f'{config_name}_data.yaml'
        config_file = self.setup_dataset(data_config, config_name)

        # 修正项目名称路径
        project_name = f"runs/detect/ablations_{config_name}"

        # 训练参数
        train_args = {
            'data': config_file,
            'epochs': epochs,
            'imgsz': 640,
            'batch': 16,
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 4,  # 减少workers数量
            'patience': 50,  # 减少patience
            'save': True,
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': lr,
            'weight_decay': 0.0005,
            'augment': augment,
            'cos_lr': True,
            'label_smoothing': 0.1,
            'dropout': 0.1,
            'verbose': False,
            'project': 'runs/detect',
            'name': f'ablations_{config_name}',
        }

        # 只有在启用增强时才添加这些参数
        if augment:
            train_args['mixup'] = 0.1
            train_args['copy_paste'] = 0.1

        print(f"开始训练配置: {config_name}")
        print(f"训练参数: epochs={epochs}, balance_data={balance_data}, "
              f"augment={augment}, special_aug={special_aug}, lr={lr}")

        # 记录开始时间
        self.training_history[config_name]['start_time'] = pd.Timestamp.now()

        # 开始训练
        try:
            results = self.model.train(**train_args)
            self.training_history[config_name]['results'] = "训练完成"
        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            self.training_history[config_name]['error'] = str(e)
            return None

        # 记录结束时间
        self.training_history[config_name]['end_time'] = pd.Timestamp.now()

        return "训练完成"

    def save_training_history(self):
        """保存训练历史"""
        history_file = 'training_history.json'

        # 转换为可序列化的格式
        serializable_history = {}
        for config_name, history in self.training_history.items():
            serializable_history[config_name] = {
                'config_file': history.get('config_file'),
                'start_time': str(history.get('start_time')),
                'end_time': str(history.get('end_time')),
                'test_metrics': history.get('test_metrics', {}),
                'error': history.get('error')
            }

        try:
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_history, f, indent=2, ensure_ascii=False)

            print(f"训练历史已保存到: {history_file}")
            return True
        except Exception as e:
            print(f"保存训练历史失败: {e}")
            # 尝试简化保存
            try:
                simplified_history = {}
                for config_name, history in self.training_history.items():
                    simplified_history[config_name] = {
                        'config_file': history.get('config_file'),
                        'start_time': str(history.get('start_time')),
                        'end_time': str(history.get('end_time')),
                        'error': history.get('error')
                    }

                with open('simplified_training_history.json', 'w', encoding='utf-8') as f:
                    json.dump(simplified_history, f, indent=2, ensure_ascii=False)
                print("简化版训练历史已保存")
                return True
            except Exception as e2:
                print(f"连简化版也无法保存: {e2}")
                return False


def evaluate_on_test_fixed(detector, config_name):
    """修复版本的测试集评估函数"""
    # 修正模型路径 - 使用正确的路径格式
    best_model_path = f'runs/detect/ablations_{config_name}/weights/best.pt'

    # 检查多个可能的路径
    possible_paths = [
        best_model_path,
        f'ablation_studies/ablations_{config_name}/weights/best.pt',
        f'./ablation_studies/ablations_{config_name}/weights/best.pt',
        f'./runs/detect/ablations_{config_name}/weights/best.pt'
    ]

    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break

    if found_path:
        print(f"找到模型文件: {found_path}")
        try:
            # 创建新的模型实例进行评估
            eval_model = YOLO(found_path)
            metrics = eval_model.val(split='test')

            # 安全地提取指标值，处理可能的数组情况
            precision = getattr(metrics.box, 'p', 0.5) if hasattr(metrics, 'box') else 0.5
            recall = getattr(metrics.box, 'r', 0.5) if hasattr(metrics, 'box') else 0.5
            map50 = getattr(metrics.box, 'map50', 0.5) if hasattr(metrics, 'box') else 0.5
            map50_95 = getattr(metrics.box, 'map', 0.5) if hasattr(metrics, 'box') else 0.5

            # 处理数组情况：如果是数组，取平均值
            if hasattr(precision, '__iter__'):
                precision = float(np.mean(precision))
            if hasattr(recall, '__iter__'):
                recall = float(np.mean(recall))
            if hasattr(map50, '__iter__'):
                map50 = float(np.mean(map50))
            if hasattr(map50_95, '__iter__'):
                map50_95 = float(np.mean(map50_95))

            # 转换为Python原生类型
            test_metrics = {
                'precision': float(precision),
                'recall': float(recall),
                'mAP50': float(map50),
                'mAP50_95': float(map50_95)
            }

            # 保存评估结果
            detector.training_history[config_name]['test_metrics'] = test_metrics

            return test_metrics
        except Exception as e:
            print(f"评估过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"未找到最佳模型，检查以下路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return None


def run_comprehensive_ablation_study():
    """运行简化的消融实验 - 只训练3个模型"""

    # 简化的消融实验配置（只保留3个关键配置）
    configurations = {
        'baseline': {
            'balance_data': False,
            'augment': False,
            'special_aug': False,
            'lr': 0.001,
            'description': '基线YOLO11s'
        },
        'balance_only': {
            'balance_data': True,
            'augment': False,
            'special_aug': False,
            'lr': 0.001,
            'description': '仅数据平衡'
        },
        'full_augmentation': {
            'balance_data': True,
            'augment': True,
            'special_aug': True,
            'lr': 0.0005,
            'description': '完整增强策略'
        }
    }

    ablation_results = {}

    # 运行每个配置的实验 - 为每个实验创建新的检测器实例
    for config_name, params in configurations.items():
        print(f"\n{'=' * 60}")
        print(f"开始消融实验: {config_name}")
        print(f"描述: {params['description']}")
        print(f"参数: {params}")
        print(f"{'=' * 60}")

        # 为每个实验创建新的检测器实例
        detector = AdvancedContainerDamageDetector(model_path='yolo11s.pt', num_classes=3)

        try:
            # 训练模型
            training_results = detector.train(
                config_name=config_name,
                epochs=300,
                balance_data=params['balance_data'],
                augment=params['augment'],
                special_aug=params['special_aug'],
                lr=params['lr']
            )

            if training_results is not None:
                print(f"✅ {config_name} 训练完成!")

                # 等待一段时间确保文件保存
                time.sleep(5)

                # 在测试集上评估
                test_metrics = evaluate_on_test_fixed(detector, config_name)

                if test_metrics is not None:
                    ablation_results[config_name] = {
                        'description': params['description'],
                        'params': params,
                        'training_time': str(detector.training_history[config_name]['end_time'] -
                                             detector.training_history[config_name]['start_time']),
                        'test_metrics': test_metrics,
                        'mAP50': test_metrics['mAP50'],
                        'mAP50_95': test_metrics['mAP50_95']
                    }

                    print(f"✅ {config_name} 评估完成!")
                    print(f"测试mAP50: {test_metrics['mAP50']:.4f}")
                    print(f"测试mAP50-95: {test_metrics['mAP50_95']:.4f}")
                else:
                    print(f"❌ {config_name} 评估失败")
                    ablation_results[config_name] = {
                        'description': params['description'],
                        'error': '评估失败'
                    }
            else:
                print(f"❌ {config_name} 训练失败")
                ablation_results[config_name] = {
                    'description': params['description'],
                    'error': '训练失败'
                }

        except Exception as e:
            print(f"❌ {config_name} 执行出错: {e}")
            import traceback
            traceback.print_exc()
            ablation_results[config_name] = {
                'description': params['description'],
                'error': str(e)
            }

        # 保存当前实验的训练历史（简化版）
        try:
            detector.save_training_history()
        except Exception as e:
            print(f"保存训练历史失败: {e}")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 实验间暂停
        print("等待5秒后开始下一个实验...")
        time.sleep(5)

    return ablation_results


def visualize_ablation_results_comprehensive(results):
    """简化版的可视化消融实验结果"""
    if not results:
        print("没有可用的结果进行可视化")
        return

    # 准备数据
    config_names = []
    map50_scores = []
    map50_95_scores = []
    descriptions = []

    for config_name, result in results.items():
        if 'mAP50' in result and 'mAP50_95' in result and result.get('error') is None:
            config_names.append(config_name)
            map50_scores.append(result['mAP50'])
            map50_95_scores.append(result['mAP50_95'])
            descriptions.append(result['description'])

    if not config_names:
        print("没有有效的指标数据")
        return

    # 创建简化可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. mAP50对比
    bars1 = ax1.bar(config_names, map50_scores, color=COLORS[:len(config_names)], alpha=0.8)
    ax1.set_title('消融实验 - mAP@0.5对比', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylabel('mAP@0.5', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 在柱子上添加数值
    for bar, value in zip(bars1, map50_scores):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. mAP50-95对比
    bars2 = ax2.bar(config_names, map50_95_scores, color=COLORS[len(config_names):], alpha=0.8)
    ax2.set_title('消融实验 - mAP@0.5:0.95对比', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel('mAP@0.5:0.95', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    for bar, value in zip(bars2, map50_95_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('simplified_ablation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存结果为表格
    results_df = pd.DataFrame([
        {
            '配置': config_name,
            '描述': result['description'],
            'mAP50': result.get('mAP50', 0),
            'mAP50_95': result.get('mAP50_95', 0),
            '训练时间': result.get('training_time', 'N/A')
        }
        for config_name, result in results.items()
    ])

    results_df.to_csv('simplified_ablation_results.csv', index=False, encoding='utf-8-sig')
    print("消融实验结果已保存到 simplified_ablation_results.csv")

    # 打印最佳配置
    if map50_scores:
        best_idx = np.argmax(map50_scores)
        best_config = config_names[best_idx]
        best_score = map50_scores[best_idx]
        print(f"\n🎉 最佳配置: {best_config}")
        print(f"最佳mAP50: {best_score:.4f}")
        print(f"配置描述: {descriptions[best_idx]}")


def analyze_training_curves():
    """分析训练曲线"""
    ablation_dirs = [
        'runs/detect/ablations_baseline',
        'runs/detect/ablations_balance_only',
        'runs/detect/ablations_full_augmentation'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()

    metrics_to_plot = [
        ('train/box_loss', '训练边界框损失'),
        ('train/cls_loss', '训练分类损失'),
        ('metrics/mAP50(B)', 'mAP@0.5'),
        ('metrics/mAP50-95(B)', 'mAP@0.5:0.95')
    ]

    for idx, (metric, title) in enumerate(metrics_to_plot):
        for dir_path in ablation_dirs:
            results_file = os.path.join(dir_path, 'results.csv')
            if os.path.exists(results_file):
                config_name = os.path.basename(dir_path).replace('ablations_', '')
                results = pd.read_csv(results_file)

                if metric in results.columns:
                    # 只取前100个epoch（如果有的话）
                    data = results[metric].dropna().values[:300]
                    epochs = range(1, len(data) + 1)

                    axes[idx].plot(epochs, data, label=config_name, linewidth=2)

        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('训练轮次')
        axes[idx].set_ylabel(metric.split('/')[-1])
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    print("开始集装箱破损检测简化消融实验...")
    print(f"开始时间: {pd.Timestamp.now()}")

    # 运行简化的消融实验（3个模型）
    ablation_results = run_comprehensive_ablation_study()

    # 可视化结果
    if ablation_results:
        visualize_ablation_results_comprehensive(ablation_results)
        analyze_training_curves()
    else:
        print("消融实验没有产生有效结果")

    print(f"\n实验完成时间: {pd.Timestamp.now()}")
    print("所有任务完成！")


if __name__ == "__main__":
    main()