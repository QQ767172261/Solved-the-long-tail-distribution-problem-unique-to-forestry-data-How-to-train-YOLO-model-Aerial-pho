无人机-航拍森林树木健康状况检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e8e2ea91b9074f96a5ddf396094003ea.png)
，657张，yolo，voc，coco三种标注方式
图像尺寸:1024*1024
类别数量:4类
训练集图像数量:460; 验证集图像数量:134； 测试集图像数量:63
类别名称: 每一类图像数 ，每一类标注数
Dead Tree 枯树 508 8118
Alive Tree 活树 657 70977
Debris 残枝杂物 405 4989
Beetle-Fire Tree 病虫害 / 火烧树木 205 5072
image num: 657

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0951e4d404454ece88e4a785980c58a8.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3b11a9fb31ff4266bdde119b1513e07c.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6aec9e1854b84663b53676e5f715e171.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb1acea6b54b466fa4d100d5fc33bb42.png)=

### 📊 1. 航拍森林树木健康状况检测数据集概况

| 项目属性 | 详细说明 |
| :--- | :--- |
| **数据集名称** | Aerial-Forest-Health-Detection |
| **数据总量** | **657** 张高清航拍图 |
| **图像分辨率** | **1024 × 1024** 像素 (高分辨率) |
| **标注格式** | **三格式支持**：YOLO (`.txt`), VOC (`.xml`), COCO (`.json`) |
| **数据划分** | **训练集**: 460 张 (70%)**验证集**: 134 张 (20%)**测试集**: 63 张 (10%) |
| **类别数量** | **4 类** (生态健康关键指标) |
| **总标注框数** | **89,169** 个目标 (平均每张图约 135 棵树) |
| **应用场景** | 森林资源普查、病虫害早期预警、火灾后评估、林业碳汇估算 |

#### 🏷️ 类别分布与不平衡分析 (关键！)

| 类别 ID | 英文名称 | 中文含义 | 涉及图像数 | 标注框数量 | 占比 | 挑战等级 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | `Dead Tree` | 枯树 | 508 | 8,118 | 9.1% | ⭐⭐⭐ (需区分死树与阴影) |
| 1 | `Alive Tree` | 活树 | 657 | **70,977** | **79.6%** | ⭐ (背景类，易过拟合) |
| 2 | `Debris` | 残枝杂物 | 405 | 4,989 | 5.6% | ⭐⭐ (形状不规则，易漏检) |
| 3 | `Beetle-Fire Tree` | **病虫害/火烧** | 205 | 5,072 | 5.7% | ⭐⭐⭐⭐ (**核心目标**，样本少，需重点优化) |
| **总计** | - | - | - | **89,169** | 100% | - |

> **⚠️ 数据洞察**：
> 1.  **密集检测**：单张图片平均有 135 个目标，对模型的 NMS (非极大值抑制) 和小目标检测能力要求极高。
> 2.  **长尾分布**：核心关注的“病虫害/火烧”类别数量仅为“活树”的 **1/14**。如果不加干预，模型可能会忽略这一类。
> 3.  **分辨率问题**：1024x1024 对于标准 YOLO (通常输入 640) 来说太大。直接缩放会丢失小病斑细节；不缩放则需使用**切片辅助推理 (Slicing Aided Hyper Inference, SAHI)** 或在训练时开启大图训练模式。

---

### 💻 2. YOLOv8 训练代码 (针对航拍大图 + 长尾分布优化)

此代码专门解决了**高分辨率输入**和**类别不平衡**两个痛点：
1.  **大尺寸输入**：将 `imgsz` 设置为 **1024**，充分利用航拍图细节，避免缩小导致小病斑消失。
2.  **类别权重平衡**：通过 `cls_weight` 或损失函数策略，强迫模型关注稀有的 `Beetle-Fire Tree`。
3.  **密集目标优化**：调整 NMS 阈值和 IoU 计算方式，适应每图百树的场景。

#### 📄 配置文件 (`forest_health.yaml`)

```yaml
path: ./Forest_Health_Dataset  # 数据集根目录
train: images/train            # 对应 460 张
val: images/val                # 对应 134 张
test: images/test              # 对应 63 张

# 类别数量
nc: 4

# 类别名称 (顺序必须与标注文件中的索引一致)
names:
  0: Dead_Tree
  1: Alive_Tree
  2: Debris
  3: Beetle_Fire_Tree
```

#### 🐍 训练脚本 (`train_forest.py`)

```python
from ultralytics import YOLO
import torch
import os

def train_forest_health_model():
    print("🌲 开始训练航拍森林健康检测模型 (高分辨率版)...")
    
    # 1. 加载模型
    # 推荐使用 yolov8m.pt 或 yolov8l.pt
    # 由于输入尺寸大 (1024)，n/s 版本可能特征提取能力不足，m/l 更佳
    model = YOLO('yolov8m.pt') 

    # 2. 定义训练参数
    results = model.train(
        data='forest_health.yaml',
        epochs=150,                   # 足够轮次以收敛
        imgsz=1024,                   # 【关键】保持原始分辨率，捕捉细小病斑
        batch=8,                      # 【关键】1024分辨率显存占用大，需减小 Batch Size (根据显存调整，24G显存可试12)
        device='0' if torch.cuda.is_available() else 'cpu',
        workers=4,                    # 大图加载较慢，线程数不宜过多
        
        # --- 优化器设置 ---
        optimizer='SGD',
        lr0=0.01,
        lrf=0.05,
        momentum=0.937,
        weight_decay=0.0005,
        
        # --- 数据增强 (针对航拍视角) ---
        augment=True,
        hsv_h=0.015,                  # 色调微调 (不同季节植被颜色差异)
        hsv_s=0.7,                    # 饱和度
        hsv_v=0.4,                    # 亮度 (云层遮挡导致的阴影)
        degrees=0.0,                  # 航拍图通常正北朝上，或随机旋转？若无人机角度随意可设为 10.0
        translate=0.1,
        scale=0.5,                    # 缩放 (模拟不同飞行高度)
        shear=0.0,
        perspective=0.0,
        flipud=0.0,                   # 禁止上下翻转 (保持地理方位感，除非确认无影响)
        fliplr=0.5,                   # 允许左右翻转
        mosaic=1.0,                   # Mosaic (对密集树林场景极其有效，模拟更大视野)
        mixup=0.05,
        
        # --- 针对长尾分布的优化策略 ---
        # YOLOv8 默认有一定的自动平衡，但对于 1:14 的比例，建议手动干预
        # 注意：ultralytics 新版可能不直接支持 cls_weight 参数在 train() 中，
        # 若需强力干预，需修改 loss 函数或使用 'fl_gamma' (Focal Loss)
        fl_gamma=2.0,                 # 开启 Focal Loss，让模型专注难分类样本 (即稀有的病虫害树)
        label_smoothing=0.0,          # 关闭标签平滑，因为类别已经很不平衡，需要明确边界
        
        # --- 训练策略 ---
        patience=30,
        save=True,
        save_period=-1,
        verbose=True,
        project='runs/forest_health',
        name='forest_yolov8m_1024_v1',
        exist_ok=False,
        seed=42,
        deterministic=True,
        single_cls=False,
        rect=False,                   # 关闭矩形训练，1024本身就是正方形
        
        # --- 密集目标优化 ---
        iou=0.6,                      # 提高 NMS 的 IoU 阈值，防止把挨得很近的树合并掉 (默认0.7可能太高，0.6更保守)
        conf=0.001,                   # 训练时极低置信度阈值，确保不漏掉微小病斑
    )
    
    print(f"✅ 训练完成！最佳模型：{results.save_dir}/weights/best.pt")
    
    # 3. 验证与评估
    print("📊 正在测试集上进行高精度验证...")
    # 针对大图推理，建议使用 tta (Test Time Augmentation) 提升精度
    metrics = model.val(split='test', tta=True) 
    
    print("-" * 50)
    print(f"🏆 整体 mAP50: {metrics.box.map50:.4f}")
    print(f"🎯 整体 mAP50-95: {metrics.box.map:.4f}")
    print("-" * 50)
    
    # 打印各类别表现 (重点关注 Beetle_Fire_Tree)
    class_names = list(model.names.values())
    print("📈 各类别 mAP50-95 详情:")
    for i, name in enumerate(class_names):
        map_score = metrics.box.maps[i]
        status = "⚠️ 需优化" if map_score < 0.3 else "✅ 良好"
        print(f"   [{i}] {name}: {map_score:.4f} {status}")

if __name__ == '__main__':
    if not os.path.exists('forest_health.yaml'):
        print("❌ 错误：未找到 forest_health.yaml，请先创建！")
    else:
        train_forest_health_model()
```

### 💡 3. 针对航拍林业场景的特别建议

#### A. 推理阶段的“切片辅助” (SAHI)
由于训练使用了 1024 分辨率，但在实际应用中，无人机图片可能是 4K 甚至 8K 的。直接缩放会丢失细节。
**强烈建议**在推理时使用 **SAHI (Slicing Aided Hyper Inference)** 技术：
*   **原理**：将大图切割成多个 1024x1024 的小图分别检测，然后将结果合并去重。
*   **代码示例** (推理时)：
    ```python
    from sahi.predict import get_sliced_prediction
    from sahi.model import Yolov8DetectionModel

    # 加载训练好的模型
    detection_model = Yolov8DetectionModel(
        model_path="runs/forest_health/forest_yolov8m_1024_v1/weights/best.pt",
        confidence_threshold=0.25,
        device="cuda"
    )

    # 执行切片推理
    result = get_sliced_prediction(
        "large_aerial_image.jpg",
        detection_model,
        slice_height=1024,  # 与训练尺寸一致
        slice_width=1024,
        overlap_height_ratio=0.2, # 20% 重叠，防止目标在切缝处被切断
        overlap_width_ratio=0.2
    )
    # result 包含所有检测到的树木及其类别
    ```

#### B. 解决“活树”淹没“病树”的问题
如果训练后发现 `Beetle-Fire Tree` 的召回率依然很低：
1.  **欠采样 (Undersampling)**：在训练数据加载时，随机丢弃部分只包含 `Alive Tree` 的图片，使正负样本比例更平衡。
2.  **复制粘贴增强**：将少量的 `Beetle-Fire Tree` 目标抠图，复制粘贴到其他健康森林背景中，人工增加该类别的样本量。
3.  **两级检测策略**：
    *   第一级：检测所有 `Tree` (Alive + Dead + Beetle)。
    *   第二级：将所有检测到的树裁剪出来，送入一个二分类模型 (健康 vs 病害)。这通常比直接做多类别检测更精准。



