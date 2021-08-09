# pointnet_paddle

**Paper:** [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)
## 一、简介
![prediction example](https://github.com/charlesq34/pointnet/blob/master/doc/teaser.png)

[PointNet](https://arxiv.org/pdf/1612.00593.pdf)
## 二、复现精度
| 指标 | 原论文 | 复现精度 |
| --- | --- | --- |
| top-1 Acc | 89.2 | 90.2 |

## 三、数据集
使用的数据集为：[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)。

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0
  - tqdm

## 五、快速开始
### Data Preparation
Download [alignment ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and put it in `./dataset/modelnet40_normal_resampled/`

### Train
```
python train_modelnet.py --process_data
```

### Test
```
python test_modelnet.py --log_dir path_to_model
```

## 六、代码结构与详细说明

### 6.1 代码结构

```
|—— README.md
|—— provider.py    # 点云数据增强
|—— ModelNetDataset.py # 数据集定义及加载
|── train_modelnet.py       # 训练网络
|── test_modelnet.py     # 测试网络
|—— models        # 模型文件定义
```
### 6.2 参数说明

可以在 `train_modelnet.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  ----  |  ----  |  ----  |  ----  |
| batch_size  | 24 | batch_size 大小 ||
| epoch  | 200, 可选 | epoch次数 ||
| batch_size  | 32, 可选 | batch_size 大小 ||
| learning_rate | 0.001, 可选 | 初始学习率 ||
| num_point | 1024, 可选 | 采样的点的个数 ||
| decay_rate | 1e-4, 可选 | weight decay ||
| use_normals | False, 可选 | normalize 点 ||
| use_uniform_sample | False, 可选 | 均匀采样 ||
| process_data | False, 可选 | 是否预处理数据，如果没有下载预处理的数据需要为true ||
| feature_transform | False, 可选 | 使用特征变换 ||

**Reference Implementation:**
* [TensorFlow (Official)](https://github.com/charlesq34/pointnet2)
* [PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)