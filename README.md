# pointnet_paddle

**Paper:** [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/pdf/1612.00593.pdf)

**Reference Implementation:**
* [TensorFlow (Official)](https://github.com/charlesq34/pointnet)
* [PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

## Usage

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

## Performance
Accuracy: 89.76%