# Volume-MCSR
## Multi-view Large Kernel Attention Network for Multi-contrast MRI Volumetric Super-resolution

## 1、Environment
Python 3.8, Pytorch1.7.1, Cuda 11.0.
```
# Compile DCNv2:
cd $ROOT/models/modules/DCNv2
sh make.sh
```
For more implementation details about DCN, please see [[DCNv2]](https://github.com/lucasjinreal/DCNv2_latest).

## 2、 Datasets
### 2.1. Parpare Datasets for IXI, BrainTS and HCP dataset:
The IXI, BraTS2018 and HCP datasets can be downloaded at:
 [[IXI dataset]](https://brain-development.org/ixi-dataset/),  [[BrainTS dataset]](http://www.braintumorsegmentation.org/), and [[HCP dataset]](https://www.humanconnectome.org/study/hcp-young-adult/data-releases).    
 The original data are _**.nii**_ data. Split your data set into training sets, validation sets, and test sets.

[T1 folder:]

XXX1.nii,  XXX2.nii,  XXX3.nii,  XXX4.nii ...

[T2 folder:]

XXX1.nii,  XXX2.nii,  XXX3.nii,  XXX4.nii ...

#### Note that the images in the T1 and T2 folders correspond one to one. The LR images will be automatically generated in the training phase.

## 3、 Model training: 
Set your data set path and training parameters in **[configs/volumeSR.yaml]**, then run 

```bash
sh train.sh
```

## 4、 Model testing:
Modify the test configurations in Python file **[test_psnr.py]**. Then run:
```bash
CUDA_VISIBLE_DEVICES=0 python test_PSNR.py
```

## Acknowledgement
Our codes are built based on [BasicSR](https://github.com/XPixelGroup/BasicSR), thank them for releasing their codes!





