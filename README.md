# Crowd-Counting-With-Deep-Structured-Scale-Integration-Network
[Lingbo Liu, Zhilin Qiu, Guanbin Li , Shufan Liu, Wanli Ouyang, Liang Lin. Crowd Counting with Deep Structured Scale Integration Network, ICCV, 2019](https://arxiv.org/abs/1908.08692)
![image](https://github.com/Legion56/Legion56.github.io/blob/master/images/overview.png)
<p> &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 &#12288 Overview of our approach </p>

## Introduction
This is the repo for Crowd Counting with Deep Structured Scale Integration Network in ICCV 2019, which delivered a state-of-the-art framework for crowd counting task and two effective module to cope with huge scale variant in the crowd.

## Usage
### Requirements
```
CUDA 9.0 or higher
Python 2.7
opencv, PIL, scikit-learn
pytorch 0.4.2 or or higher
```

### Data preprocessing
#### Datasets
1. ShanghaiTech partA and partB
2. UCF_QNRF
3. UCF_CC_50
4. WorldExpo'10
#### Setting up
1. We implemented fix and adaptive gaussian kernel density map generation in python, and density maps are generated during training on the fly;
2. During testing, no density map is generated and gt counts are the number of annotated points in ROI;
3. Edit "/src/datasets.py"  to change the path to your original dataset foldered as the released ShanghaiTech dataset and set the density maps setting including, sigma for gaussian kernel, train_val split and mean_std;
#### Training
```
python nowtrain.py --dataset      'the dataset to train'
                   --model        'network to train; CRFVGG\CRFVGG_prune'
                   --loss         'default: MSE, MSE/NORMMSSSIM'
```
Please refer to /src/train_options.py for more options; Default scripts for training ShanghaiTech PartA avaliable on /scripts/train.sh
#### Testing
```
python nowtest.py  --dataset      'the dataset to test'
                   --model        'network to train; CRFVGG\CRFVGG_prune'
                   --model_path   'the path to the saved model to test'
```
Please refer to /nowtest.py for more options; Default scripts for training ShanghaiTech PartA avaliable on /scripts/test.sh
We will release the model reported on our paper, links on the performance session.
#### Tips
We train and test the UCF-QNRF dataset with its original resolution. 
   During training, to fit in memory, we pre-crop images to non-overlap or less-overlap image patches(in high resolution) and iterate through images via randomly choose one patch with prior to dense patches, follow by other data augment on the fly.
   During testing, images are croped to strictly non-overlap patches and add up the predicted count as the final estimation.


#### Performance

| Dataset | MAE | MSE |
| ---- | ---- | ---- |
| [ShanghaiTech Part A](https://www.dropbox.com/sh/wx8ah2c6pavod5p/AACDoJvNHrKJ_YaT_ObrCV-3a?dl=0)| 60.63 | 96.04 |
| [ShanghaiTech Part A(pruned-vgg)](https://www.dropbox.com/sh/wx8ah2c6pavod5p/AACDoJvNHrKJ_YaT_ObrCV-3a?dl=0)| 61.16 | 102.91 |
| [ShanghaiTech Part B](https://www.dropbox.com/sh/wx8ah2c6pavod5p/AACDoJvNHrKJ_YaT_ObrCV-3a?dl=0)| 6.85 | 10.34 |
| [UCF-QNRF](https://www.dropbox.com/sh/wx8ah2c6pavod5p/AACDoJvNHrKJ_YaT_ObrCV-3a?dl=0) | 99.1 | 159.2|
| UCF-CC-50 | 216.9 | 302.4 |
| WorldExpo'10 | 6.67(average) | |
| TRANCOS | 2.72| | 


## Citation 
If you use this code for your research, please cite our papers.

```
@inproceedings{liu2019crowd,
  title={Crowd Counting with Deep Structured Scale Integration Network},
  author={Liu, Lingbo and Qiu, Zhilin and Li, Guanbin and Liu, Shufan and Ouyang, Wanli and Lin, Liang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision (ICCV)},
  year={2019}
}
```