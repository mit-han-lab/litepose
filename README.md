# Lite Pose

### [slides](assets/LitePose-slides.pdf)|[paper](https://arxiv.org/abs/2205.01271)|[video](https://www.youtube.com/watch?v=TodvXYrswDI)

![demo](assets/LitePose-Mobile.gif)

## Abstract

Pose estimation plays a critical role in human-centered vision applications. However, it is difficult to deploy state-of-the-art HRNet-based pose estimation models on resource-constrained edge devices due to the high computational cost (more than 150 GMACs per frame). In this paper, we study efficient architecture design for real-time multi-person pose estimation on edge. We reveal that HRNet's high-resolution branches are redundant for models at the low-computation region via our **gradual shrinking** experiments. Removing them improves both efficiency and performance. Inspired by this finding, we design **LitePose**, an efficient single-branch architecture for pose estimation, and introduce two simple approaches to enhance the capacity of LitePose, including **Fusion Deconv Head** and **Large Kernel Convs**. Fusion Deconv Head  removes the redundancy in high-resolution branches, allowing scale-aware feature fusion with low overhead. Large Kernel Convs significantly improve the model's capacity and receptive field while maintaining a low computational cost. With only 25\% computation increment, $7\times7$ kernels achieve $+14.0$ mAP better than $3\times 3$ kernels on the CrowdPose dataset. On mobile platforms, LitePose reduces the latency by up to $5.0\times$ without sacrificing performance, compared with prior state-of-the-art efficient pose estimation models, pushing the frontier of real-time multi-person pose estimation on edge.

## Results


### CrowdPose Test

![image](assets/Figure-CrowdPose.png)

<table>
    <tr>
        <td rowspan="2">Model</td>
        <td rowspan="2">mAP</td>
        <td rowspan="2">#MACs</td>
        <td colspan="3" align="center">Latency (ms)</td>
    </tr>
    <tr>
        <td>Nano</td>
        <td>Mobile</td>
        <td>Pi</td>
    </tr>
    <tr>
        <td>HigherHRNet-W24</td>
        <td>57.4</td>
        <td>25.3G</td>
        <td>330</td>
        <td>289</td>
        <td>1414</td>
    </tr>
    <tr>
        <td>EfficientHRNet-H<sub>-1</sub></td>
        <td>56.3</td>
        <td>14.2G</td>
        <td>283</td>
        <td>267</td>
        <td>1229</td>
    </tr>
    <tr>
        <td>LitePose-Auto-S <b>(Ours)</b></td>
        <td>58.3</td>
        <td>5.0G</td>
        <td>97</td>
        <td>76</td>
        <td>420</td>
    </tr>
    <tr>
        <td>LitePose-Auto-XS <b>(Ours)</b></td>
        <td>49.4</td>
        <td>1.2G</td>
        <td>22</td>
        <td>27</td>
        <td>109</td>
    </tr>
</table>

### COCO Val/Test 2017

<table>
    <tr>
        <td rowspan="2">Model</td>
        <td rowspan="2" align="center">mAP<br>(val)</td>
        <td rowspan="2" align="center">mAP<br>(test-dev)</td>
        <td rowspan="2">#MACs</td>
        <td colspan="3" align="center">Latency (ms)</td>
    </tr>
    <tr>
        <td>Nano</td>
        <td>Mobile</td>
        <td>Pi</td>
    </tr>
    </tr>
        <td>EfficientHRNet-H<sub>-1</sub></td>
        <td>59.2</td>
        <td align="center">59.1</td>
        <td>14.4G</td>
        <td>283</td>
        <td>267</td>
        <td>1229</td>
    </tr>
    </tr>
        <td>Lightweight OpenPose</td>
        <td>42.8</td>
        <td align="center">-</td>
        <td>9.0G</td>
        <td>-</td>
        <td>97</td>
        <td>-</td>
    </tr>
    <tr>
        <td>LitePose-Auto-M <b>(Ours)</b></td>
        <td>59.8</td>
        <td align="center">59.7</td>
        <td>7.8G</td>
        <td>144</td>
        <td>97</td>
        <td>588</td>
    </tr>
</table>

*Note*: For more details, please refer to our paper.

## Usage

- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
- [Results](#results)
- [Training Process Overview](#training-process-overview)
- [Evaluation](#evaluation)
- [Models](#models)

### Prerequisites

1. Install [PyTorch](https://pytorch.org/) and other dependencies:
```
pip install -r requirements.txt
```

2. Install COCOAPI and CrowdPoseAPI following [Official HigherHRNet Repository](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation).

### Data Preparation

1. Please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for training and evalutation.
2. Please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is training and evaluation.
3. Refer to [Official HigherHRNet Repository](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) for more details about the data arrangement.

### Training Process Overview

#### Super-net Training

To train a supernet from scratch with the search space specified by [arch_manager.py](https://github.com/mit-han-lab/litepose-dev/blob/main/arch_manager.py), use

```
python dist_train.py --cfg experiments/crowd_pose/mobilenet/supermobile.yaml
```

#### Weight Transfer

After training the super-net, you may want to extract a specific sub-network (e.g. search-XS) from the super-net. The following script will be useful:

```
python weight_transfer.py --cfg experiments/crowd_pose/mobilenet/supermobile.yaml --superconfig mobile_configs/search-XS.json TEST.MODEL_FILE your_supernet_checkpoint_path
```

#### Normal Training

To train a normal network with a specific architecture (e.g. search-XS), please use the following script:

*Note*: Please change the **resolution** in configuration (e.g. experiments/crowd_pose/mobilenet/mobile.yaml) in accord with the architecture configuration (e.g. search-XS.json) before training.

```
python dist_train.py --cfg experiments/crowd_pose/mobilenet/mobile.yaml --superconfig mobile_configs/search-XS.json
```

### Evaluation

To evaluate the model with a specific architecture (e.g. search-XS), please use the following script:

```
python valid.py --cfg experiments/crowd_pose/mobilenet/mobile.yaml --superconfig mobile_configs/search-XS.json TEST.MODEL_FILE your_checkpoint_path
```

### Models

#### Pre-trained Models

To re-implement results in the paper, we need to load pre-trained checkpoints before training super-nets. These checkpoints are provided in [COCO-Pretrain](https://drive.google.com/file/d/18WOtQ6yi-pn69bAOeYojXMI7l8sZZG3p/view?usp=sharing) and [CrowdPose-Pretrain](https://drive.google.com/file/d/1fojt0DJA5WPg3IqdkGTpyOps4mdxpGn9/view?usp=sharing).

#### Result Models

We provide the checkpoints corresponding to the results in our paper.

<table>
    <tr>
        <td>Dataset</td>
        <td>Model</td>
        <td>#MACs</td>
        <td>mAP</td>
    </tr>
    <tr>
        <td rowspan="4" align="center">CrowdPose</td>
        <td>LitePose-Auto-L</td>
        <td>13.7</td>
        <td>61.9</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/file/d/189VHaeFg3RkH2wBxm7iIM57cNVf7dN9y/view?usp=sharing">LitePose-Auto-M</a></td>
        <td>7.8</td>
        <td>59.9</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/file/d/1kCmjJfrOScaGFDpguEadwxdyJ_ESoR5P/view?usp=sharing">LitePose-Auto-S</a></td>
        <td>5.0</td>
        <td>58.3</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/file/d/1U3jIFEmPLbxSUhScZJv1JPAoiFboA8Y6/view?usp=sharing">LitePose-Auto-XS</a></td>
        <td>1.2</td>
        <td>49.5</td>
    </tr>
    <tr>
        <td rowspan="4" align="center">COCO</td>
        <td><a href="https://drive.google.com/file/d/1_zJCRaYMDK77wmaYul6bHb6_M5qlJ8zP/view?usp=sharing">LitePose-Auto-L</a></td>
        <td>13.8</td>
        <td>62.5</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/file/d/1OIXIwE1VMSlWbDsZzYJU-qlIxPNAnBPh/view?usp=sharing">LitePose-Auto-M</a></td>
        <td>7.8</td>
        <td>59.8</td>
    </tr>
    <tr>
        <td><a href="https://drive.google.com/file/d/10NCT0UrQvMmTdjMLR7l-WvFBrvMofjXs/view?usp=sharing">LitePose-Auto-S</a></td>
        <td>5.0</td>
        <td>56.8</td>
    </tr>
    <tr>
        <td>LitePose-Auto-XS</td>
        <td>1.2</td>
        <td>40.6</td>
    </tr>
</table>

## Acknowledgements

Lite Pose is based on [HRNet-family](https://github.com/HRNet), mainly on [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation). Thanks for their well-organized code!

About Large Kernel Convs, several recent papers have found similar conclusions: [ConvNeXt](https://arxiv.org/abs/2201.03545), [RepLKNet](https://arxiv.org/abs/2203.06717). We are looking forward to more applications of large kernels on different tasks!

## Citation

If Lite Pose is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```bibtex
@article{wang2022lite,
  title={Lite Pose: Efficient Architecture Design for 2D Human Pose Estimation},
  author={Wang, Yihan and Li, Muyang and Cai, Han and Chen, Wei-Ming and Han, Song},
  journal={arXiv preprint arXiv:2205.01271},
  year={2022}
}
```
