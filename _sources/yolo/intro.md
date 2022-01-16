# 简介

要想现在就开始，请查看 [快速入门指南](quick-start.md)。

## YOLO 是什么

YOLO 是 "你只看一次"（You only look once） 的首字母缩写，是一种目标检测算法，它将图像划分为一个网格系统。网格中的每个单元负责检测其内部的物体。

由于其速度和准确率（accuracy），YOLO 是最著名的目标检测算法之一。

## YOLO 历史

### YOLOv5

在 YOLOv4 发布后不久，Glenn Jocher 引入使用 Pytorch 框架的 [YOLOv5](https://github.com/ultralytics/yolov5)。


- 作者：[Glenn Jocher](https://www.linkedin.com/in/glenn-jocher)  
- 发表：18 May 2020

### YOLOv4

随着原作者对 YOLO 的工作陷入停顿，YOLOv4 由 Alexey Bochoknovskiy、Chien-Yao Wang 和 Hong-Yuan Mark Liao 发布。

这篇论文的题目是 [YOLOv4：目标检测的最佳速度和准确度](https://arxiv.org/abs/2004.10934v1)

- 作者：[Alexey Bochoknovskiy](https://ru.linkedin.com/in/alexey-bochkovskiy-1213b542), [Chien-Yao Wang](https://www.researchgate.net/profile/Chien-Yao-Wang) 和[Hong-Yuan Mark Liao](https://en.wikipedia.org/wiki/Mark_Liao)  
- 发表： 23 April 2020

### YOLOv3

YOLOv3 在 YOLOv2 的基础上进行了改进，原作者 Joseph Redmon 和 Ali Farhadi 都有贡献。他们一起发表了 [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767v1)  

- 作者：[Joseph Redmon](https://pjreddie.com/) 和 [Ali Farhadi](https://www.cs.washington.edu/people/faculty/ali)  
- 发表： 8 Apr 2018

### YOLOv2

YOLOv2 是由 YOLO 的原作者 Joseph Redmon 和 Ali Farhadi 共同完成的。他们一起出版了 [YOLO9000：更好、更快、更强](https://arxiv.org/abs/1612.08242v1)  

- 作者：[Joseph Redmon](https://pjreddie.com/) 和 [Ali Farhadi](https://www.cs.washington.edu/people/faculty/ali)  
- 发表：  25 Dec 2016

### YOLOv1

YOLOv1 是由 Joseph Redmon 作为研究论文发布的。该论文的标题是 [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)  

- 作者：[Joseph Redmon](https://pjreddie.com/)  
- 发表： 8 Jun 2015

参考：[YOLOv5 | PyTorch](https://pytorch.org/hub/ultralytics_yolov5/)

![](https://zenodo.org/badge/264818686.svg)

## 模型描述

![YOLOv5 模型比较](./images/model_comparison.png)

[YOLOv5](https://ultralytics.com/yolov5)🚀 是在 COCO 数据集上训练的复合尺度目标检测模型系列，包括测试时间增强（Test Time Augmentation，简称 TTA）、模型集成、超参数演化以及导出到 ONNX、CoreML 和 TFLite 的简单功能。

| Model | size <sup>(pixels)</sup> | mAP<sup>val 0.5:0.95</sup> | mAP<sup>test 0.5:0.95</sup> | mAP<sup>val 0.5</sup> | Speed <sup>V100 (ms)</sup> |   | params <sup>(M)</sup> | FLOPS <sup>640 (B)</sup> |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [YOLOv5s6](https://github.com/ultralytics/yolov5/releases) | 1280 | 43.3 | 43.3 | 61.9 | **4.3** |   | 12.7 | 17.4 |
| [YOLOv5m6](https://github.com/ultralytics/yolov5/releases) | 1280 | 50.5 | 50.5 | 68.7 | 8.4 |   | 35.9 | 52.4 |
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases) | 1280 | 53.4 | 53.4 | 71.1 | 12.3 |   | 77.2 | 117.7 |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases) | 1280 | **54.4** | **54.4** | **72.0** | 22.4 |   | 141.8 | 222.9 |
| [YOLOv5x6](https://github.com/ultralytics/yolov5/releases) TTA | 1280 | **55.0** | **55.0** | **72.0** | 70.8 |

```{admonition} 表格描述
* AP{sup}`test` 表示 COCO [test-dev2017](http://cocodataset.org/#upload) 的结果，所有其他 AP 结果表示 val2017 的 accuracy。
* 除非另有说明，AP 值是单模型单尺度的。通过 `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65` **重现** mAP。
* Speed{sub}`GPU` 使用 GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 实例对 5000 多张 COCO val2017 图像进行了平均，并包括 FP16 推理、后处理和 NMS。通过 `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45` **重现**。
* 所有检查点（checkpoints）都以默认设置和超参数（没有自动增强）训练到 300 个 epochs。
* 测试时间增强（[TTA](https://github.com/ultralytics/yolov5/issues/303)）包括反射（reflection）和尺度增强（scale augmentation）。通过 `python test.py --data coco.yaml --img 1536 --iou 0.7 --augment` **重现** TTA。
```

````{margin}
```{admonition} 图片描述
* GPU 速度度量每张图像的端到端时间，平均为 5000 张 COCO val2017 图像，使用 V100 GPU，批次大小为 32，包括图像预处理、PyTorch FP16 推理、后处理和 NMS。
* EfficientDet 数据来自 [google/automl](https://github.com/google/automl)，批量大小为 8。通过 `python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt` **重现**。
```
````

![](./images/model_plot.png)

## 训练

运行下面的命令在 COCO 数据集上重现结果（数据集第一次使用时自动下载）。在单个 V100 上，YOLOv5s/m/l/x 的训练时间是2/4/6/8天（多 gpu 更快）。使用 GPU 允许的最大批处理大小（16GB 设备的批处理大小）。

```shell
python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

![](./images/coco-yolov5.png)

有关训练、测试和部署的完整文档，请参见 [YOLOv5 文档](https://docs.ultralytics.com/)。
