# YOLOV5

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

有关训练、测试和部署的完整文档，请参见 [YOLOv5 文档](https://docs.ultralytics.com/)。
