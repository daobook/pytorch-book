# 快速入门

从安装了 `PyTorch>=1.7` 的 `Python>=3.8` 环境开始（本教程使用的是 Python3.9+，PyTorch 1.10+）。安装 YOLOv5 依赖项：

```shell
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -qr requirements.txt
```

或者直接使用在线文档 {download}`requirements.txt` 安装：

```
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

## 打包环境

为了快速和无忧无虑的安装，YOLOv5 已经打包了所有的依赖性，适用于以下环境  

包括 [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn)、[Python](https://www.python.org/) 和 [PyTorch](https://pytorch.org/)

- **谷歌 Colab 和 Kaggle** 笔记本的免费 GPU：<a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **谷歌云** 深度学习虚拟机。查看 [GCP 快速入门指南](environments/GCP-Quickstart.md)
- **Amazon** 深度学习AMI。查看 [AWS 快速入门指南](environments/AWS-Quickstart.md)
- **Docker 镜像**。查看 [Docker 快速入门指南](environments/Docker-Quickstart.md) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>

## 推理——检测对象

### 从你的克隆库中提取

要开始使用 [最新的 YOLO 模型](https://github.com/ultralytics/yolov5/releases) 进行目标检测，从你的版本库根目录中运行这个命令。结果被保存到 `'./runs/detect'`。

```bash
$ python detect.py --source OPTION
```

用你的选择取代 OPTION，以检测从：

* **Webcam**：`(OPTION = 0)` 用于从你连接的网络摄像头检测实时物体。
* **Image**：`(OPTION = filename.jpg)` 创建一个带有物体检测 overlay 的图像副本
* **Video**：`(OPTION = filename.mp4)` 创建一个带有物体检测 overlay 的视频副本
* **Directory**：`(OPTION = directory_name/)` 创建一个带有物体检测 overlay 的所有文件的副本
* **Global File Type** `(OPTION = directory_name/*.jpg)` 创建一个带有物体检测 overlay 的所有文件的副本
* **RTSP stream**：`(OPTION = rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa)` 对于来自流的实时物体检测（live object detection）
* **RTMP stream**：`(OPTION = rtmp://192.168.1.105/live/test)` 用于从流中检测实时物体
* **HTTP stream**：`(OPTION =  http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8)` 对于流中的实时物体检测

目前支持的文件格式如下：

* **图片：** bmp, jpg, jpeg, png, tif, tiff, dng, webp, mpo
* **视频** mov, avi, mp4, mpg, mpeg, m4v, wmv, mkv