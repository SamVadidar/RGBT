# RGBT

Here we deal with RGB and Thermal (IR) Fusion.

## ToDo List

- [ ] Finish literature review of both Object Detection and Fusion 
- [ ] Find the conversion Matrix btw. RGB and IR Frame
- [ ] Convert Darknet Weights to TF
- [ ] Implement Yolov3 in TF from scratch
- [ ] Dataset preprocessing - All Datasets must be converted to same labeling and folder structure
- [ ] Starting to concatinate features

## 1. Papers

### 1.1 RGB Object Detection

**Summary** [[Link](https://drive.google.com/file/d/1mGZx7mDXnWvDElDW12yBqb0TLn7rtnPc/view?usp=sharing)]

#### 2020
- <a name=""></a> 20-11-Scaled-Yolov4 [[Paper](https://arxiv.org/abs/2011.08036)]
- <a name=""></a> 20-04-Yolov4 [[Paper](https://arxiv.org/abs/2004.10934)]

#### 2018
- <a name=""></a> 18-04-Yolov3 [[Paper](https://arxiv.org/abs/1804.02767)]

#### 2016
- <a name=""></a> 16-12-Yolov2 [[Paper](https://arxiv.org/abs/1612.08242)]
- <a name=""></a> 16-05-Yolov1 [[Paper](https://arxiv.org/abs/1506.02640)]

### 1.2 RGBT Fusion

**Summary** [[Link](https://drive.google.com/file/d/1TWDQcAVUJwCN9DdyT3kNgoq_5BUKMdc0/view?usp=sharing)]

#### 2021
- <a name=""></a> 21-01-Guided Attentive Feature Fusion for Multispectral Pedestrian Detection [[Paper](https://openaccess.thecvf.com/content/WACV2021/papers/Zhang_Guided_Attentive_Feature_Fusion_for_Multispectral_Pedestrian_Detection_WACV_2021_paper.pdf)]

#### 2020
- <a name=""></a> 20-11-ABiFN: Attention-based bi-modal fusion network for object detection at night time [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/el.2020.1952)]
- <a name=""></a> 20-09-Multispectral Fusion for Object Detection with Cyclic Fuse-and-Refine Blocks [[Paper](https://arxiv.org/abs/2009.12664)]
- <a name=""></a> 20-07-Borrow from Anywhere: Pseudo Multi-modal Object Detection in Thermal Imagery [[Paper](https://arxiv.org/abs/1905.08789)] [[Code](https://github.com/tdchaitanya/MMTOD)]
- <a name=""></a> 20-07-CNN based Color and Thermal Image Fusion for Object Detection in Automated Driving [[Paper](https://www.researchgate.net/publication/342736973_CNN_based_Color_and_Thermal_Image_Fusion_for_Object_Detection_in_Automated_Driving)]
- <a name=""></a> 20-06-Night Vision Surveillance: Object Detection using thermal and visible images [[Paper](https://ieeexplore.ieee.org/document/9154066)]

#### 2019
- <a name=""></a> 19-09-Multi-Domain Attentive Detection Network [[Paper](https://ieeexplore.ieee.org/document/8803206)]
- <a name=""></a> 19-04-Deep Learning based Pedestrian Detection at all Light Conditions [[Paper](https://ieeexplore.ieee.org/document/8698101)]

#### 2017
- <a name=""></a> 17-11-Deep object classification in low resolution LWIR imagery via transfer learning[[Paper](https://pureadmin.qub.ac.uk/ws/portalfiles/portal/134854047/main.pdf)]

### 1.3 Thermal (IR) Object Detection

**Summary** [[Link](https://drive.google.com/file/d/1_ci6XkaW29tP6wxKqHz5fgeigytHS6_A/view?usp=sharing)]

#### 2020
- <a name=""></a> 20-07-Thermal Object Detection in Difficult Weather Conditions Using YOLO [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9133581)]
- <a name=""></a> 20-04-Using Deep Learning in Infrared Images to Enable Human Gesture Recognition for Autonomous Vehicles [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9079509)]

#### 2018
- <a name=""></a> 18-12-Object Recognition on Long Range Thermal Image Using State of the Art DNN [[Paper](https://ieeexplore.ieee.org/document/8572026)]

### 1.4 GAN

#### 2020
- <a name=""></a> 20-01-Thermal Image Super-resolution: A Novel Architecture and Dataset [[Paper](http://158.109.8.37/files/RSV2020.pdf)]

## 2. Related Blog Posts 

- <a name=""></a> Object detection on thermal images_2020 [[Link](https://medium.com/@joehoeller/object-detection-on-thermal-images-f9526237686a)] [[Code](https://github.com/joehoeller/Object-Detection-on-Thermal-Images)]
- <a name=""></a> Object detection on thermal images_2019 [[Link](https://medium.com/swlh/object-detection-on-thermal-images-4f3410a89db4)] [[Code](https://github.com/enesozi/object-detection)]

## 3. Related Works

- <a name=""></a> Yolov4-TF [[Repo](https://github.com/hunglc007/tensorflow-yolov4-tflite)]

## 4. Dataset

- <a name=""></a> 19-10-Driving Datasets Literature Review [[Link](https://arxiv.org/abs/1910.11968)]

**IR**
- <a name=""></a> 2018-FLIR [[Link](https://www.flir.com/oem/adas/adas-dataset-form/)]
- <a name=""></a> 2017-Multispectral Semantic Segmentation Dataset [[Link](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)]
- <a name=""></a> 2015-KAIST Multispectral Pedestrian Detection Benchmark [[Link](https://sites.google.com/site/pedestrianbenchmark/)] [[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

**RGB**
- <a name=""></a> COCO [[Link](https://cocodataset.org/#home)]
- <a name=""></a> Pascal Voc [[Link](http://host.robots.ox.ac.uk/pascal/VOC/)]
