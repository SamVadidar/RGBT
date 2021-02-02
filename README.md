# RGBT

Here we deal with RGB and Thermal(IR) Fusion.

## ToDo List

- [ ] Finish literature review of both Object Detection and Fusion 
- [ ] Convert Darknet Weights to TF
- [ ] Implement Yolov3 in TF from scratch
- [ ] Start with the network adaptation for IR
- [ ] Dataset preprocessing - All Datasets must be converted to same labeling and folder structure

## 1. Papers

### 1.1 RGB Object Detection

**My Summary** [[Link](https://drive.google.com/file/d/1JqPu_-s-JKxE3qVvJS_efO3kZUitP1VH/view?usp=sharing)]

#### 2020
- <a name=""></a> Yolov4-Scaled [[Paper](https://drive.google.com/file/d/19n-CJ8Bt3egjaxcWETfEv5QXyu7KQ2jS/view?usp=sharing)]
- <a name=""></a> Yolov4 [[Paper](https://drive.google.com/file/d/1FDsMBfLjIBgIBuUJjw2r_pm9wLlKN_1T/view?usp=sharing)]

#### 2018
- <a name=""></a> Yolov3 [[Paper](https://drive.google.com/file/d/1ztX_qpFn7XenS3fjRCNj4E9PhnpKwkjy/view?usp=sharing)]

#### 2016
- <a name=""></a> Yolov2 [[Paper](https://drive.google.com/file/d/1VF-lJsSCS-xzKDYE_Eq7WZykLvHH8O7x/view?usp=sharing)]
- <a name=""></a> Yolov1 [[Paper](https://drive.google.com/file/d/1VDlmjDIvMrUDps2geiQELo5KvOVbpSAE/view?usp=sharing)]

### 1.2 Thermal(IR) Object Detection

#### 2020
- <a name=""></a> Thermal object detection in difficult weather using Yolo [[Paper](https://drive.google.com/file/d/1HgePMFEwBB1XbkwVt8svWzNMo5Qmi6bw/view?usp=sharing)]
- <a name=""></a> Real time target detection for infrared images [[Paper](https://drive.google.com/file/d/1ub95QLOVAXUSxMLbcr8GFDXN4vLRoAlE/view?usp=sharing)]
- <a name=""></a> Human gesture recognition using Yolov3 [[Paper](https://drive.google.com/file/d/1XaMM0bGEWz-1fOfKXm9IzU4j69v06K-r/view?usp=sharing)]

#### 2019
- <a name=""></a> Human detection in thermal imaging using Yolo [[Paper](https://drive.google.com/file/d/1S2fGjzgNi8ri273Cl3NVMef3QET19oMF/view?usp=sharing)]
- <a name=""></a> (Review Paper) Lightweight CNN for vehicle recognition [[Paper](https://drive.google.com/file/d/1Q4ekOgBh21eXvaou02AbMCYSCuVoJgvd/view?usp=sharing)]

#### 2018
- <a name=""></a> Object recognition on LWIR using Yolo [[Paper](https://drive.google.com/file/d/1qrbEuAILS947vU5JurtAK3o5Uacrd-zm/view?usp=sharing)]
- <a name=""></a> CNN-based thermal infrared person detection by domain adaptation [[Paper](https://drive.google.com/file/d/1Zb_PZqeh214FLLuvugMNGsZzlhkosoJR/view?usp=sharing)]

### 1.3 RGBT Fusion (With Obj. Detection)

#### 2021
- <a name=""></a> Guided attentive feature fusion [[Paper](https://drive.google.com/file/d/1h7Wkq2zlO5T5dp67FVHXgw3SknVEADnD/view?usp=sharing)]

#### 2020
- <a name=""></a> Object detection in automotive driving [[Paper](https://drive.google.com/file/d/1-vLFw_QOADS9mfxrxFVDxys_6oqzFlkz/view?usp=sharing)]

#### 2019
- <a name=""></a> Enhancing object detection in adverse condition [[Paper](https://drive.google.com/file/d/1v6rwjBx-SVBXeD3UJEVkqaV0fAQIwEJ2/view?usp=sharing)]

### 1.4 RGBT Fusion (Without Obj. Detection)

#### 2019
- <a name=""></a> Dense Fuse [[Paper](https://drive.google.com/file/d/14vhuKERvxszemPcAypAbhSA5ban4eG55/view?usp=sharing)]
- <a name=""></a> Fast and efficient zero-learning fusion [[Paper](https://drive.google.com/file/d/1dy_JcuO0vEXpJiBhcsbDyHJqEfNnrqFK/view?usp=sharing)]

#### 2018
- <a name=""></a> Infrared and visible image fusion [[Paper](https://drive.google.com/file/d/1iHY_EEgOfcwvAiLGXnv3gAJu7RGY_YVc/view?usp=sharing)]

## 2. Related Blog Posts 

- <a name=""></a> Object detection on thermal images_2020 [[Link](https://medium.com/@joehoeller/object-detection-on-thermal-images-f9526237686a)] [[Code](https://github.com/joehoeller/Object-Detection-on-Thermal-Images)]
- <a name=""></a> Object detection on thermal images_2019 [[Link](https://medium.com/swlh/object-detection-on-thermal-images-4f3410a89db4)] [[Code](https://github.com/enesozi/object-detection)]
- <a name=""></a> 
- <a name=""></a> 

## 3. Related Works

- <a name=""></a> Yolov4-TF [[Repo](https://github.com/hunglc007/tensorflow-yolov4-tflite)]
- <a name=""></a> 

## 4. Dataset

**IR**
- <a name=""></a> FLIR [[Link](https://www.flir.com/oem/adas/adas-dataset-form/)]
- <a name=""></a> KAIST Multispectral Pedestrian Detection Benchmark [[Link](https://sites.google.com/site/pedestrianbenchmark/)] [[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]

**RGB**
- <a name=""></a> COCO [[Link](https://cocodataset.org/#home)]
- <a name=""></a> Pascal Voc [[Link](http://host.robots.ox.ac.uk/pascal/VOC/)]
