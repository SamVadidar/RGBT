# RGB and Thermal Feature-level Sensor Fusion (RGBT)
Download the FLIR pre-processed dataset [[**here**](https://drive.google.com/file/d/1M8N90Y9fexxu3TzGpLOGr6u9SB3ATiV1/view?usp=sharing)]

## ToDo List

- [x] Finish literature review of both Object Detection and Fusion 
- [x] Find the conversion Matrix btw. RGB and IR Frame
- [x] Prepare the FLIR Dataset for fusion
- [x] Implement Scaled_Yolov4
- [x] Test the RGB frames from FLIR Dataset on Scaled_Yolov4 (Baseline)
- [x] Ablation Study for the Baseline
- [x] Dive deeper into SPP, PAN, FPN, DCN, SAM, GIOU and other relevant papers for Fusion
- [x] Design the Fusion Network
- [x] Vanilla Fusion Training
- [x] CBAM Fusion Training
- [x] Entropy-based Channel Attention Module (EBAM)
- [x] Entropy-based Spatial Attention Module
- [x] Entropy-based Channel & Spatial Attention Module
- [x] EBAM Fusion Training
- [x] Day & Night Analysis
- [x] Synthetic Fog Analysis
- [ ] TensorRT and Latency Results
- [ ] Installation and User Guidlines

| Model | Test Size | Person<sub>AP@.5</sub><sup>test</sup> | Bicycle<sub>AP@.5</sub><sup>test</sup> | Car<sub>AP@.5</sub><sup>test</sup> | Overall<sub>mAP@.5</sub><sup>test</sup> | Num. of Param. |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| **RGB Baseline** | 320 | 39.6% | 50.4% | 79.4% | 56.6% | 52.5 | 
| **IR Baseline** | 320 | 49.6% | 54.9% | 84.4% | 63.0% | 52.5 | 
|  |  |  |  |  |
| **Vanila Fusion** | 320 | 56.9% | 56.7% | 82.0% | 65.2% | 81.8 | 
| **Fusion + CBAM** | 320 | 57.6% | 60.5% | 83.6% | 67.2% | 82.7 | 
| **Fusion + EBAM_C** | 320 | 62.6% | 65.9% | 86.0% | 71.5% | 82.7% | 
| **RGBT** | 320 | 63.7% | 67.1% | 86.4% | 72.4% | 82.7 | 
| **CFR_3** | 640 | 74.4% | 57.7% | 84.9% | 72.3% | 276 | 
| **RGBT** | 640 | **80.1%** | **76.7%** | **91.8%** | **82.9%** | **82.7%** | 
|  |  |  |  |  |

## Repo Structure:
Datasets: A summary of all available dataset with IR-RGB image pair\
FLIR_PP: Preprocessing of the FLIR dataset - The cross-labeling algorithm can be used by using pp.py\
Fusion: The implementation of the RGBT fusion network\
Related_Works: Literature Review\
**To Train** use the train_org.py\
**To Test** use the test_org.py
