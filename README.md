# RGB and Thermal Feature-level Sensor Fusion (RGBT)
Remark: Before running the python scripts, make sure you are in the RGBT folder, to avoid relative path issues!
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



## Repo Structure:
Datasets: A summary of all available dataset with IR-RGB image pair\
FLIR_PP: Preprocessing of the FLIR dataset - The cross-labeling algorithm can be used by using pp.py\
Fusion: The implementation of the RGBT fusion network\
Related_Works: Literature Review\
**To Train** use the train_org.py\
**To Test** use the test_org.py
