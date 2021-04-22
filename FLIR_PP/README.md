# FLIR Dataset Sanitizer

## ToDo List

- [x] Find all the different RGB frame-resolution in the dataset 
- [x] Find all the non paired IR frames (RGB frame missing)
- [x] Handle the asynchronous data in the video-set (Timestamp problem!)
- [x] Get rid of blank RGB images and HFOV_RGB < HFOV_IR
- [x] Handle the asynchronous data in the Train/Val-set (Timestamp problem!)
- [x] Cross labelling the RGB frames from IR
- [x] Preprocess the dataset: Crop RGB to 640*512 considering the FOV of IR
- [x] Test the label compatibility on the processed RGB images
- [x] Final data cleaning of miss-labeled data by FLIR
- [x] Manually labelling the objects, which were not seen by IR sensor (For Fusion)
- [x] Convert labels from json file (COCO format) to Yolo format
- [x] Merge two label files

## How to use:

```
# clone the repo.
https://github.com/SamVadidar/RGBT.git
cd RGBT/FLIR_PP

# Create a virtualenv. and install the requirements
python3 -m venv <name of your virtualenv.>
source path/to/env./bin/activate
pip install -r requirements.txt

# Make sure the path to the Dataset is pointing to a folder containing Train, Val and Video sets.
python3 pp.py path/to/FLIR/DATASET

# Output:
  A folder with FLIR_PP will be created under your FLIR Dataset directory, which contains the sane version of the original dataset. It will help you, if you are
  planning to fuse RGB and IR.
```

![Labeled RGB](https://github.com/SamVadidar/RGBT/blob/main/readmeFiles/FLIR_02743.jpg)
