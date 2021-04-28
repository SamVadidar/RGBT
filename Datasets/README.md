Summary of what should be done:
===================================
- [x] Find all the Datasets, which provide IR images 
- [x] Review their specifications 
- [ ] Have a look on: 20-03-(Yolov3 PedestrianOnly)Pedestrian Detection in Severe Weather Condition.pdf

DataSet Review:
===================================

## FLIR [[Link](https://www.flir.com/oem/adas/adas-dataset-form/)]:
    * Specification:
       * Published on 2020
       * IR-Visible, Only IR is labeled
       * An aligned version exist where the RGB frames are cropped and aligned with IR frames
       * but the IOU of the RGB BBs are not optimal
       * 60% Day 40% Night but is not seperated in the Dataset
       * 50116 Person Obj.
       * 60705 Car Obj.
       * 5662 Bicycle Obj.

## CVC-14 [[Link](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/)]:
    * Specification:
        * Published on 2016
        * GrayScale RGB frames
        * Day-night, fir-visible, FramesNeg-FramesPos available, frame_size = 640*471 for both cameras
        * Pedestrian only
        * For training 3695 imegas during the day, and 3390 images during night, with ~1500 mandatory pedestrian annotated for each sequence.
        * For testing ~700 images for both sequences with ~2000 pedestrian during day, and ~1500 pedestrian during night
        Frame format is tif, therefore I assume that they are 16-bit

## KAIST Multispectral pedestrian Detection [[Link](https://sites.google.com/site/pedestrianbenchmark/)] [[Code](https://github.com/SoonminHwang/rgbt-ped-detection)]:
    * Specification:
        * Published on 2015 
        * Pedestrian Only
        * FIR-visible, day-night
        * 95k color-thermal pairs (640*480)
        * 62578 day frames = 70679 Objects
        * 32750 night frames = 44871 Objects
        * Check labels of crowds (Group of people)

## Others:

#### Dense [[Link](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets/)]:
    * Specification:
      * Published on 2020
      * Pedestrian, Vehicle
          * PassengerCar: SUV, VAN, No Trailors
          * LargeVehicle: Caravan, Truck, Trailor, Tram, Bus
          * RidableVehicle: Bicycle, Tricycle, Motorcycle, WheelChair, Scooter
      * Diverse weather condition available (Fog, Rain, Snow), Day-Night
      * 12000 real world driving and 1500 controlled fog chamber
      * cam_sterio available with 12-bit and 8-bit
      * FIR frame 8-bit
      * Very bad resolution and FOV for IR frames

#### ZUT-FIR-ADAS [[Link](https://github.com/pauliustumas/ZUT-FIR-ADAS)]:
    * Specification:
        * Published on 22.01.2020
        * IR Frames Only
        * Pedestrian Only
        * ZUT claims that the labeling is very high quality, where KAIST and SCUT failed to do so group of people are all in one box, which can make the filter extraction very hard therefore all the mentioned dataset has to go through some quality check
        * Each recording folder contains:
            * 16bitframes - 16bit original capture without processing.
            * 16bittransformed - 16bit capture with low pass filter applied and scaled to 640x480.
            * annotations - annotations and 8bit images made from 16bittransformed.
            * carparams.csv - a can details with coresponding frame id.
            * weather.txt - weather information in which the recording was made.
 
    * To have images without low pass filter applied:
        * take 16bit images from 16BitFrames folder and open with OpenCV function like: Mat input = imread(<image_full_path>, -1);
        * then use convertTo function like: input.convertTo(output, input.depth(), sc, sh), where output is transformed Mat, sc is scale and sh is shift from carParams.csv file.
        * finally, scale image to 640x480

#### CAMEL [[Link](https://camel.ece.gatech.edu/)]:
    * Specification:
        * Published on 2018
        * The frames are mostly not on-board frames
        * FIR-Visible, Well aligned, One label per pair image in the following format:
        * <Frame Number> <Track ID Number > <Annotation Class> <Bounding box top left x coordinate> <Bounding box top left y coordinate> <Bounding Box Width> <Bounding Box Height>
        * 30 Sequences which are mostly not on board images
        * Details like Humidity, Visibility, Weather condition, Temp., Time of recording are available
        * Both sensors are recorded with a jpg format, which seems like 8-bit recording

#### CVC-09 [[Link](http://adas.cvc.uab.es/elektra/enigma-portfolio/item-1/)]:
    * Specification:
        * Published on 2014
        * Day-night, fir only, FramesNeg-FramesPos available
        * Pedestrian only
        * 2 Sets: the first set contains 5990 frames and the second 5081, divided in training and testing sets each sequence.
        * Frame format is png, therefore I assume that they are 8-bit

#### Multispectral image recognition [[Link](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)]:
    * Specification:
        * Published on 2017
        * FIR-mir-nir-visible, Label in xml and txt format for each sensor (not in the same reference frame)
        * xml has additional info. such as truncated, crowd, and not normalized cordinates
        * Paper is hard to find for more details about the dataset!
        * Labels (.txt format) are same as Annotation_ConvertedSummarized but in yolo format
        * ir_label.py is available to convert xml to txt (yolo format)
        * noAnnotationslist.txt contains the images without labels
        * 7512 images in total for each sensor

#### SCUT-CV [[Link](https://github.com/SCUT-CV/SCUT_FIR_Pedestrian_Dataset):
    * Specification:
       * Published on 2019
       * IR Frames Only
       * Pedestrian Only

#### OTCBVS [[Link](http://vcipl-okstate.org/pbvs/bench/):
    * Dataset 01: pedestrian, above long shot
    * Dataset 03: pedestrian, above long shot


- <a name=""></a> 19-10-Driving Datasets Literature Review [[Link](https://arxiv.org/abs/1910.11968)]
- <a name=""></a> Image Databases [[Link](http://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)]