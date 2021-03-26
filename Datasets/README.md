Summary of what should be done:
===================================
1) Configure the GPU, Server, and Linux Laptop (Hardware-Software connection)
2) Review the possible datasets and a testbench you want to test on (Dataset)
3) Summary of Summaries and find a hole (Target)
4) Implement RGB Network
5) Implement the state of the art Fusion Network
6) Train the IR branch of the Fusion network with IR only Datasets
7) Test on FLIR-aligned version (Testbench?)


DataSet Review:
===================================
## CAMEL:
    Specification:
        Published on 2018 
        FIR-Visible, Well aligned, One label per pair image in the following format:
        <Frame Number> <Track ID Number > <Annotation Class> <Bounding box top left x coordinate> <Bounding box top left y coordinate> <Bounding Box Width> <Bounding Box Height>
        30 Sequences which are mostly not on board images
        Details like Humidity, Visibility, Weather condition, Temp., Time of recording are available
        Both sensors are recorded with a jpg format, which seems like 8-bit recording

## FLIR:
    Specification:
        Published on 2020
        IR-Visible, Only IR is labeled
        * An aligned version exist where the RGB frames are cropped and aligned with IR frames
        but the IOU of the RGB BBs are not optimal
        60% Day 40% Night but is not seperated in the Dataset
        
        50116 Person Obj.
        60705 Car Obj.
        5662 Bicycle Obj.
    
        FLIR_video_00203.jpeg Pedestrian visible in IR Frame but not in RGB Frame
        Therefore we cannot simply take all the labels from IR and punish the RGB network based
        on that!

        The white images from RGB images have to be deleted if RGB is to be used, but I do not
        see the need to use RGB images, since YOLO is already trained on RGB images pretty well!

## SCUT-CV:
    Specification:
        Published on 2019
        IR-Visible, Only IR is labeled
        * An aligned version exist where the RGB frames are cropped and aligned with IR frames
        but the IOU of the RGB BBs are not optimal
        60% Day 40% Night but is not seperated in the Dataset
    
## ZUT-FIR-ADAS:
    Specification:
        Published on 22.01.2020
        IR Frames Only
        Pedestrian Only
        Each recording folder contains:
            16BitFrames - 16bit original capture without processing.
            16bittransformed - 16bit capture with low pass filter applied and scaled to 640x480.
            annotations - annotations and 8bit images made from 16bittransformed.
            carparams.csv - a can details with coresponding frame id.
            weather.txt - weather information in which the recording was made.

        To have images without low pass filter applied:
            Take 16bit images from 16BitFrames folder and open with OpenCV function like: Mat input = imread(<image_full_path>, -1);
            Then use convertTo function like: input.convertTo(output, input.depth(), sc, sh), where output is transformed Mat, sc is scale and sh is shift from carParams.csv file.
            Finally, scale image to 640x480
            
    *ZUT claims that the labeling is very high quality, where KAIST and SCUT failed to do so
    Group of people are all in one box, which can make the filter extraction very hard
    Therefore all the mentioned dataset has to go through some quality check

## KAIST Multispectral Pedestrian Detection:
    Specification:
        Published on 2015 
        FIR-Visible, Day-Night
        95k color-thermal pairs (640*480)
        62578 Day Frames = 70679 Objects
        32750 Night Frames = 44871 Objects

## CVC-14:
    Specification:
        Published on 2016
        Day-Night, FIR-Visible, FramesNeg-FramesPos available
        Pedestrian Only
        For training 3695 imegas during the day, and 3390 images during night, with ~1500 mandatory pedestrian annotated for each sequence.
        For testing ~700 images for both sequences with ~2000 pedestrian during day, and ~1500 pedestrian during night
        Frame format is tif, therefore I assume that they are 16-bit

## CVC-09:
    Specification:
        Published on 2014
        Day-Night, FIR Only, FramesNeg-FramesPos available
        Pedestrian Only
        2 Sets: The first set contains 5990 frames and the second 5081, divided in training and testing sets each sequence.
        Frame format is png, therefore I assume that they are 8-bit

## Multispectral Image Recognition:
    Specification:
        Published on 2017
        FIR-MIR-NIR-Visible, Label in xml and txt format for each sensor
        xml has additional info. such as truncated, crowd, and not normalized cordinates
        Paper is hard to find for more details about the dataset!
        Labels (.txt format) are same as Annotation_ConvertedSummarized but in yolo format
        ir_label.py is available to convert xml to txt (yolo format)
        noAnnotationsList.txt contains the images without labels
        7512 images in total for each sensor

## OTCBVS:
    Dataset 01: Pedestrian, above long shot
    Dataset 03: Pedestrian, above long shot


Master thesis has to be:
===================================
Specific:
    What question are you exactly trying to answer?
    What concept do you want to address in-depth?
Measurable:
    Performance metrics and qualitative assessments


Evaluation of master thesis:
===================================
    
mAP comparison between RGB only and RGBT network in Day and Night scenarios.
Since we do not have any weather related data (i.e. Fog, Sun glare, etc.) we cannot evaluate that.

If we want to have 3 classes, we have only FLIR.
The problem with FLIR is that the RGB images are not labeled and cannot be labeled with any
cross-correlation labeling algorithm precisely. Whatever trick we use will be an estimated label.

If we focus on pedestrian only, we have other datasets, which can help us, but then we have to
take care of cleaning many datasets and bring the labels from different dataset to the same format.
This can take time.

What we can idealy do is to train RGBT with what we have and compare RGB with RGBT networks
using some pictures taken from our own sensor in all the challenging scenarios.
But here there is high risk of failure since our datasets are very small to be able to generalize
this well to work with our self taken IR and RGB images, which are very different w.r.t. the data
we have trained the network with.

19.03.2021 Meeting output:
    compare Fused vs. only RGB -> Pedestrian Class
    DenseNet plus YoloV4 attention combination


Considered Papers to workon:
===================================
* 20-03-(Yolov3 PedestrianOnly)Pedestrian Detection in Severe Weather Condition.pdf