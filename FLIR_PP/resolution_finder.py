import cv2
import os
import glob
import pickle
from pathlib import Path


def res_list_creator(list_name, dataset_path, method=0):
    '''
    Here we creat a list of all the different RGB-frame resolutions in FLIR Dataset.
    We also count how many images are there per resolution(i.e. Camera).
    '''
    if os.path.isfile(list_name):
        os.remove(list_name)
    file = open(list_name, "a")
    previous_res = 0
    counter_1536 = 0
    counter_1600 = 0
    counter_1024 = 0
    counter_480 = 0

    # method 0
    if method == 0:
        for folder in glob.glob(str(dataset_path) + "*"):
            for img in Path(folder).rglob('*.jpg'):
                _, img_name = os.path.split(img)

                rgb_frame = cv2.imread(str(img))
                res = rgb_frame.shape
                if res == (1536, 2048, 3):
                    counter_1536 +=1
                elif res == (1600, 1800, 3):
                    counter_1600 +=1
                elif res == (1024, 1280, 3):
                    counter_1024 +=1
                elif res == (480, 720, 3):
                    counter_480 +=1

                if res != previous_res:
                    file.write(str(res) + '\t' + str(img_name) + '\n')
                    print((str(res) + '\t' + str(img_name)))
                        
                previous_res = res
    
    # method 1
    else:
        for root, d, files in os.walk(str(dataset_path), topdown=True):
            for img in files:
                # find jpg files
                if img[-2] == 'p':
                    path = os.path.join(root, img)
                    print(path)

                    rgb_frame = cv2.imread(str(path))
                    res = rgb_frame.shape

                    if res != previous_res:
                        file.write(str(res) + '\t' + str(path) + '\n')
                        print((str(res) + '\t' + str(path)))
                            
                    previous_res = res    

    print(counter_1536, counter_1600, counter_1024, counter_480)
    file.close()

def res_dictionary(list_name):
    '''
    This function read the long list of repeated resolution in the Dataset and
    print all the unique resolutions with one example (image_name).
    '''
    file = open(list_name, "r")
    lines = file.readlines()

    res_dict = {}
    for line in lines:
        frame_height = line.split(')', 1)[0]
        frame_height += ')'
        frame_name_index = line.find('.jpg')
        frame_name = line[frame_name_index-10:frame_name_index+4]

        # If the resolution was not already in the list
        if res_dict.get(str(frame_height)) == None:
            res_dict.update({str(frame_height): str(frame_name)})

    for key in res_dict:
        print(key, '->', res_dict.get(key))

def find_missing_frames(dataset_path, file_name, Sensor):
    # separate because in video folder the frame numbers start over!
    img_num_list_val_train = []
    img_num_list_video = []

    if os.path.isfile(file_name):
        user_input = input("Are you sure you want to redo and remove the previous rgb_missing_frames? (y/n)\n")
        if user_input == 'y':
            os.remove(file_name)
        if user_input == 'n':
            exit()
    file = open(file_name, "a")

    for folder in glob.glob(str(dataset_path) + "/*"):
        # for train and val folder
        # separate because in video folder the frame numbers start over!
        if(folder!= dataset_path + '/video' and Sensor == 'IR'):
            for img in Path(folder).rglob('*.jpeg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-10:-5])
                img_num_list_val_train.append(img_num)
        elif(folder!= dataset_path + '/video' and Sensor == 'RGB'):
            for img in Path(folder).rglob('*.jpg'):
                rgb_frame = cv2.imread(str(img))
                res = rgb_frame.shape
                # Only if the resolution is relevant for us
                if res == (1600, 1800, 3):
                    _, img_name = os.path.split(img)
                    img_num = int(str(img_name)[-9:-4])
                    img_num_list_val_train.append(img_num)
        # for video folder
        if(folder == dataset_path + '/video' and Sensor == 'IR'):
            for img in Path(folder).rglob('*.jpeg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-10:-5])
                img_num_list_video.append(img_num)
        elif(folder == dataset_path + '/video' and Sensor == 'RGB'):
            for img in Path(folder).rglob('*.jpg'):
                rgb_frame = cv2.imread(str(img))
                res = rgb_frame.shape
                # Only if the resolution is relevant for us
                if res == (1600, 1800, 3):
                    _, img_name = os.path.split(img)
                    img_num = int(str(img_name)[-9:-4])
                    img_num_list_video.append(img_num)
    img_num_list_val_train.sort()
    img_num_list_video.sort()

    file.write('TRAIN_VAL SET:' + '\n')

    for index, num in enumerate(img_num_list_val_train):
        # 1) index starts from 0 but num from 1
        # 2) We want to check if the next file is named as the current file
        if index<len(img_num_list_val_train)-1:
            if ((num+1)!=img_num_list_val_train[index+1]):
                file.write(str((num+1)) + '\n')
                print(index)
    
    file.write('END OF TRAIN_VAL SET.' + '\n')
    file.write('VIDEO SET:' + '\n')

    for index, num in enumerate(img_num_list_video):
        # 1) index starts from 0 but num from 1
        # 2) We want to check if the next file is named as the current file
        if index<len(img_num_list_video)-1:
            if ((num+1)!=img_num_list_video[index+1]):
                file.write(str((num+1)) + '\n')
                print(index)
    file.close()

def delete_rgb_missing_frames_from_ir(dataset_path, delete_list):
    file = open(delete_list, "r")
    lines = file.readlines()

    for folder in glob.glob(str(dataset_path) + "/*"):
        if(folder!= dataset_path + '/video'):
            for img in Path(folder).rglob('*.jpeg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-10:-5])
                for line in lines:
                    if(line.isnumeric and line == img_num):
                        os.remove(img)

def delete_rgb_low_res(dataset_path, file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)
    file = open(file_name, "a")
    for folder in glob.glob(str(dataset_path) + "/*"):
        # video set is 1800*1600 already
        if(folder!= dataset_path + '/video'):
            for img in Path(folder).rglob('*.jpg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-9:-4])
                rgb_frame = cv2.imread(str(img))
                res = rgb_frame.shape
                # Only if the resolution is relevant for us
                if res != (1600, 1800, 3):
                    os.remove(str(img))
                    file.write(str(img_num) + '\n')
    file.close()

def merge_two_files(file1, file2, merged_file):
    data = data2 = ""
  
    # Reading data from file1
    with open(file1) as fp:
        data = fp.read()
    
    # Reading data from file2
    with open(file2) as fp:
        data2 = fp.read()
    
    # Merging 2 files
    # To add the data of file2 from next line
    data += "\n"
    data += data2
    
    with open (merged_file, 'w') as fp:
        fp.write(data)

    fp.close()

def delete_frames_ir(dataset_path, delete_list):


if __name__ == "__main__":
    pp_dataset_path = "/home/ub145/Documents/Dataset/FLIR/FLIR_PP"

    # # find all the different available RGB resolutions
    # rgb_res_file_name = "./rgb_resolution_list.txt"
    # res_list_creator(rgb_res_file_name, dataset_path, method=0)
    # res_dictionary(rgb_res_file_name)

    # # find all the missing rgb images and make sure the images are synced here!
    # rgb_missing_frame_list = "./missing_frame_list_rgb.txt"
    # ir_missing_frame_list = "./missing_frame_list_ir.txt"
    # find_missing_frames(pp_dataset_path, rgb_missing_frame_list, Sensor='RGB')

    # # delete all the frames which have smaller HFOV than IR
    # rgb_deleted_low_res = "./deleted_low_res_rgb.txt"
    # delete_rgb_low_res(pp_dataset_path, rgb_deleted_low_res)

    # # merge all the deleted lists
    # total_deleted_rgb = './deleted_total_rgb.txt'
    # merge_two_files(rgb_missing_frame_list, rgb_deleted_low_res, total_deleted_rgb)

    # # delete all the rgb-deleted frames from IR

    # # find the parameters to pre-process the RGB images

    # # Pre-process RGB frames

    # # Check labels on RGB frames







