import cv2
import os
import glob
import pickle
from pathlib import Path
import shutil


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
        for root, _, files in os.walk(str(dataset_path), topdown=True):
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
        # All RGB are 1800*1600, therefore we do not check the resolution
        if(folder == dataset_path + '/video' and Sensor == 'IR'):
            for img in Path(folder).rglob('*.jpeg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-10:-5])
                img_num_list_video.append(img_num)
        elif(folder == dataset_path + '/video' and Sensor == 'RGB'):
            for img in Path(folder).rglob('*.jpg'):
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
                file.write(str((num+1)).zfill(5) + '\n')
                print(index)
    
    file.write('END OF TRAIN_VAL SET.' + '\n')
    file.write('VIDEO SET:' + '\n')

    for index, num in enumerate(img_num_list_video):
        # 1) index starts from 0 but num from 1
        # 2) We want to check if the next file is named as the current file
        if index<len(img_num_list_video)-1:
            if ((num+1)!=img_num_list_video[index+1]):
                file.write(str((num+1)).zfill(5) + '\n')
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
    pass

# both end frame and start frames are also included in the rename range
def rename_rgb_frames_to_sync(dataset_path, start_frame_num, end_frame_num, added_amount):
    done_jobs = 0
    rgb_folder = dataset_path + '/video/rgb/'

    num_of_frame = (end_frame_num - start_frame_num)

    # if we rename to increase start from the last image
    # else start from the first to decrease the file string number
    # to avoid conflict of two files with same name
    if added_amount>0:
        img_num = end_frame_num
    else:
        img_num = start_frame_num

    while done_jobs <= num_of_frame:
        img_name = rgb_folder + 'FLIR_video_' + str(img_num).zfill(5) + '.jpg'

        img_num_new = img_num + added_amount
        img_name_new = rgb_folder + 'FLIR_video_' + str(img_num_new).zfill(5) + '.jpg'

        if os.path.exists(img_name_new) == True:
            raise ValueError('The following file exist, cannot rename!: ' + str(img_name_new))
        else:
            os.rename(img_name, img_name_new)
            print(str(img_num).zfill(5), ' is renamed to: ', str(img_num_new).zfill(5), ' ', done_jobs)
            
            done_jobs += 1
            if added_amount>0:
                img_num -= 1
            else:
                img_num += 1

def remove_frames(dataset_path, start_frame_num, end_frame_num, sensor):
    done_jobs = 0
    num_of_frame = (end_frame_num - start_frame_num)
    img_num = end_frame_num
    folder = ' '
    while done_jobs <= num_of_frame:
        if sensor=='rgb':
            folder = dataset_path + '/video/rgb/'
            img_name = folder + 'FLIR_video_0' + str(img_num) + '.jpg'
            os.remove(img_name)
            print(str(img_name), ' is removed. ', done_jobs)
            # until we reach the start_frame_num
            done_jobs += 1
            img_num -= 1
        elif sensor=='ir':
            folder = dataset_path + '/video/thermal_8_bit/'
            img_name = folder + 'FLIR_video_0' + str(img_num) + '.jpeg'
            os.remove(img_name)
            print(str(img_name), ' is removed. ', done_jobs)
            # until we reach the start_frame_num
            done_jobs += 1
            img_num -= 1
        else:
            raise ValueError('PLEASE SELECT THE LAST ARGUMENT OF THE FUNCTION!')

def sync_video_set(dataset_path):
    # # #step 1
    # remove_frames(dataset_path, 4224, 4224, 'ir')
    # rename_rgb_frames_to_sync(dataset_path, 3154, 4224, added_amount=-1)

    # # step 2
    # rename_rgb_frames_to_sync(dataset_path, 2800, 3151, added_amount=1)
    # remove_frames(dataset_path, 3153, 3153, 'rgb')
    # remove_frames(dataset_path, 3153, 3153, 'ir')

    # # step 3
    # rename_rgb_frames_to_sync(dataset_path, 1945, 2772, added_amount=28)
    # remove_frames(dataset_path, 2800, 2800, 'rgb')
    # remove_frames(dataset_path, 2800, 2800, 'ir')
    # remove_frames(dataset_path, 1945, 1972, 'ir')

if __name__ == "__main__":
    pp_dataset_path = "/home/sam/Documents/Dataset/FLIR/FLIR_PP"

    # # find all the different available RGB resolutions
    # rgb_res_file_name = "./rgb_resolution_list.txt"
    # res_list_creator(rgb_res_file_name, pp_dataset_path, method=0)
    # res_dictionary(rgb_res_file_name)

    # # find all the missing rgb images
    # rgb_missing_frame_list = "./missing_frame_list_rgb.txt" # Result: 333 rgb frames are missing
    # ir_missing_frame_list = "./missing_frame_list_ir.txt" # Result: no IR frame is missing
    # find_missing_frames(pp_dataset_path, rgb_missing_frame_list, Sensor='RGB')

    # Sync ir-rgb frames in video set
    sync_video_set(pp_dataset_path) # video set is ready for cross labelling

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