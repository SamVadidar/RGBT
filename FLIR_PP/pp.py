import cv2
import os
import glob
import pickle
from pathlib import Path
import shutil

from arg_parser import DATASET_PATH, DATASET_PP_PATH
from align_IR2RGB import calc_para
from crop_RGB2IR import crop_resolution_1800_1600
from annotation_handler import draw_rgb_annotation
from annotation_handler import print_progress


def make_FLIR_PP_folder(dataset_path):
    FLIR_PP_Path = os.path.join(dataset_path, 'FLIR_PP')
    if os.path.isdir(FLIR_PP_Path):
        print(FLIR_PP_Path, ' already exist!')
        user_input = input("Are you sure you want to redo and remove the previous 'FLIR_PP' folder? (y/n)\n")
        if user_input == 'y':
            shutil.rmtree(FLIR_PP_Path)
            os.mkdir(FLIR_PP_Path)
        if user_input == 'n':
            exit()

    for folder in glob.glob(str(dataset_path) + "/*"):
        if folder != FLIR_PP_Path and os.path.isdir(folder):
            _, set_name = os.path.split(folder)
            SET_PP_Path = os.path.join(FLIR_PP_Path, set_name)

            os.mkdir(SET_PP_Path)
            # I KNOW IT IS STRANGE! BUT FLIR DATASET COMES INCOHERENT IN NAMING
            # IT IS JUST AN SMALL EXAMPLE!
            if set_name == 'video':
                src = os.path.join(folder, 'rgb')
                dst = os.path.join(SET_PP_Path, 'rgb')
            else:
                src = os.path.join(folder, 'RGB')
                dst = os.path.join(SET_PP_Path, 'RGB')

            print('Copying RGB frames from ', str(set_name), 'Set')
            shutil.copytree(src, dst)

            src = os.path.join(folder, 'thermal_8_bit')
            dst = os.path.join(SET_PP_Path, 'thermal_8_bit')
            print('Copying thermal_8_bit frames from ', str(set_name), 'Set')
            shutil.copytree(src, dst)

            src = os.path.join(folder, 'thermal_annotations.json')
            dst = os.path.join(SET_PP_Path, 'thermal_annotations.json')
            print('Copying thermal_annotations from ', str(set_name), 'Set')
            shutil.copy(src, dst)
            
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

def find_frame_num_gap(dataset_path, file_name, Sensor, resolution_check=False):
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
        if(folder!= dataset_path + '/video' and Sensor == 'ir'):
            for img in Path(folder).rglob('*.jpeg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-10:-5])
                img_num_list_val_train.append(img_num)
        elif(folder!= dataset_path + '/video' and Sensor == 'rgb'):
            for img in Path(folder).rglob('*.jpg'):
                if resolution_check == True:
                    rgb_frame = cv2.imread(str(img))
                    res = rgb_frame.shape
                    # Only if the resolution is relevant for us
                    if res == (1600, 1800, 3):
                        _, img_name = os.path.split(img)
                        img_num = int(str(img_name)[-9:-4])
                        img_num_list_val_train.append(img_num)
                else:
                    _, img_name = os.path.split(img)
                    img_num = int(str(img_name)[-9:-4])
                    img_num_list_val_train.append(img_num)
        # for video folder
        # All RGB are 1800*1600, therefore we do not check the resolution
        if(folder == dataset_path + '/video' and Sensor == 'ir'):
            for img in Path(folder).rglob('*.jpeg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-10:-5])
                img_num_list_video.append(img_num)
        elif(folder == dataset_path + '/video' and Sensor == 'rgb'):
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

def delete_rgb_lowRes_and_blankFrames(dataset_path, file_name):
    if os.path.isfile(file_name):
        user_input = input("Are you sure you want to redo and remove the previous file? (y/n)\n")
        if user_input == 'y':
            os.remove(file_name)
        else:
            exit()
    file = open(file_name, "a")
    for folder in glob.glob(str(dataset_path) + "/*"):
        # video set is 1800*1600 and clean already
        if(folder!= dataset_path + '/video'):
            for img in Path(folder).rglob('*.jpg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-9:-4])
                img_size = os.path.getsize(str(img))
                rgb_frame = cv2.imread(str(img))
                res = rgb_frame.shape
                # Only if the resolution is relevant for us
                # Or size is less than 60kB
                if res != (1600, 1800, 3) or img_size<60000:
                    os.remove(str(img))
                    print(str(img), ' is removed.')
                    file.write(str(img_num).zfill(5) + '\n')
    file.close()

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

def remove_frames(dataset_path, start_frame_num, end_frame_num, sensor, which_set):
    done_loops = 0
    num_of_frame = (end_frame_num - start_frame_num)
    img_num = end_frame_num
    folder = ' '
    while done_loops <= num_of_frame:
        # for Video set
        if sensor=='rgb' and which_set == 'video':
            folder = dataset_path + '/video/rgb/'
            img_name = folder + 'FLIR_video_' + str(img_num).zfill(5) + '.jpg'
        elif sensor=='ir' and which_set == 'video':
            folder = dataset_path + '/video/thermal_8_bit/'
            img_name = folder + 'FLIR_video_' + str(img_num).zfill(5)  + '.jpeg'
        # for Train
        elif sensor=='rgb' and which_set == 'train':
            folder = dataset_path + '/train/rgb/'
            img_name = folder + 'FLIR_' + str(img_num).zfill(5)  + '.jpg'
        elif sensor == 'ir' and which_set == 'train':
            folder = dataset_path + '/train/thermal_8_bit/'
            img_name = folder + 'FLIR_' + str(img_num).zfill(5)  + '.jpeg'
        # for Val
        elif sensor=='rgb' and which_set != 'val':
            folder = dataset_path + '/val/rgb/'
            img_name = folder + 'FLIR_' + str(img_num).zfill(5)  + '.jpg'
        elif sensor == 'ir' and which_set != 'val':
            folder = dataset_path + '/val/thermal_8_bit/'
            img_name = folder + 'FLIR_' + str(img_num).zfill(5)  + '.jpeg'
        else:
            raise ValueError('Something went wrong in remove_frames function!')
        try:
            os.remove(img_name)
            print(str(img_name), ' is removed. ', done_loops)
            # until we reach the start_frame_num
            done_loops += 1
            img_num -= 1

        except:
            # until we reach the start_frame_num
            done_loops += 1
            img_num -= 1
            continue

def sync_video_set(dataset_path):
    # step 1
    remove_frames(dataset_path, 4224, 4224, 'ir', 'video')
    rename_rgb_frames_to_sync(dataset_path, 3154, 4224, added_amount=-1)

    # step 2
    rename_rgb_frames_to_sync(dataset_path, 2800, 3151, added_amount=1)
    remove_frames(dataset_path, 3153, 3153, 'rgb', 'video')
    remove_frames(dataset_path, 3153, 3153, 'ir', 'video')

    # step 3
    rename_rgb_frames_to_sync(dataset_path, 1945, 2772, added_amount=28)
    remove_frames(dataset_path, 2800, 2800, 'rgb', 'video')
    remove_frames(dataset_path, 2800, 2800, 'ir', 'video')
    remove_frames(dataset_path, 1945, 1972, 'ir', 'video')

def sync_train_val_set(dataset_path, file_name):
    rgb_list = []
    f = open(file_name, 'a')

    for folder in glob.glob(str(dataset_path) + "/*"):
        if(folder!= dataset_path + '/video'):
            for img in Path(folder).rglob('*.jpg'):
                _, img_name = os.path.split(img)
                img_num = int(str(img_name)[-9:-4])
                rgb_list.append(str(img_num).zfill(5))

    for folder in glob.glob(str(dataset_path) + "/*"):
        if(folder!= dataset_path + '/video'):
            for img in Path(folder).rglob('*.jpeg'):
                _, ir_name = os.path.split(img)
                ir_num = int(str(ir_name)[-10:-5])
                ir_num = str(ir_num).zfill(5)
                if not ir_num in rgb_list:
                    os.remove(img)
                    f.write(ir_num + '\n')
                    print(img, ' is removed')

def crop_resize_save(dataset_path, history_file_path, calc_parameter = False):
    rgb_cropped_folder = dataset_path + '/rgb_cropped'
    total_file_num = int(sum([len(files) for r, d, files in os.walk(dataset_path)])*0.5) # 0.5, since each image pair is one iteration
    iteration = 0

    if os.path.isdir(rgb_cropped_folder):
        user_input = input("Are you sure you want to redo the 'crop, resize and save' process? (y/n)\n")
        if user_input == 'y':
            shutil.rmtree(rgb_cropped_folder)
            os.mkdir(rgb_cropped_folder)
            f = open(history_file_path, 'a')
        else:
            print('Process is cancelled')
            exit()
    else:
        os.mkdir(rgb_cropped_folder)
        f = open(history_file_path, 'a')

    for folder in glob.glob(str(dataset_path) + '/*'):
        for img in Path(folder).rglob('*.jpg'):
            print_progress(iteration, total_file_num)
            iteration += 1

            _, rgb_name = os.path.split(img)
            rgb_num = int(str(rgb_name)[-9:-4])
            ir_matched_path = os.path.join(dataset_path, folder) + '/thermal_8_bit/' + rgb_name[:-3] + 'jpeg'
            if calc_parameter == True:
                max_val_glob, max_loc_glob, scale_w_glob = calc_para(str(ir_matched_path), str(img))
                f.write(str(rgb_name) + '\t' + str(scale_w_glob) + '\t' + str(max_loc_glob) + '\t' + str(max_val_glob) + '\n')
            else:
                scale_w_glob = 2.479
                max_loc_glob = (158, 148)
                f.close()
            crop_resolution_1800_1600(str(img), os.path.join(rgb_cropped_folder, rgb_name), scale_w_glob, max_loc_glob)
    f.close()

def make_subfolders(rgb_cropped_folder, dataset_path):
    train_folder = dataset_path + '/train/RGB_cropped'
    val_folder = dataset_path + '/val/RGB_cropped'
    video_folder = dataset_path + '/video/RGB_cropped'

    print('Train folder exist!') if os.path.isdir(train_folder) else os.mkdir(train_folder)
    print('Val folder exist!') if os.path.isdir(val_folder) else os.mkdir(val_folder)
    print('Video folder exist!') if os.path.isdir(video_folder) else os.mkdir(video_folder)

    for img in Path(rgb_cropped_folder).rglob('*.jpg'):
        _, rgb_name = os.path.split(img)
        rgb_num = int(str(rgb_name)[-9:-4])

        if rgb_num < 8863 and ('video' not in rgb_name):
            dst = os.path.join(train_folder, rgb_name)
            print(dst)
            shutil.move(img, dst)
        elif rgb_num >= 8863 and ('video' not in rgb_name):
            dst = os.path.join(val_folder, rgb_name)
            print(dst)
            shutil.move(img, dst)
        elif rgb_num =< 4223 and ('video' in rgb_name):
            dst = os.path.join(video_folder, rgb_name)
            print(dst)
            shutil.move(img, dst)
        else:
            print('Strange attitude for file: ', str(img))

    os.rmdir(rgb_cropped_folder)


# if __name__ == "__main__":
    # # Backup the main Dataset folder and work on a subdirectory
    # make_FLIR_PP_folder(DATASET_PATH)
    
    # # find all the different available RGB resolutions
    # rgb_res_file_name = "./rgb_resolution_list.txt"
    # res_list_creator(rgb_res_file_name, DATASET_PP_PATH, method=0)
    # res_dictionary(rgb_res_file_name)

    # # find all the missing rgb images
    # rgb_missing_frame_list = "./missing_frame_list_rgb.txt" # Result: 333 rgb frames are missing
    # ir_missing_frame_list = "./missing_frame_list_ir.txt" # Result: no IR frame is missing
    # find_frame_num_gap(DATASET_PP_PATH, rgb_missing_frame_list, Sensor='RGB')

    # # Sync ir-rgb frames in video set
    # sync_video_set(DATASET_PP_PATH) # video set is ready for cross labelling

    # # delete all the frames which have smaller HFOV than IR + all the blank RGB images
    # rgb_deleted_lowRes_and_blankFrames = "./deleted_lowRes_and_blankFrames_rgb.txt"
    # delete_rgb_lowRes_and_blankFrames(DATASET_PP_PATH, rgb_deleted_lowRes_and_blankFrames)

    # # delete all the rgb-deleted frames from IR (non existing rgb images from IR)
    # sync_train_val_set(DATASET_PP_PATH, './final_ir_delete_from_train_val.txt')

    # # Pre-process RGB frames - Crop, resize and save
    # crop_resize_save(DATASET_PP_PATH, './save_and_crop_history.txt')
    # rgb_cropped_folder = DATASET_PP_PATH + '/rgb_cropped'
    # make_subfolders(rgb_cropped_folder, DATASET_PP_PATH)

    # # Check labels on RGB frames
    # draw_rgb_annotation(DATASET_PP_PATH, 'train')
    # draw_rgb_annotation(DATASET_PP_PATH, 'val')
    # draw_rgb_annotation(DATASET_PP_PATH, 'video')

# TODO
'''
Go through all the images and write the frame number in a file for:
    1) failed image 86, 65, 50, 338, 362, 416, 498, 658
    2) IR miss labelled images # 74, 78, 79, 94, 97, ...
    3) non synced images # 87, 408, 420, 421

Test the RGB frames with Yolo and see what is the baseline for the RGB
Later we will need this to compare the Fused version to RGB 
'''