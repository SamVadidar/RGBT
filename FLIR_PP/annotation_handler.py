import json
import csv
import os
import shutil
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from arg_parser import DATASET_PP_PATH

def print_progress(iteration, total_file_num, decimals = 1, length = 100, fill = '#', prefix = 'Processing:', suffix = '', print_end = '\r'):
    percent = ("{0:." + str(decimals) + "f}").format(iteration * 100 / float(total_file_num))
    filledLength = int(length * iteration // total_file_num)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = print_end)

def count_objects(dataset_path, set_folder):
    json_path = os.path.join(dataset_path, set_folder) + '/thermal_annotations.json'
    json_file = open(json_path, 'r')
    json_data = json.load(json_file)

    person_obj = bicycle_obj = car_obj = iscrowd_obj = 0

    for i in json_data['annotations']:
        if int(i['category_id']) == 1:
            person_obj += 1
        elif int(i['category_id']) == 2:
            bicycle_obj += 1
        elif int(i['category_id']) == 3:
            car_obj += 1
        elif int(i['iscrowd']) == 1:
            iscrowd_obj += 1
            print(i['image_id'])

    f = open(('./obj_count_'  + set_folder + '.txt'), 'w')
    f.write('person_obj: ' + str(person_obj) + '\n' + 
            'bicycle_obj: ' + str(bicycle_obj) + '\n' +
            'car_obj: ' + str(car_obj) + '\n' +
            'iscrowd_obj: ' + str(iscrowd_obj) + '\n'
            )

    json_file.close()
    f.close()

    return person_obj, bicycle_obj, car_obj, iscrowd_obj

def count_objects_all(dataset_path):
    person_obj_val, bicycle_obj_val, car_obj_val, iscrowd_obj_val = count_objects(dataset_path, 'val')
    person_obj_train, bicycle_obj_train, car_obj_train, iscrowd_obj_train = count_objects(dataset_path, 'train')
    person_obj_video, bicycle_obj_video, car_obj_video, iscrowd_obj_video = count_objects(dataset_path, 'video')

    person_obj_total = person_obj_val + person_obj_train + person_obj_video
    bicycle_obj_total = bicycle_obj_val + bicycle_obj_train + bicycle_obj_video
    car_obj_total = car_obj_val + car_obj_train + car_obj_video
    print(' ')
    print('Number of Person Obj.:', person_obj_total)
    print('Number of Bicycle Obj.:', bicycle_obj_total)
    print('Number of Car Obj.:', car_obj_total)
    
def draw_and_save(dataset_path, set_folder, rgb_cropped_annotated_folder, json_data):
    rgb_cropped_set_folder = os.path.join(dataset_path, set_folder) + '/RGB_cropped'
    total_file_num = sum([len(files) for r, d, files in os.walk(rgb_cropped_set_folder)])
    iteration = 0
    history_file = open('./save_and_crop_history.txt', 'r')
    history_reader = csv.reader(history_file, delimiter='\t')

    # history_line = history_file.readlines()

    for img in Path(rgb_cropped_set_folder).rglob('*.jpg'):
        print_progress(iteration, total_file_num)
        iteration += 1
        scale_fact = 0

        rgb = cv2.imread(str(img))
        _, rgb_name = os.path.split(img)
        rgb_num = int(str(rgb_name)[-9:-4])

        # get the scaling factor from the text file
        for line in history_reader:
            if line[0] == rgb_name:
                scale_fact = float(line[1])
                break

        # get the bbox cords from json file
        for i in json_data['annotations']:
            # frames in val folder start from 8863, but in json file from 0!
            if set_folder == 'val':
                if int(i['image_id']) == rgb_num - 8863 :
                    bbox = i['bbox']
                    # bbox = [int(bb * scale_fact) for bb in bbox]
                    cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)

            else:
                # frames start from 1 but json file from 0
                if int(i['image_id']) == rgb_num -1 :
                    bbox = i['bbox']
                    # bbox = [int(bb * scale_fact) for bb in bbox]
                    cv2.rectangle(rgb, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)

        cv2.imwrite(str(os.path.join(rgb_cropped_annotated_folder, rgb_name)), rgb)
        # plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        # plt.show()

# train/val/video set as second argument
def draw_rgb_annotation(dataset_path, set_folder):
    rgb_cropped_annotated_folder = os.path.join(dataset_path, set_folder) + '/RGB_cropped_annotated'
    json_path = os.path.join(dataset_path, set_folder) + '/thermal_annotations.json'

    if os.path.isdir(rgb_cropped_annotated_folder):
        user_input = input("Are you sure you want to redo the 'rgb-annotation and save' process? (y/n)\n")
        if user_input == 'y':
            shutil.rmtree(rgb_cropped_annotated_folder)
            os.mkdir(rgb_cropped_annotated_folder)
        else:
            print('Process is cancelled')
            exit()
    else:
        os.mkdir(rgb_cropped_annotated_folder)

    json_file = open(json_path, 'r')
    json_data = json.load(json_file)
    draw_and_save(dataset_path, set_folder, rgb_cropped_annotated_folder, json_data)
    json_file.close()


if __name__ == '__main__':
    draw_rgb_annotation(DATASET_PP_PATH, 'val')
    # count_objects_all(DATASET_PP_PATH)