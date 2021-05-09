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

    f = open(('./FLIR_PP/obj_count_'  + set_folder + '.txt'), 'w')
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
    # in case you had calc_para = True for every image and made a text file of the parameters
    # history_file = open('./FLIR_PP/save_and_crop_history.txt', 'r')
    # history_reader = csv.reader(history_file, delimiter='\t')

    for img in Path(rgb_cropped_set_folder).rglob('*.jpg'):
        iteration += 1
        print_progress(iteration, total_file_num)
        scale_fact = 0

        rgb = cv2.imread(str(img))
        _, rgb_name = os.path.split(img)
        rgb_num = int(str(rgb_name)[-9:-4])

        # # in case you had calc_para = True for every image and made a text file of the parameters
        # # get the scaling factor from the text file
        # for line in history_reader:
        #     if line[0] == rgb_name:
        #         scale_fact = float(line[1])
        #         break

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
def draw_rgb_annotation_from_json(dataset_path, set_folder):
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

# COCO to YOLO
def convert_labels_to_yolo_format(dataset_path, which_set):
    path_to_set = dataset_path + '/' + str(which_set) + '/'
    json_path = dataset_path + '/' + str(which_set) + '/thermal_annotations.json'
    dst_path = os.path.join(path_to_set, 'yolo_format_labels')

    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    else:
        shutil.rmtree(dst_path)
        os.mkdir(dst_path)

    f = open(json_path, 'r')
    data = json.load(f)
    images = data['images']
    annotations = data['annotations']

    file_names = []
    for i in range(0, len(images)):
        file_names.append(images[i]['file_name'])

    total_file_num = int(len(images))
    iteration = 0

    width = 640.0
    height = 512.0

    for i in range(0, len(images)):
        converted_results = []
        for ann in annotations:
            if ann['image_id'] == i and ann['category_id'] <= 3:
                cat_id = int(ann['category_id'])
                # if cat_id <= 3:
                left, top, bbox_width, bbox_height = map(float, ann['bbox'])

                # Yolo classes are starting from zero index
                cat_id -= 1
                x_center, y_center = (
                    left + bbox_width / 2, top + bbox_height / 2)

                # Yolo expects relative values wrt image width&height
                x_rel, y_rel = (x_center / width, y_center / height)
                w_rel, h_rel = (bbox_width / width, bbox_height / height)
                converted_results.append(
                    (cat_id, x_rel, y_rel, w_rel, h_rel))
        image_name = images[i]['file_name']
        image_name = image_name[14:-5]
        # print(image_name)
        file = open(str(dst_path) + '/' + str(image_name) + '.txt', 'w+')
        file.write('\n'.join('%d %.6f %.6f %.6f %.6f' % res for res in converted_results))
        file.close()
        iteration += 1
        print_progress(iteration, total_file_num)

# merge manually labelled data to the original labels
def merge_labels(dataset_path, secondary_folder):

    for secondary_file in Path(secondary_folder).rglob('*.txt'):
        _, secondary_file_name = os.path.split(secondary_file)
        img_num = int(str(secondary_file_name)[-9:-4])
        if img_num < 8863:
            main_folder = dataset_path + '/train/yolo_format_labels'
        # labels are corrected only for train and val sets
        # video set is already clean and complete
        else:
            main_folder = dataset_path + '/val/yolo_format_labels'
        for main_file in Path(main_folder).rglob('*.txt'):
            _, main_file_name = os.path.split(main_file)
            if main_file_name == secondary_file_name:
                f_main = open(main_file, 'a')
                f_secondary = open(secondary_file, 'r')

                # main_data_lines = f_main.readlines()
                sec_data_lines = f_secondary.readlines()
                f_main_size = os.path.getsize(main_file)
                if f_main_size != 0:
                    f_main.write('\n')
                for line in sec_data_lines:
                    f_main.writelines(line)

                f_main.close()
                f_secondary.close()
                print(main_file_name, ' manually added labels are merged!')

def correct_LabelImg_classes():
    for secondary_file in Path('/home/ub145/dev/RGBT/FLIR_PP/manual_data_cleaning/label_RGB_manual').rglob('*.txt'):
        _, secondary_file_name = os.path.split(secondary_file)
        img_num = int(str(secondary_file_name)[-9:-4])
        f = open(secondary_file, 'r')
        f_dst = open('/home/ub145/dev/RGBT/FLIR_PP/manual_data_cleaning/label_RGB_manual_corrected/'+str(secondary_file_name), 'a')

        lines = f.readlines()
        for line in lines:
            if line[0] == '4':
                f_dst.write(line.replace('4', '2', 1))
                print(secondary_file_name)
            else:
                f_dst.write(line)

def plot_yoloFormat_labels(dataset_path, which_set):
    rgb_folder_path = dataset_path + '/' + which_set + '/RGB_cropped'
    dst_path = rgb_folder_path + '_annotated_YoloFormat'
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    for img in Path(rgb_folder_path).rglob('*.jpg'):
        rgb = cv2.imread(str(img))
        _, rgb_name = os.path.split(img)
        rgb_label = rgb_name.replace('.jpg', '.txt')
        rgb_label_path = dataset_path + '/' + which_set + '/yolo_format_labels/' + rgb_label
        f = open(rgb_label_path, 'r')
        lines = f.readlines()

        for line in lines:
            line = line.split()
            x1 = int(float(line[1])*640 - float(line[3])*640/2) # x - w/2 = x1
            y1 = int(float(line[2])*512 - float(line[4])*512/2) # y - h/2 = y1
            x2 = int(float(line[3])*640 + x1)
            y2 = int(float(line[4])*512 + y1)
            cv2.rectangle(rgb, (x1,y1), (x2,y2), (255,0,0))
        plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        plt.show()


if __name__ == '__main__':
    # draw_rgb_annotation_from_json(DATASET_PP_PATH, 'val')
    count_objects_all(DATASET_PP_PATH)

    # convert_labels_to_yolo_format(DATASET_PP_PATH, 'train')
    # convert_labels_to_yolo_format(DATASET_PP_PATH, 'val')
    # convert_labels_to_yolo_format(DATASET_PP_PATH, 'video')

    # manually_added_labels = './FLIR_PP/manual_data_cleaning/label_RGB_manual'
    # merge_labels(DATASET_PP_PATH, manually_added_labels)
    plot_yoloFormat_labels(DATASET_PP_PATH, 'train')