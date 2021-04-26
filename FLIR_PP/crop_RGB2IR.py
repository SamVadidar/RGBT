import cv2
import PIL 
from matplotlib import pyplot as plt
from gt_bb_cords import get_cords
from align_IR2RGB import calc_para
from align_IR2RGB import DATASET_PP_PATH


def crop_resolution_1800_1600(rgb_path, save_location, scale_width_glob, max_location_glob):

    # The parameters are calculate using align_IR2RGB script
    # max_location_glob = (155, 155) # Offset
    # scale_width_glob = 2.477
    IR_WIDTH = 640
    IR_HEIGHT = 512

    # path_train_set = '/home/ub145/Documents/Dataset/FLIR/FLIR/train/'
    # rgb = cv2.imread(rgb_path)
    rgb = PIL.Image.open(rgb_path)
    # height
    y1 = max_location_glob[1]
    y2 = int(max_location_glob[1]+scale_width_glob*IR_HEIGHT)
    # width
    x1 = max_location_glob[0]
    x2 = int(max_location_glob[0]+scale_width_glob*IR_WIDTH)

    # rgb_cropped = rgb[y1:y2, x1:x2]
    rgb_cropped = rgb.crop((x1, y1, x2, y2)) # left, top, right, bottom
    rgb_cropped = rgb_cropped.resize((640, 512), PIL.Image.ANTIALIAS)
    rgb_cropped.save(save_location)

    # cv2.imwrite(save_location, rgb_cropped)
    # rgb_cropped_height, rgb_cropped_width, _ = rgb_cropped.shape

    # _, max_loc_glob, scale_w_glob = calc_para(path_train_set, imageNumber, save_location, scale_fact=2.45, method='from_cropped_rgb')
    # # Crop the offset
    # rgb_cropped = rgb_cropped[max_loc_glob[1]:, max_loc_glob [0]:]

    # cropped_resize = cv2.resize(rgb_cropped, (int(rgb_cropped_width/scale_w_glob), int(rgb_cropped_height/scale_w_glob)))

    # print(cropped_resize.shape)

    # ir = cv2.imread(path_train_set + 'thermal_8_bit/FLIR_' + imageNumber + ".jpeg")
    # # plot(rgb_cropped, cropped_resize, ir)
    # plot_bb(imageNumber, cropped_resize)

def plot(cropped, resized, IR):
    fig = plt.figure(figsize=(50,50))
    rows = 1
    cols = 3

    fig.add_subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title('cropped')

    fig.add_subplot(rows, cols, 2)
    plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    plt.title('resized')

    fig.add_subplot(rows, cols, 3)
    plt.imshow(cv2.cvtColor(IR, cv2.COLOR_BGR2RGB))
    plt.title('IR')

    plt.show()

def plot_bb(imageNumber, cropped_frame):
    
    bb_gtruth = get_cords(imageNumber)
    for bb in bb_gtruth:
        # cropped frame has to be in the IR coordinate system, so that the boxes match
        cv2.rectangle(cropped_frame, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0, 0, 255), 1)
    
    plt.imshow(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == '__main__':
    imageNumber = '26'
    path_rgb = DATASET_PP_PATH + '/train/RGB/FLIR_' + str(imageNumber).zfill(5) + ".jpg"
    scale_width_glob = 2.542
    max_location_glob = (156, 136)
    crop_resolution_1800_1600(path_rgb, './rgb_cropped.png', scale_width_glob, max_location_glob)

    cropped_frame = cv2.imread('./rgb_cropped.png')
    plot_bb(str(imageNumber).zfill(5), cropped_frame)
