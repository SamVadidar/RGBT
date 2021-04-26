import cv2
import matplotlib.pyplot as plt
import time

from gt_bb_cords import get_cords
from arg_parser import DATASET_PP_PATH


def plot(ir_th, rgb_th, ir, rgb):

    fig = plt.figure(figsize=(50,50))
    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(ir_th , 'gray')
    plt.title('ir_th')

    fig.add_subplot(rows, cols, 2)
    plt.imshow(rgb_th, 'gray') # rgb_edge
    plt.title('rgb_th')

    fig.add_subplot(rows, cols, 3)
    plt.imshow(ir, 'gray')
    plt.title('IR')

    fig.add_subplot(rows, cols, 4)
    plt.imshow(rgb, 'gray')
    plt.title('rgb')

    plt.show()

def calc_para(path_ir, path_rgb, cropped_rgb_location=' ', scale_fact=2.44, method='dataset_original'):
    '''
    Here we try to find 3 transfer parameters (i.e. ScalingFactor, X_Offset, Y_Offset)
    Using the parameters, we can map each pixel in IR_Frame to RGB_Frame.
    Therefore, we can use IR_Labels for RGB_Frames as well.

    Return: max_val_glob, max_loc_glob, scale_w_glob
    '''

    # 0 if ir is tempelate image (Smaller)
    # 1 if rgb is tempelate image (Smaller)
    mode = 0

    ir = cv2.imread(path_ir, 0)
    ir_blur = cv2.GaussianBlur(ir, (5,5), 0)

    ir_low_threshold = 25
    ir_ratio = 3
    ir_kernelSize = 3
    ir_edge = cv2.Canny(ir_blur, ir_low_threshold, ir_low_threshold*ir_ratio, ir_kernelSize) # 75, 170
    ir_edge = cv2.GaussianBlur(ir_edge, (5,5), 0)
    _, ir_th = cv2.threshold(ir_edge,70,255,cv2.THRESH_BINARY)
    
    # 512 * 640
    ir_y, ir_x = ir.shape
    aspect_ratio_ir = ir_x/ir_y

    if method == 'from_cropped_rgb':
        rgb = cv2.imread(cropped_rgb_location, 0)
    else:
        rgb = cv2.imread(path_rgb, 0)

    rgb_blur = cv2.GaussianBlur(rgb, (5, 5), 0)

    rgb_th = cv2.adaptiveThreshold(rgb_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,2) # 19
    rgb_low_threshold = 65
    rgb_ratio = 3
    rgb_kernelSize = 3

    rgb_edge = cv2.Canny(rgb_th, rgb_low_threshold, rgb_low_threshold*rgb_ratio, rgb_kernelSize) # 100, 150
    rgb_edge = cv2.GaussianBlur(rgb_edge, (5,5), 0)

    # rgb_th = cv2.adaptiveThreshold(rgb_edge, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,2) # 19
    _, rgb_th = cv2.threshold(rgb_edge,90,255,cv2.THRESH_BINARY)
    # rgb_th = cv2.medianBlur(rgb_th,11)
    # rgb_th = cv2.GaussianBlur(rgb_th, (5, 5), 0)
    # rgb_th = cv2.GaussianBlur(rgb_th, (5, 5), 0)
    
    # 1600 * 1800
    rgb_y, rgb_x = rgb.shape
    aspect_ration_rgb = rgb_x/rgb_y

    plot(ir_th, rgb_th, ir, rgb)

    scale_w = scale_fact
    size_w = scale_w_glob = max_val_glob = max_loc_glob = 0

    # Until both frames have the same width
    #scale_w < (rgb_y/ir_y)
    while(size_w < rgb.shape[1]): # rgb.shape[1]
    # while(scale_fact < 2.4): # 2.5 comes from looking at the save_and_crop_history.txt and checking the good labelled rgb images
    #                           # I decieded to loop over scale factor instead of size_w, because it makes the preprocessing faster!

        # 0 if ir is tempelate image (Smaller)
        # 1 if rgb is tempelate image (Smaller)
        if(mode==1):
            temp = rgb.copy()
            temp_y, temp_x, _ = temp.shape
            shrink_percentage_w = (temp_x-(temp_x/scale_w))/temp_x

            size_w = (1-shrink_percentage_w) * temp_x
            size_h = (1-shrink_percentage_w) * temp_y
            temp_resized = cv2.resize(temp, (int(size_w), int(size_h)))

            res = cv2.matchTemplate(ir_th, temp_resized, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_val_glob:
                max_val_glob = max_val
                max_loc_glob = max_loc
                scale_w_glob = scale_w
            scale_w += 0.001
            # scale_w += 0.01

        elif(mode==0):
            temp_edge = ir_th.copy()
            temp_y, temp_x = temp_edge.shape
            size_w = temp_x * scale_w
            size_h = size_w / aspect_ratio_ir

            # temp_edge = cv2.Canny(cv2.GaussianBlur(temp, (3,3), 0), 75, 170)
            temp_edge_resized = cv2.resize(temp_edge, (int(size_w), int(size_h)))
            # print(temp_edge_resized.shape)

            res = cv2.matchTemplate(rgb_th, temp_edge_resized, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_val_glob:
                max_val_glob = max_val
                max_loc_glob = max_loc
                scale_w_glob = scale_w
            scale_w += 0.001
            # scale_w += 0.01

    print(f"scale: {scale_w_glob}")
    print(f"Offset: {max_loc_glob}")
    print(f"Val: {max_val_glob}")

    return max_val_glob, max_loc_glob, scale_w_glob


if __name__ == '__main__':
    path_train_set = DATASET_PP_PATH + '/train/'
    path_val_set = DATASET_PP_PATH + '/val/'
    imageNumber = 26

    # TODO (Fun_Fact)
    # 5251 Sun Glare!
    # [3670:4088], [135:186] Night
    #TODO
    # 51, 4423, 4431, 9833
    # 2.477 and (155, 155) works perfectly for all the frames, except the above mentioned!

    # 2nd try:
    # for 7 -> 2.551, (133, 130), 0.21358
    # for 15 -> 2.482, (154, 147), 0.2
    # for 26 -> 2.519, (157, 139), 0.154
    # for 51 -> !
    # for 70 -> 2.629, (111, 115), 0.15978
    # for 1655 -> !
    # for 4423 -> 
    # for 4431 -> Sync. Problem
    # for 4433 ->
    # for 9833 -> 2.479, (186, 148), 0.18

    # 3rd try: (THE BEST)
    # 7, 15, 26, 70, 9833, worked just same
    # 51 ! 2.486, (154, 141), .16915
    # for 1655 -> 2.454, (184, 154), 0.157286
    # for 4433 -> 2.482, (156, 144), 0.225

    bb_gtruth = get_cords(str(imageNumber).zfill(5))

    if imageNumber < 8863:
        path_ir = path_train_set + 'thermal_8_bit/FLIR_' + str(imageNumber).zfill(5) + ".jpeg"
        path_rgb = path_train_set + 'RGB/FLIR_' + str(imageNumber).zfill(5) + ".jpg"
        _, max_loc_glob, scale_w_glob = calc_para(path_ir, path_rgb)
        rgb = cv2.imread(path_rgb)
    else:
        path_ir = path_val_set + 'thermal_8_bit/FLIR_' + str(imageNumber).zfill(5) + ".jpeg"
        path_rgb = path_val_set + 'RGB/FLIR_' + str(imageNumber).zfill(5) + ".jpg"
        _, max_loc_glob, scale_w_glob = calc_para(path_ir, path_rgb)
        rgb = cv2.imread(path_rgb)

    # if rgb.shape != (1600, 1800, 3):
    #scale
    # scale_w_glob = 2.479#2.476
    #offset
    # max_loc_glob = (186, 148)
    # elif rgb.shape == (1536, 2048, 3):
    #     print(rgb.shape)
    # elif rgb.shape == (1024, 1280, 3):
    #     print(rgb.shape)
    # elif rgb.shape == (480, 720, 3):
    #     print(rgb.shape)

    #ir frame size
    temp_x = 640
    temp_y = 512

    x1 =max_loc_glob[0]
    y1 = max_loc_glob[1]
    pt1 = (int(x1), int(y1))

    x2 = max_loc_glob[0]+scale_w_glob*temp_x
    y2 = max_loc_glob[1]+scale_w_glob*temp_y
    pt2 = (int(x2), int(y2))

    # for bb in bb_gtruth:
    cv2.rectangle(rgb, pt1, pt2, (0, 255, 0), 1)

    # Fix the scale and draw all the boxes from ground truth coords.
    for bb in bb_gtruth:
        bb = [int(x * scale_w_glob) for x in bb]
        bb[0] += max_loc_glob[0]
        bb[1] += max_loc_glob[1]

        cv2.rectangle(rgb, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.show()