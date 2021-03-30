import cv2
import matplotlib.pyplot as plt
import time

from gt_bb_cords import get_cords


def plot(ir_edge, rgb_edge, ir, rgb_th3):

    fig = plt.figure(figsize=(50,50))
    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(cv2.cvtColor(ir_edge, cv2.COLOR_BGR2RGB))
    plt.title('ir_edge')

    fig.add_subplot(rows, cols, 2)
    plt.imshow(cv2.cvtColor(rgb_edge, cv2.COLOR_BGR2RGB)) # rgb_edge
    plt.title('rgb_edge')

    fig.add_subplot(rows, cols, 3)
    plt.imshow(cv2.cvtColor(ir, cv2.COLOR_BGR2RGB))
    plt.title('IR')

    fig.add_subplot(rows, cols, 4)
    plt.imshow(cv2.cvtColor(rgb_th3, cv2.COLOR_BGR2RGB)) # th1
    plt.title('rgb_th3')

    plt.show()

def calc_para(path, imageNumber):

    path_ir = path + 'thermal_8_bit/FLIR_' + imageNumber+ '.jpeg'
    path_rgb = path + 'RGB/FLIR_' + imageNumber + '.jpg'

    # 0 if ir is tempelate image (Smaller)
    # 1 if rgb is tempelate image (Smaller)
    mode = 0

    ir = cv2.imread(path_ir)
    ir_blur = cv2.GaussianBlur(ir, (5,5), 0)

    ir_low_threshold = 50 # 70 for 1800*1600
    ir_ratio = 3
    ir_kernelSize = 3
    ir_edge = cv2.Canny(ir_blur, ir_low_threshold, ir_low_threshold*ir_ratio, ir_kernelSize) # 75, 170
    ir_edge = cv2.GaussianBlur(ir_edge, (5,5), 0)
    
    ir_y, ir_x, _ = ir.shape
    aspect_ratio_ir = ir_x/ir_y

    rgb = cv2.imread(path_rgb, 0)
    rgb_blur = cv2.GaussianBlur(rgb, (5, 5), 0)

    rgb_th3 = cv2.adaptiveThreshold(rgb, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,19,2)
    rgb_th3 = cv2.medianBlur(rgb_th3,11)
    rgb_th3 = cv2.GaussianBlur(rgb_th3, (5, 5), 0)
    rgb_th3 = cv2.GaussianBlur(rgb_th3, (5, 5), 0)

    rgb_low_threshold = 65
    rgb_ratio = 3
    rgb_kernelSize = 3
    rgb_edge = cv2.Canny(rgb_th3, rgb_low_threshold, rgb_low_threshold*rgb_ratio, rgb_kernelSize) # 100, 150
    rgb_edge = cv2.GaussianBlur(rgb_edge, (5,5), 0)
    
    rgb_y, rgb_x = rgb.shape
    aspect_ration_rgb = rgb_x/rgb_y

    plot(ir_edge, rgb_edge, ir, rgb_th3)

    scale_w = 1
    # scale_w = 3
    size_w = scale_w_glob = max_val_glob = max_loc_glob = 0

    # Until both frames have the same width
    #scale_w < (rgb_y/ir_y)
    while(size_w < rgb.shape[0]): # rgb.shape[1]

        # 0 if ir is tempelate image (Smaller)
        # 1 if rgb is tempelate image (Smaller)
        if(mode==1):
            temp = rgb.copy()
            temp_y, temp_x, _ = temp.shape
            shrink_percentage_w = (temp_x-(temp_x/scale_w))/temp_x

            size_w = (1-shrink_percentage_w) * temp_x
            size_h = (1-shrink_percentage_w) * temp_y
            temp_resized = cv2.resize(temp, (int(size_w), int(size_h)))

            res = cv2.matchTemplate(ir, temp_resized, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_val_glob:
                max_val_glob = max_val
                max_loc_glob = max_loc
                scale_w_glob = scale_w
            scale_w += 0.001

        elif(mode==0):
            temp_edge = ir_edge.copy()
            temp_y, temp_x = temp_edge.shape
            size_w = temp_x * scale_w
            size_h = size_w / aspect_ratio_ir

            # temp_edge = cv2.Canny(cv2.GaussianBlur(temp, (3,3), 0), 75, 170)
            temp_edge_resized = cv2.resize(temp_edge, (int(size_w), int(size_h)))
            print(temp_edge_resized.shape)

            res = cv2.matchTemplate(rgb_edge, temp_edge_resized, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > max_val_glob:
                max_val_glob = max_val
                max_loc_glob = max_loc
                scale_w_glob = scale_w
            scale_w += 0.001

    print(f"scale: {scale_w_glob}")
    print(f"Offset: {max_loc_glob}")
    print(f"Val: {max_val_glob}")

    # pt1 = (int(max_loc_glob[0]), (max_loc_glob[1]))
    # pt2 = (int(max_loc_glob[0]+scale_w_glob*temp_x), int(max_loc_glob[1]+scale_w_glob*temp_y))
    # cv2.rectangle(rgb, pt1, pt2, (0, 255, 0), 2)
    # plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),)
    # plt.show()
    return max_val_glob, max_loc_glob, scale_w_glob


if __name__ == '__main__':

    path_train_set = '/home/ub145/Documents/Dataset/FLIR/FLIR/train/'
    path_val_set = '/home/ub145/Documents/Dataset/FLIR/FLIR/val/'
    imageNumber = '01656'

    bb_gtruth = get_cords(imageNumber)

    _, max_loc_glob, scale_w_glob = calc_para(path_train_set, imageNumber)
    rgb = cv2.imread(path_train_set + 'RGB/FLIR_' + imageNumber + ".jpg")

    # if rgb.shape == (1600, 1800, 3):    
    #     #scale
    #     scale_w_glob = 2.5 #2.476
    #     #offset
    #     max_loc_glob = (148, 155) #(157, 145)
    # elif rgb.shape == (1536, 2048, 3):
    #     pass
    # elif rgb.shape == (1024, 1280, 3):
    #     pass
    # elif rgb.shape == (480, 720, 3):
    #     pass

    #ir frame size
    temp_x = 640
    temp_y = 512

    pt1 = (int(max_loc_glob[0]), (max_loc_glob[1]))
    pt2 = (int(max_loc_glob[0]+scale_w_glob*temp_x), int(max_loc_glob[1]+scale_w_glob*temp_y))

    for bb in bb_gtruth:
        cv2.rectangle(rgb, pt1, pt2, (0, 255, 0), 2)

    # Fix the scale
    for bb in bb_gtruth:
        bb = [int(x * scale_w_glob) for x in bb]
        bb[0] += max_loc_glob[0]
        bb[1] += max_loc_glob[1]

        cv2.rectangle(rgb, (bb[0], bb[1]), (bb[0]+bb[2], bb[1]+bb[3]), (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    plt.show()