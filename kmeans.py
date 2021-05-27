from os import path
from Fusion.utils.general import *
from FLIR_PP.arg_parser import DATASET_PP_PATH
from pathlib import Path


if __name__ == '__main__':

    anchor_num = 9
    # path = Path(DATASET_PP_PATH / Path('/Train_Test_Split/train/'))
    _ = kmean_anchors(path='Fusion/FLIR.yaml', n=anchor_num, img_size=640, thr=4, gen=1000, verbose=True)