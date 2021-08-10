import argparse
import sys
import os


dataset_path_parser = argparse.ArgumentParser(description='Just one argument is needed -> Path/to/FLIR/Dataset')
dataset_path_parser.add_argument('Path', metavar='path', type=str, help='Path to the FLIR Dataset')
args = dataset_path_parser.parse_args()
input_path = args.Path

if not os.path.isdir(input_path):
    print('Dataset Path does not exist')
    sys.exit()

else:
    print('The given path found in: ', input_path)
    # Main Dataset Path
    DATASET_PATH = input_path
    # A copied version for preprocessing
    DATASET_PP_PATH = os.path.join(input_path, 'FLIR_PP')
    # DATASET_PP_PATH = '/data/Sam/FLIR_PP_Plus'