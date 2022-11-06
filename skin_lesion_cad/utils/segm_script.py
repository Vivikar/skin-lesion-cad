import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from skin_lesion_cad.data import Segment

current_path = Path().resolve()
raw_data = current_path / 'data/raw'

# parse path to folder for processing
parser = argparse.ArgumentParser(description='Path to raw data to process from data/raw.')
parser.add_argument('path', metavar='p', type=str,
                    help='Path to raw data to process from data/raw. For example chall1/train/nevus')

parser.add_argument('--resize', metavar='r', type=float, default=1,
                    help='Resize factor for images. For example 0.5')
args = parser.parse_args()
process_path = args.path
resize = args.resize
img_paths = raw_data/process_path
img_paths = list(img_paths.glob('*'))

if __name__ == '__main__':

    segm = Segment()

    for fn in tqdm(img_paths):
        image = cv2.imread(str(fn))
        img_segm, inp_img = segm.segment(image, fn, save=True, resize=resize)
