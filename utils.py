
import torch
import config
import os
import shutil
import numpy as np
import cv2
from utility import divide_image_symmetry_line, get_symmetry_line, remove_blobs, remove_blobs_spine
from thorax import segment_thorax
import pickle

def extract_cranial_features(cranial_image):
    w,h = cranial_image.shape;
    contours, _ = cv2.findContours(cranial_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    total_area = 0;
    total_diameter = 0;
    for c in contours:
        total_area += cv2.contourArea(c);
        total_diameter += cv2.arcLength(c, True);

    total_area /= w*h;
    total_diameter /= ((w+h)*2);
    return [total_area, total_diameter];

def create_folder(folder_name):
    if os.path.exists(f'results\\{folder_name}') is True:
        shutil.rmtree(f'results\\{folder_name}');
    os.mkdir(f'results\\{folder_name}');


