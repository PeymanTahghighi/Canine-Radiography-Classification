#===========================================================
#===========================================================
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import albumentations as A
import pickle

from tqdm import tqdm
from thorax import segment_thorax

from utility import divide_image_symmetry_line, get_symmetry_line

#===========================================================
#===========================================================

ROOT = 'C:\\PhD\\Miscellaneous\\Spine and Ribs\\labels'
names = ['602 (1)', '620', '606', '636', '659 (1)', '669', '660', '744', 'DV12', '317']
def preload_data():
    all_masks = glob(f'{ROOT}\\*.meta');
    hemithoraces_dict = dict();
    for t in tqdm(all_masks):
        file_name = os.path.basename(t);
        file_name = file_name[:file_name.rfind('.')];
        meta_data = pickle.load(open(os.path.join(ROOT, f'{t}'), 'rb'));

        if 'Spine' in meta_data.keys() and 'Ribs' in meta_data.keys() and file_name in names:
            spine_mask = cv2.imread(os.path.join(ROOT,  meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
            spine_mask = np.where(spine_mask > 0, 255, 0).astype(np.uint8);
            ribs_mask = cv2.imread(os.path.join(ROOT, meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
            ribs_mask = np.where(ribs_mask > 0, 255, 0).astype(np.uint8);

            sym_line = get_symmetry_line(spine_mask);
            ribs_left, ribs_right = divide_image_symmetry_line(ribs_mask, sym_line);
            thorax_left = segment_thorax(ribs_left);
            thorax_right = segment_thorax(ribs_right);

            hemithoraces_dict[file_name] = [thorax_left, thorax_right];

    return hemithoraces_dict;


def preprocess_train_dataset(hemithoraces_dict):
    #process folds data
    image_list, mask_list, lbl_list = pickle.load(open('all_data.dmp', 'rb'));
    for i in range(5):
        train_idxs = pickle.load(open(f'{i}.dmp', 'rb'))[0];
        test_idxs = pickle.load(open(f'{i}.dmp', 'rb'))[1];

        train_imgs = image_list[train_idxs];
        for t in train_imgs:
            file_name = os.path.basename(t);
            file_name = file_name[:file_name.rfind('.')];
            meta_data = pickle.load(open(os.path.join(ROOT, f'{file_name}.meta'), 'rb'));

            if 'Spine' in meta_data.keys() and 'Ribs' in meta_data.keys() and file_name in names:

                if os.path.exists(f'{i}\\train') is False:
                    os.mkdir(f'{i}\\train');

                cv2.imwrite(f'{i}\\train\\{file_name}_left.png', hemithoraces_dict[file_name][0]);
                cv2.imwrite(f'{i}\\train\\{file_name}_right.png', hemithoraces_dict[file_name][1]);

if __name__ == "__main__":
    # hemithoraces = preload_data();
    # preprocess_train_dataset(hemithoraces);    
    
    for i in range(5):
        test_list = glob(f'{i}\\test\\*.');
        train_list = glob(f'{i}\\test\\*.');



