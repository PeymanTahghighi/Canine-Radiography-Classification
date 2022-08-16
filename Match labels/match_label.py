from copy import copy
import pickle
from glob import glob
import os

from tqdm import tqdm

DEST_FOLDER = 'C:\\PhD\\Miscellaneous\\Spine and Ribs';
ORIG_FOLDER = 'C:\\PhD\\Thesis\\Dataset\\DVVD';
PROJECT_NAME = 'SpineandRibs';

def compare():
    img_list_orig_dir = glob(f'{ORIG_FOLDER}\\*');
    img_list_dest_dir = glob(f'{DEST_FOLDER}\\*');
    data_list_file = list(pickle.load(open(f'{DEST_FOLDER}\\{PROJECT_NAME}.uog', 'rb')).keys());

    img_list_orig_dir_ret = copy(img_list_orig_dir);
    data_list_file_ret = copy(data_list_file);

    for idx, img_path in tqdm(enumerate(img_list_orig_dir)):
        file_name_orig = os.path.basename(img_path);
        #file_name_in_dest = f'{DEST_FOLDER}\\{file_name_orig}';
        if file_name_orig in data_list_file:
            img_list_orig_dir_ret.remove(img_path);
            data_list_file_ret.remove(file_name_orig);
    
    print(f'Images left in orig path: \n{img_list_orig_dir_ret}');
    print(f'\n\nImages left in dest path: \n{data_list_file_ret}');

if __name__ == "__main__":
    compare();