
from cProfile import label
import os
import cv2
from glob import glob
import numpy as np
import pickle
import pandas as pd

def scale_spine(img_path, scale_factor = 2):
    file_name = os.path.basename(img_path);
    file_name = file_name[:file_name.rfind('.')];
    spine = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
    spine = np.where(spine > 0, 1, 0);
    h,w = spine.shape;
    out = np.zeros_like(spine);
    for i in range(h):
        if np.sum(spine[i,:]) > 0:
            r = spine[i,:];
            r = np.where(r == 1);
            s = r[0][0];
            e = r[0][-1];
            if s != e:
                w = int((e - s) / 2);
                out[i, s-w:e+w] = 255;
            else:
                out[i,s] = 255;
    

    cv2.imwrite(f'D:\\PhD\\Thesis\\Segmentation Results\\scaled_spine\\{file_name}.png', out.astype('uint8'));


def chech_sternum_middle(file_name):
    if os.path.exists(f'D:\\PhD\Miscellaneous\\Sternum\\labels\\{file_name}.meta') is True \
    and os.path.exists(f'D:\\PhD\\Thesis\\Segmentation Results\\scaled_spine\\{file_name}_s.png') is True:
        meta_file = pickle.load(open(f'D:\\PhD\Miscellaneous\\Sternum\\labels\\{file_name}.meta', 'rb'));
        sternum_file_name = meta_file['Sternum'][2];
        sternum_file = cv2.imread(f'D:\\PhD\Miscellaneous\\Sternum\\labels\\{sternum_file_name}', cv2.IMREAD_GRAYSCALE);
        spine_file = cv2.imread(f'D:\\PhD\\Thesis\\Segmentation Results\\scaled_spine\\{file_name}_s.png', cv2.IMREAD_GRAYSCALE);

        sternum_file = cv2.resize(sternum_file, (1024,1024));
        sternum_file = np.where(sternum_file > 0, 1, 0);
        spine_file = np.where(spine_file > 0, 1, 0);
        res = np.maximum(sternum_file - spine_file, np.zeros_like(spine_file));
        res = np.uint8(res);
        kernel = np.ones((5,5), np.uint8);
        res = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel);
        sternum_file_area_before = np.sum(sternum_file);
        sternum_file_area_after = np.sum(res);
        ratio = sternum_file_area_after / sternum_file_area_before;
        # cv2.imshow('sternum', sternum_file.astype('uint8')*255);
        # cv2.imshow('spine', spine_file.astype('uint8')*255);
        # cv2.imshow('res', res.astype('uint8')*255);
        # cv2.waitKey();
    
        return 0 if ratio < 0.1 else 1, ratio,res;
    return -1,-1,-1;



if __name__ == "__main__":
    lst = glob('D:\\PhD\\Thesis\\Segmentation Results\\spine\\*');
    for s in lst:
        scale_spine(s);
    labels_file = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');
    img_lst = list(labels_file['Image']);
    img_lst = list(map(str, img_lst));
    sternum_lbl = list(labels_file['Sternum']);

    for i in range(len(img_lst)):
        s,r,res = chech_sternum_middle(img_lst[i]);
        if s!= -1:
            lbl = sternum_lbl[i];

            if s == 0:
                if lbl == 'Yes':
                    print(f'{img_lst[i]} should be mid:{r}');
                    cv2.imshow(img_lst[i], res.astype('uint8')*255);
                    cv2.waitKey();
            else:
                if lbl == 'Mid':
                    print(f'{img_lst[i]} should not be mid:{r}');
                    cv2.imshow(img_lst[i], res.astype('uint8')*255);
                    cv2.waitKey();
