
import os
from turtle import right
import cv2
from glob import glob
import numpy as np
import pickle
import pandas as pd
from scipy.signal import savgol_filter

def smooth_boundaries(spine, dist):
    spine_thresh = np.where(spine[0,:]>0);
    h,w = spine.shape;
    left_bound = [];
    right_bound = [];
    start = -1;
    for i in range(h):
        if np.sum(spine[i,:]) > 0:
            if start == -1:
                start = i;
            spine_thresh = np.where(spine[i,:]>0);
            left_bound.append(spine_thresh[0][0]);
            right_bound.append(spine_thresh[0][-1]);

    local_minimas = [];
    local_maximas = [];
    for i in range(dist, len(left_bound)-dist):
        temp_arr = left_bound[i-dist:i+dist];
        m = np.min(temp_arr);
        if m == left_bound[i]:
            local_minimas.append([m,i+start]);
    
    for i in range(dist, len(right_bound)-dist):
        temp_arr = right_bound[i-dist:i+dist];
        m = np.max(temp_arr);
        if m == right_bound[i]:
            local_maximas.append([m,i+start]);
    
    ret = np.zeros_like(spine);
    for l in range(len(local_minimas)-1):
        spine = cv2.line(spine, (int(local_minimas[l][0]), int(local_minimas[l][1])), (int(local_minimas[l+1][0]), int(local_minimas[l+1][1])),(255,255,255),1);
    for l in range(len(local_maximas)-1):
        spine = cv2.line(spine, (int(local_maximas[l][0]), int(local_maximas[l][1])), (int(local_maximas[l+1][0]), int(local_maximas[l+1][1])),(255,255,255),1);
    
    spine = np.where(spine > 0, 1, 0);
    out = np.zeros_like(spine);
    for i in range(h):
        if np.sum(spine[i,:]) > 0:
            r = spine[i,:];
            r = np.where(r == 1);
            s = r[0][0];
            e = r[0][-1];
            if s != e:
                w = int((e - s) / 4);
                out[i, s:e] = 255;
            else:
                out[i,s] = 255;
    return out;

def scale_width(spine, multiplier):
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
                w = int((e - s) / multiplier);
                out[i, s-w:e+w] = 255;
            else:
                out[i,s] = 255;
    return out;


def scale_spine(img_path, scale_factor = 2):
    file_name = os.path.basename(img_path);
    file_name = file_name[:file_name.rfind('.')];
    file_name = file_name[:file_name.rfind('_')];
    spine = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
    spine = np.where(spine>0, 255, 0).astype("uint8")
    
    out = smooth_boundaries(spine,10);
    out = smooth_boundaries(out,25);
    #out = smooth_boundaries(out,50);
    out = scale_width(out,3);
    # cv2.imshow('orig', spine.astype("uint8"));
    # cv2.imshow('out', out.astype("uint8"));
    # # #cv2.imshow('out1', out1.astype("uint8"));
    # cv2.waitKey();
    #b = cv2.addWeighted(spine, 0.5, ret, 0.5, 0.0);
    # cv2.imshow('dilate', spine);
    # cv2.waitKey();

    # kernel = np.ones((7,7), dtype=np.uint8);
    # ret = cv2.blur()
    # cv2.imshow('orig', spine);
    # cv2.imshow('dilate', ret);
    # cv2.waitKey();

    # contours = cv2.findContours(spine, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    # ret = np.zeros_like(spine);
    # for c in contours:
    #     peri = cv2.arcLength(c, True);
    #     cvh = cv2.approxPolyDP(c, peri*0.01,True);
    #     ret = cv2.drawContours(ret, [cvh], 0, (255,255,255),-1);
    
    # cv2.imshow('orig', spine);
    

    #spine = ret;
    #cv2.imshow('hull', spine);
    #cv2.waitKey();
    
    
    

    cv2.imwrite(f'D:\\PhD\\Thesis\\Unsupervised-Canine-Radiography-Classification\\Segmentation Results\\scaled_spine\\{file_name}.png', out.astype('uint8'));
    cv2.imwrite(f'D:\\PhD\\Thesis\\Unsupervised-Canine-Radiography-Classification\\Segmentation Results\\scaled_spine\\{file_name}_orig.png', spine.astype('uint8'));


def chech_sternum_middle(file_name):
    if os.path.exists(f'C:\\Users\\Admin\OneDrive - University of Guelph\Miscellaneous\\Sternum\\labels\\{file_name}.meta') is True \
    and os.path.exists(f'D:\\PhD\\Thesis\\Unsupervised-Canine-Radiography-Classification\Segmentation Results\\scaled_spine\\{file_name}.png') is True:
        meta_file = pickle.load(open(f'C:\\Users\\Admin\OneDrive - University of Guelph\Miscellaneous\\Sternum\\labels\\{file_name}.meta', 'rb'));
        sternum_file_name = meta_file['Sternum'][2];
        sternum_file = cv2.imread(f'C:\\Users\\Admin\OneDrive - University of Guelph\Miscellaneous\\Sternum\\labels\\{sternum_file_name}', cv2.IMREAD_GRAYSCALE);
        spine_file = cv2.imread(f'D:\\PhD\\Thesis\\Unsupervised-Canine-Radiography-Classification\Segmentation Results\\scaled_spine\\{file_name}.png', cv2.IMREAD_GRAYSCALE);
        spine_file = cv2.resize(spine_file, (1024,1024));

        sternum_file = cv2.resize(sternum_file, (1024,1024));

        blend = cv2.addWeighted(spine_file, 0.5, sternum_file, 0.5, 0.0);
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
    
        return 0 if ratio < 0.06 else 1, ratio,res, blend;
    return -1,-1,-1,-1;



if __name__ == "__main__":
    meta_files_paths = glob('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\*.meta');
    #paths = ['C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\792.meta']
    for m in meta_files_paths:
        meta_file = pickle.load(open(m,'rb'));
        if 'Ribs' in meta_file.keys() and 'Spine' in meta_file.keys():
            p = os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels', meta_file['Spine'][2]);
            scale_spine(p);
    labels_file = pd.read_excel('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    img_lst = list(labels_file['Image']);
    img_lst = list(map(str, img_lst));
    sternum_lbl = list(labels_file['Sternum']);

    for i in range(len(img_lst)):
        s,r,res, blend = chech_sternum_middle(img_lst[i]);
        if s!= -1:
            lbl = sternum_lbl[i];

            if s == 0:
                if lbl == 'Yes':
                    print(f'{img_lst[i]} should be mid:{r}');
                    # cv2.imshow(img_lst[i], res.astype('uint8')*255);
                    # cv2.imshow('blend', blend);
                    # cv2.waitKey();
            else:
                if lbl == 'Mid':
                    print(f'{img_lst[i]} should not be mid:{r}');
                    # cv2.imshow(img_lst[i], res.astype('uint8')*255);
                    # cv2.imshow('blend', blend);
                    # cv2.waitKey();
