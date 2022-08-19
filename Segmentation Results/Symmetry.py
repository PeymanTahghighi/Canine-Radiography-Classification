from dis import dis
from operator import ne
from contextlib2 import closing
import cv2
from cv2 import sort
import numpy as np
from glob import glob
import os

import torch
from tqdm import tqdm
from Thorax import segment_thorax
import pickle
from Network import Unet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

transforms = A.Compose(
[
    A.Resize(1024, 1024),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
]
)

def get_symmetry_line(img):
    assert img.ndim == 2, "Image should be grayscale"
    
    w,h = img.shape;

    symmetry_line = np.zeros((w,2), dtype = np.int32);

    for i in range(w):
        first_cord = None;
        second_cord = None;
        for j in range(h):
            if img[i][j] != 0 and first_cord is None:
                first_cord = j;
            elif img[i][j] != 0:
                second_cord = j;

        if second_cord != None and first_cord != None:    
            symmetry_line[i] = ((i,(second_cord + first_cord) / 2));
    
    #check for missed points with zero values,
    #we use the average of ten points after and ten points before
    #as their value
    look_ahead = 10;
    for idx, s in enumerate(symmetry_line):
        if s[1] == 0:
            start_idx = idx;
            #attemp to estimte this value
            sum = 0;
            cnt_pos = 0;
            while(cnt_pos != look_ahead):
                if(symmetry_line[start_idx][1] != 0):
                    sum += symmetry_line[start_idx][1];
                    cnt_pos += 1;
                start_idx += 1;

                if start_idx >= w:
                    break;
            
            start_idx = idx;
            cnt_neg = 0;
            while(cnt_neg != look_ahead):
                if(symmetry_line[start_idx][1] != 0):
                    sum += symmetry_line[start_idx][1];
                    cnt_neg += 1;
                start_idx -= 1;

                if start_idx < 0:
                    break;
            sum /= cnt_neg + cnt_pos;

            symmetry_line[idx] = (idx, sum);
    
    return symmetry_line;

def divide_image_symmetry_line(img, sym_line):
    img_left = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.uint8);
    img_right = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.uint8);

    w,h = img.shape;

    for s in sym_line:
        for j in range(h):
            if j < s[1]:
                img_left[s[0], j] = img[s[0], j];
            else:
                img_right[s[0], j] = img[s[0], j];
    
    return img_left, img_right;

def remove_outliers(lst):
    ret_lst = [];
    q1 = np.quantile(lst, axis=0, q=0.25);
    q3 = np.quantile(lst, axis=0, q=0.75);
    iqr = q3-q1;
    dist_list = [];
    total_dist = 0;
    for idx, p in enumerate(lst):
        x_range_start = q1[0] - 1.5*iqr[0];
        x_range_end = q3[0] + 1.5*iqr[0];

        y_range_start = q1[1] - 1.5*iqr[1];
        y_range_end = q3[1] + 1.5*iqr[1];
        if p[0]<x_range_end and p[0]>x_range_start and p[1] < y_range_end and p[1] > y_range_start:
            ret_lst.append(idx);
    
    return ret_lst;

def remove_outliers_spine(lst):
    ret_lst = [];
    q1 = np.quantile(lst, axis=0, q=0.25);
    q3 = np.quantile(lst, axis=0, q=0.75);
    iqr = q3-q1;
    dist_list = [];
    total_dist = 0;
    for idx, p in enumerate(lst):
        x_range_start = q1[0] - 1.5*iqr[0];
        x_range_end = q3[0] + 1.5*iqr[0];

        y_range_start = q1[1] - 1.5*iqr[1];
        y_range_end = q3[1] + 1.5*iqr[1];
        if p[0]<x_range_end and p[0]>x_range_start and p[1] < y_range_end and p[1] > y_range_start:
            ret_lst.append(idx);

    return ret_lst;

def remove_blobs(ribs):
    kernel = np.ones((5,5), dtype=np.uint8);
    kernel_c = np.ones((35,35), dtype=np.uint8);
    opening = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, kernel);
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_c);
    # cv2.imshow('open', opening);
    # cv2.imshow('close', closing);
    # cv2.waitKey();
    ret_img = np.zeros_like(closing);

    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    mean_area = 0;
    for c in contours:
        mean_area += cv2.contourArea(c);
    
    mean_area /= len(contours);

    position_list = [];
    all_position = [];
    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        dia = cv2.arcLength(c, True);
        #list.append([area, dia]);
        x,y,w,h = cv2.boundingRect(c);
        center = [x+w/2,y+h/2];
        all_position.append(center);
        all_area.append([area, dia]);
    
    max_area = np.mean(all_area);
    positions = remove_outliers(all_position);
    
    q1 = np.quantile(all_area, 0.1, axis = 0);
    
    for idx, p in enumerate(contours):
        if all_area[idx][0] > max_area * 0.1:
            ret_img = cv2.fillPoly(ret_img, [contours[idx]], (255,255,255));
            
    return ret_img;


def remove_blobs_spine(ribs):
    kernel = np.ones((5,5), dtype=np.uint8);
    kernel_c = np.ones((41,41), dtype=np.uint8);
    opening = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, kernel);
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_c);
    # cv2.imshow('open', opening);
    # cv2.imshow('close', closing);
    # cv2.waitKey();
    ret_simplified = np.zeros_like(closing);
    ret_original = np.zeros_like(closing);
    contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        all_area.append(area);

    biggest = np.max(all_area);
    for idx, a in enumerate(all_area):
        if a > 0.2*biggest:
            #simplify spine
            #cvh = cv2.approxPolyDP(contours[idx], 10,True);
            #ret_simplified = cv2.fillPoly(ret_simplified, [cvh], (255,255,255));
            ret_original = cv2.drawContours(ret_original, [contours[idx]],-1, (255,255,255),-1);

    return ret_simplified, ret_original;

if __name__ == "__main__":

    ROOT = "C:\\PhD\\Thesis\\Dataset\\DVVD";
    img_list = os.listdir(ROOT);
    network = Unet(3);
    ckpt = torch.load('ckpt.pt', map_location='cuda');
    network.load_state_dict(ckpt['state_dict']);
    network = network.cuda();

    for idx in tqdm(range(len(img_list))):
        img_name = img_list[idx];
        radiograph_image = cv2.imread(os.path.join(ROOT, img_name),cv2.IMREAD_GRAYSCALE);
        img_name = img_name[:img_name.rfind('.')];
        #if os.path.exists(os.path.sep.join(["thorax", img_name + "_thorax.png"])) is False:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);

        transformed = transforms(image = radiograph_image);
        radiograph_image = transformed['image'].cuda();
        out,_ = network(radiograph_image.unsqueeze(dim = 0));
        out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
        out = np.argmax(out,axis = 2);
        ribs = (out == 1).astype("uint8")*255;
        spine = (out == 2).astype("uint8")*255;
        ribs_proc = remove_blobs(ribs);
        spine_proc_s, spine_proc_o = remove_blobs_spine(spine);

        #simplify spine


        #cv2.imwrite(f'tmp\\ribs_{img_name}.png', ribs);
        # cv2.imshow('spine_proc', spine_proc);
    

        
        # sym_line = get_symmetry_line(spine_proc);
        # ribs_left, ribs_right = divide_image_symmetry_line(ribs_proc, sym_line);
        # thorax_left = segment_thorax(ribs_left);
        # thorax_right = segment_thorax(ribs_right);
        # total_thorax = segment_thorax(ribs_proc);

        #thorax = cv2.imread(ribs_img_path, cv2.IMREAD_GRAYSCALE);
        # cv2.imwrite(os.path.sep.join(["thorax", img_name + ".png"]), thorax);
        # #cv2.imshow("thorax", thorax);
        # #cv2.waitKey();

        # #spine_img = cv2.imread(spine_img_path, cv2.IMREAD_GRAYSCALE);
        # thorax = cv2.resize(thorax, (spine.shape[1], spine.shape[0]))

        #cv2.imwrite(os.path.sep.join(["divided", img_name + "left.png"]), thorax_left);
        #cv2.imwrite(os.path.sep.join(["divided", img_name + "right.png"]), thorax_right);
        #cv2.imwrite(os.path.sep.join(["ribs", img_name + "_r.png"]), ribs_proc);
        cv2.imwrite(os.path.sep.join(["spine", img_name + "_s.png"]), spine_proc_o);
        #cv2.imwrite(os.path.sep.join(["thorax", img_name + "_thorax.png"]), total_thorax);


        
