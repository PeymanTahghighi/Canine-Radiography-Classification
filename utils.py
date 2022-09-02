
import torch
import config
import os
import shutil
import numpy as np
import cv2
from utility import divide_image_symmetry_line, get_symmetry_line, remove_blobs, remove_blobs_spine
from Symmetry.thorax import segment_thorax
import pickle
from pystackreg import StackReg

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

def create_folder(folder_name, delete_if_exists = True):
    if delete_if_exists is True:
        if os.path.exists(f'{folder_name}') is True:
            shutil.rmtree(f'{folder_name}');
    if os.path.exists(f'{folder_name}') is False:        
        os.makedirs(f'{folder_name}');

def crop_top(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    #assert len(contours) == 1, "Number of contours detected should be exactly one";
    for c in contours:
        if cv2.contourArea(c) > 10:
            ch = cv2.convexHull(c);
            #cv2.drawContours(img, [ch], -1, (255,255,255), 2);
            extLeft = tuple(ch[ch[:, :, 0].argmin()][0])
            extRight = tuple(ch[ch[:, :, 0].argmax()][0])
            extTop = tuple(ch[ch[:, :, 1].argmin()][0])
            extBot = tuple(ch[ch[:, :, 1].argmax()][0])
            total_height = np.abs(extTop[1] - extBot[1]);
            height_to_crop = int(total_height*0.25) + extTop[1]
            perimeter = cv2.arcLength(ch, True);
    
    return img[:height_to_crop,:];

def get_perimeter(img):
    img_crop = crop_top(img);
    contours_crop, _ = cv2.findContours(img_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    #assert len(contours) == 1, "Number of contours detected should be exactly one";
    #hull_list = [];
    for c in contours_crop:
        if cv2.contourArea(c) > 10:
            ch = cv2.convexHull(c);
            #hull_list.append(ch);
            
            perimeter_crop = cv2.arcLength(ch, True); 
    
    for c in contours:
        if cv2.contourArea(c) > 10:
            ch = cv2.convexHull(c);
            #hull_list.append(ch);
            
            perimeter = cv2.arcLength(ch, True); 
    
    # for idx in range(len(hull_list)):
    #     img = cv2.drawContours(img, hull_list, idx, (255,255,255), 2);

    

    # cv2.imshow('ch', img);
    # cv2.imshow('chcrop', img_crop);
    # cv2.waitKey();
    
    return perimeter, perimeter_crop;

def get_histogram(img, bins):
    temp_img = np.where(img == 255, 1, 0);
    h,w = img.shape;
    if h < bins:
        ph = bins;
        padded_img = np.zeros((ph,w));
        padded_img[:h,:] = img;
        img = padded_img;
        h = ph;

    rows_per_bin = int(h / bins);
    hist_horizontal = [];
    for i in range(0,h,rows_per_bin):
        s = temp_img[i:i+rows_per_bin,:];
        hist_horizontal.append(int(s.sum()));
    
    hist_horizontal = np.array(hist_horizontal, dtype=np.float32);
    hist_horizontal = np.expand_dims(hist_horizontal, axis=1);
    hist_horizontal = hist_horizontal / hist_horizontal.sum();

    hist_vertical = [];
    for i in range(0,w,rows_per_bin):
        s = temp_img[:,i:i+rows_per_bin];
        hist_vertical.append(int(s.sum()));
    
    hist_vertical = np.array(hist_vertical, dtype=np.float32);
    hist_vertical = np.expand_dims(hist_vertical, axis=1);
    hist_vertical = hist_vertical / hist_vertical.sum();
    
    return hist_horizontal, hist_vertical;

def IoU(img_1, img_2):
    h1,w = img_1.shape;
    h2,_ = img_2.shape;

    h = max(h1, h2);

    img_1_tmp = cv2.resize(img_1, (w,h));
    img_2_tmp = cv2.resize(img_2, (w,h));

    img_img_2_flipped = cv2.flip(img_2_tmp, 1);
    sr = StackReg(StackReg.TRANSLATION); 
    sr.register(img_1_tmp, img_img_2_flipped);
    out = sr.transform(img_img_2_flipped);
    out = np.array(out, dtype=np.uint8);
    #out = util.to_uint16(out);
    #out = np.where(out !=0, 1, 0);
    # cv2.imshow('o', out);
    # cv2.waitKey();
    intersection = cv2.bitwise_and(out, img_1_tmp);
    intersection = np.where(intersection == 255, 1, 0);
    intersection = np.sum(intersection);
    union = cv2.bitwise_or(out, img_1_tmp);
    union = np.where(union == 255, 1, 0).sum();
    xor = cv2.bitwise_xor(out, img_1_tmp)
    xor = (np.where(xor == 255, 1, 0).sum()) / (w*h);
    iou = intersection / union;
    return iou, out, xor;

def cross_entropy(p,q):
    return np.sum(-p*np.log(q));

def JSD(p,q):
    p = p + config.EPSILON;
    q = q + config.EPSILON;
    avg = (p+q)/2;
    jsd = (cross_entropy(p,avg) - cross_entropy(p,p))/2 + (cross_entropy(q,avg) - cross_entropy(q,q))/2;
    #clamp
    if jsd > 1.0:
        jsd = 1.0;
    elif jsd < 0.0:
        jsd = 0.0;
    
    return jsd;

def extract_symmetry_features(img_left, img_right):
    img_crop_left = crop_top(img_left);
    img_crop_right = crop_top(img_right);
    w,h = img_left.shape;
    area_left = (img_left == 255).sum() / (w*h);
    area_left_crop = (img_crop_left == 255).sum() / (img_crop_left.shape[0]*img_crop_left.shape[1]);
    peri_left, peri_left_crop = get_perimeter(img_left);

    area_right = (img_right == 255).sum() / (w*h);
    area_right_crop = (img_crop_right == 255).sum() / (img_crop_right.shape[0]*img_crop_right.shape[1]);
    peri_right, peri_right_crop = get_perimeter(img_right);

    f = area_left / area_right;
    f_crop = area_left_crop / area_right_crop;
    # else:
    #     f = area_right / area_left;
    
    s = 0;
    # if peri_left > peri_right:
    s = peri_left / peri_right;
    s_crop = peri_left_crop / peri_right_crop;
    #else:
    #s = peri_right / peri_left;

    max_h = max(img_crop_left.shape[0], img_crop_right.shape[0]);
    img_crop_right = cv2.resize(img_crop_right, (512, max_h));
    img_crop_left = cv2.resize(img_crop_left, (512, max_h));


    hist_left_hor, hist_left_ver = get_histogram(img_left, 256);
    hist_left_crop_hor, hist_left_crop_ver = get_histogram(img_crop_left, 256);
    hist_right_hor, _ = get_histogram(img_right, 256);
    hist_right_crop_hor, hist_right_crop_ver = get_histogram(img_crop_right, 256);
    iou,img_right_flipped, xor = IoU(img_left, img_right);
    _, hist_right_ver = get_histogram(img_right_flipped, 256);


    jsd1 = JSD(hist_left_hor, hist_right_hor);
    jsd2 = JSD(hist_left_ver, hist_right_ver);
    jsd3 = JSD(hist_left_crop_hor, hist_right_crop_hor);
    jsd4 = JSD(hist_left_crop_ver, hist_right_crop_ver);
    diff1 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_INTERSECT);
    diff2 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_BHATTACHARYYA);
    diff3 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_CHISQR);
    diff4 = cv2.compareHist(hist_left_hor, hist_right_hor, cv2.HISTCMP_CORREL);
    iou_crop_lr, out1, xor_crop_lr = IoU(img_crop_left, img_crop_right);
    iou_crop_rl, out2, xor_crop_rl = IoU(img_crop_right,img_crop_left);

    iou_lr, out1, xor_lr = IoU(img_left, img_right);
    iou_rl, out2, xor_rl = IoU(img_right,img_left);
    #print(f"{img_list[i]}: {jsd}")
    #total_data.append(np.concatenate([jsd1, jsd2, diff1, diff2, diff3, f, s], axis=0));

    feat = [];
    feat.append(f_crop);
    feat.append(diff2);
    feat.append(diff3);
    feat.append(iou_crop_rl);

    return feat;
