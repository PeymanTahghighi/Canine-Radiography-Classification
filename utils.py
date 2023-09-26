
from cgitb import small
from copy import deepcopy
import torch
import config
import os
import shutil
import numpy as np
import cv2
from Symmetry.thorax import segment_thorax
import pickle

import matplotlib.pyplot as plt

def get_corner(mask, rev = False):
    w,h = mask.shape;
    contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c;
    max_cnt = np.squeeze(max_cnt);
    if rev is True:
        max_cnt[:,0]= h - max_cnt[:,0];
        # tmp = np.zeros_like(mask);
        # tmp = cv2.drawContours(tmp, [max_cnt], 0, (255,255,255), -1);
        # cv2.imshow('orig', mask);
        # cv2.imshow('rev', tmp);
        # cv2.waitKey();
        
        
    max_cnt = cv2.approxPolyDP(max_cnt, cv2.arcLength(max_cnt, True)*0.01, True);
    mask_ret = np.zeros_like(mask);
    mask_ret = cv2.drawContours(mask_ret, [max_cnt], 0, (255,255,255), -1);
    pints = max_cnt.squeeze();
    top = pints[np.argmin(pints[:,1])];
    s = np.sum(pints, axis = 1);
    crn_top_left = pints[np.argmin(s)];
    a = np.abs(pints[:,0]-pints[:,1]);
    crn_top_right = pints[np.argmin((pints[:,1]-pints[:,0]))];

    diff = np.sqrt((top[0]-crn_top_right[0])**2 + (top[1]-crn_top_right[1])**2);
    mask_ret = cv2.cvtColor(mask_ret, cv2.COLOR_GRAY2RGB);
    
    if diff < 0.1:
        if rev is False:
            mask_ret = cv2.circle(mask_ret, (crn_top_left[0], crn_top_left[1]), 10, (255,0,0), -1);
        ret_point = crn_top_left;
        #thorax_ret = cv2.circle(thorax_ret, (crn_top_right[0], crn_top_right[1]), 5, (255,0,255), -1);
        #thorax_ret = cv2.circle(thorax_ret, (top[0], top[1]), 2, (0,0,255), -1);
    else:
        #thorax_ret = cv2.cvtColor(thorax_ret, cv2.COLOR_GRAY2RGB);
        #thorax_ret = cv2.circle(thorax_ret, (crn_top_left[0], crn_top_left[1]), 10, (255,0,0), -1);
        # thorax_ret = cv2.circle(thorax_ret, (crn_top_right[0], crn_top_right[1]), 5, (255,0,255), -1);
        if rev is False:
            mask_ret = cv2.circle(mask_ret, (top[0], top[1]), 2, (0,0,255), -1);
        ret_point = top;
    
    if rev is True:
        ret_point[0] = h - ret_point[0];
        mask_ret = cv2.circle(mask_ret, (ret_point[0], ret_point[1]), 5, (255,0,0,), -1);
    return ret_point, mask_ret;

def top_left(img):
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c;
    

    s = np.sum(max_cnt.squeeze(), axis = 1);
    idx = np.argmin(s);
    return max_cnt[idx].squeeze();

def top_right(img):
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c.squeeze();
    
    tmp = max_cnt[:,0] - max_cnt[:,1];
    idx = np.argmax(tmp);
    return max_cnt[idx].squeeze();




def create_folder(folder_name, delete_if_exists = True):
    if delete_if_exists is True:
        if os.path.exists(f'{folder_name}') is True:
            shutil.rmtree(f'{folder_name}');
    if os.path.exists(f'{folder_name}') is False:        
        os.makedirs(f'{folder_name}');

