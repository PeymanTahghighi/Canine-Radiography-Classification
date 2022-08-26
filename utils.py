
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

def save_test_data(fold_cnt, models, test_data):
    for radiograph_image_path in test_data:
        file_name = os.path.basename(radiograph_image_path);
        file_name = file_name[:file_name.rfind('.')];

        file_name = os.path.basename(radiograph_image_path);
        file_name = file_name[:file_name.rfind('.')];

        radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);


        transformed = config.valid_transforms(image = radiograph_image);
        radiograph_image = transformed["image"];
        radiograph_image = radiograph_image.to(config.DEVICE);
        
        #spine and ribs
        out = models[0](radiograph_image.unsqueeze(dim=0));
        out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
        out = np.argmax(out,axis = 2);

        ribs = (out == 1).astype("uint8")*255;
        spine = (out == 2).astype("uint8")*255;

        ribs = remove_blobs(ribs);
        spine = remove_blobs_spine(spine);
        #----------------------------------------------------

        #diaphragm
        diaphragm = models[1](radiograph_image.unsqueeze(dim=0));
        diaphragm = torch.sigmoid(diaphragm)[0].permute(1,2,0).detach().cpu().numpy();
        diaphragm = diaphragm > 0.5;
        diaphragm = np.uint8(diaphragm)*255;
        #----------------------------------------------------

        #sternum
        sternum = models[2](radiograph_image.unsqueeze(dim=0));
        sternum = torch.sigmoid(sternum)[0].permute(1,2,0).detach().cpu().numpy();
        sternum = sternum > 0.5;
        sternum = np.uint8(sternum);
        #----------------------------------------------------

        #Symmetry
        sym_line = get_symmetry_line(spine);
        ribs_left, ribs_right = divide_image_symmetry_line(ribs, sym_line);
        thorax_left = segment_thorax(ribs_left);
        thorax_right = segment_thorax(ribs_right);
        whole_thorax = segment_thorax(ribs);
        #symmetry features
        #----------------------------------------------------

        #Cranial
        cranial = spine - whole_thorax;
        cranial_features = extract_cranial_features(cranial);
        #-----------------------------------------------------

        #Caudal
        caudal = diaphragm - whole_thorax;
        caudal_features = extract_cranial_features(caudal);
        #-----------------------------------------------------

        #sternum
        sternum = np.logical_and(sternum.squeeze(), whole_thorax).astype(np.uint8);
        sternum_features = np.sum(sternum, (1,2));
        #-----------------------------------------------------

        pickle.dump([cranial_features, caudal_features, sternum_features], f'results\\{fold_cnt}\\test\\{file_name}.feat');

