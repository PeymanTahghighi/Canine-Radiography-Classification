from PyQt5 import QtCore
from cv2 import IMREAD_GRAYSCALE
from pandas.io import pickle
from Utility import get_radiograph_label_meta
from PIL import ImageColor, Image
import PIL
from imgaug.augmenters.meta import Sometimes
import numpy as np
import os
from numpy.lib.function_base import copy
from numpy.lib.type_check import imag
from scipy.sparse.construct import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
from sklearn.utils import shuffle
import torch
from glob import glob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import Config
import logging

class DatasetLoader():
    def __init__(self, root_path) -> None:
        self.__root_path = root_path;
        self.__exposure_label_dict = dict();
        self.__rotation_label_dict = dict();
        self.__exposure_label_counter = dict();
        self.__rotation_label_counter = dict();
        self.__exposure_cur_label_idx = 0;
        self.__rotation_cur_label_idx = 0;
        pass

    def load(self):
        """
            In this function we search for images and labels that match
            together in "images" and "labels" folder respectively.
        """

        images_root = os.path.sep.join([self.__root_path, "images"]);
        labels_root = os.path.sep.join([self.__root_path, "labels"]);

        images_root_list = glob(images_root+"\\*");
        labels_root_list = glob(labels_root + "\\*.meta");

        images_list = [];
        labels_list = [];

        for img_path in images_root_list:
            img_name, ext = os.path.splitext(os.path.basename(img_path));
            dummy_label_path = os.path.sep.join([self.__root_path, "labels", img_name + ".meta"]);
            if dummy_label_path in labels_root_list:
                img_0 = cv2.imread(os.path.sep.join([self.__root_path, "labels", img_name + "_0.png"]), cv2.IMREAD_COLOR);
                img_1 = cv2.imread(os.path.sep.join([self.__root_path, "labels", img_name + "_1.png"]), cv2.IMREAD_COLOR);
                combined = img_0+ img_1;
                images_list.append(combined);
                df = pickle.load(open(dummy_label_path,'rb'));
                #print(df['rot']);
                #print(img_path);
                #cv2.imshow("t", combined);
                #cv2.waitKey();
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
                hist = cv2.calcHist(img, [1], None, [256], [0,255] );
                labels_list.append(self.__convert_to_class_label(df['exp'], df['rot']));
            
        images_list, labels_list = shuffle(images_list, labels_list, random_state=40);

        exp_weights, rot_weights = self.__get_label_weights();
        return images_list, labels_list, rot_weights;
    
    def __convert_to_class_label(self, exp, rot):
        rot_label_idx = 0;
        exp_label_idx = 0;
        if rot in self.__rotation_label_dict.keys():
            rot_label_idx = self.__rotation_label_dict[rot];
            self.__rotation_label_counter[rot] += 1;
        else:
            self.__rotation_label_dict[rot] = self.__rotation_cur_label_idx;
            rot_label_idx = self.__rotation_cur_label_idx;
            self.__rotation_cur_label_idx += 1;
            self.__rotation_label_counter[rot] = 1;
        

        if exp in self.__exposure_label_dict.keys():
            exp_label_idx = self.__exposure_label_dict[exp];
            self.__exposure_label_counter[exp] += 1;
        else:
            self.__exposure_label_dict[exp] = self.__exposure_cur_label_idx;
            exp_label_idx = self.__exposure_cur_label_idx;
            self.__exposure_cur_label_idx += 1;
            self.__exposure_label_counter[exp] = 1;
        
        return [exp_label_idx, rot_label_idx];
    
    def __get_label_weights(self):
        total_rot = 0;
        total_exp = 0;
        for k in self.__exposure_label_counter.keys():
            total_exp += self.__exposure_label_counter[k];
        
        for k in self.__rotation_label_counter.keys():
            total_rot += self.__rotation_label_counter[k];
        
        ret_exp = [];
        for k1 in self.__exposure_label_counter.keys():
            total_other = 0;
            for k2 in self.__exposure_label_counter.keys():
                if k1 != k2:
                    total_other+=self.__exposure_label_counter[k2];
            ret_exp.append(total_other/total_exp);
        
        ret_rot = [];
        for k1 in self.__rotation_label_counter.keys():
            total_other = 0;
            for k2 in self.__rotation_label_counter.keys():
                if k1 != k2:
                    total_other+=self.__rotation_label_counter[k2];
            ret_rot.append(total_other/total_rot);
        
        ret_rot = np.array(ret_rot, dtype=np.float);
        ret_rot = ret_rot / np.sqrt(np.sum(ret_rot ** 2));
        
        return ret_exp, ret_rot;
        
    
    def display_histograms(self, root_path, label):
        self.__root_path = root_path;
        img_list, lbl_list = self.load();

        cnt = 1;
        for img, lbl in zip(img_list, lbl_list):
            if lbl[0] == label:
                img = cv2.imread(img, cv2.IMREAD_GRAYSCALE);
                plt.figure();   
                plt.subplot(1,2,1);
                plt.hist(img.ravel(), 256, [0,256]);
                plt.subplot(1,2,2);
                plt.imshow(img, cmap='gray');
                plt.title(lbl);
                cnt+=2;
        plt.show();


class NetworkDataset(Dataset):
    def __init__(self, radiographs, masks, transform, layer_names = None, train = True):
        self.train = train;
        
        self.radiographs = radiographs;
        self.masks = masks;

        self.transform = transform;
            
    def __len__(self):
        logging.info(f"Data size: {len(self.radiographs)}");
        # if self.train:
        #     return 100;
        return len(self.radiographs);

    def __getitem__(self, index):
        logging.info("start of get item");
        radiograph_image = self.radiographs[index];
        img_class = self.masks[index];
        #radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_COLOR);

        #radiograph_image = np.expand_dims(radiograph_image, axis=2);
        #radiograph_image = np.repeat(radiograph_image, 3,axis=2);
       
        # for j in range(1,7):
        #     for i in range(3,11,2):
        #         clahe = cv2.createCLAHE(j,(i,i));
        #         radiograph_image_q = clahe.apply(radiograph_image);
        #         cv2.imwrite(f"fhist equal_{i}{j}_5.png", radiograph_image_q);
        # cv2.imwrite("normal.png", radiograph_image);
        #sns.distplot(radiograph_image.ravel(), label=f'Mean : {np.mean(radiograph_image)}, std: {np.std(radiograph_image)}');
        #plt.legend(loc='best');
        #plt.savefig('dist-before.png');

        #radiograph_image  = np.array(F.equalize(Image.fromarray(radiograph_image)));
        #cv2.imwrite('normal.png',radiograph_image);
       # cv2.imwrite('equalize.png',temp);
        #plt.imshow(mask_image, cmap='gray');
        # plt.waitforbuttonpress();
        #cv2.imshow("rad", radiograph_image);
        #cv2.imshow("mask", mask_image);
        #cv2.waitKey();
        # cv2.imshow('mask', mask_image);
        # cv2.waitKey();
        # mask_image[mask_image == 255] = 1;

        if self.transform is not None:
            transformed = self.transform(image = radiograph_image);
            radiograph_image = transformed["image"];
            radiograph_image = radiograph_image /255.0;
            #ri = radiograph_image.permute(1,2,0).cpu().detach().numpy()*255;
            #sns.distplot(ri.ravel(), label=f'Mean : {np.mean(ri)}, std: {np.std(ri)}');
            #plt.legend(loc='best');
            #plt.savefig('dist-after.png');
            #mask_image = transformed["mask"]/255;
            #m = mask_image.detach().cpu().numpy();
            # cv2.imshow('m',m*255);
            # cv2.waitKey();

        return radiograph_image, img_class, index;

def analyze_dataset():
    radiograph_root = os.path.sep.join(["dataset","CXR_png"]);
    mask_root = os.path.sep.join(["dataset","masks"]);

    radiograph_list = os.listdir(radiograph_root);
    mask_list = os.listdir(mask_root);

    mask_images_names = [];
    radiograph_images_names = [];

    negative = 0;
    positive = 0;
    for m in radiograph_list:
        b = m.find('MCU');
        mask_name = m[0:m.find('.')] + "_mask.png" if b else m;
        if mask_name in mask_list:
            mask_images_names.append(mask_name);
            radiograph_images_names.append(m);
            sample_img = cv2.imread(os.path.sep.join([mask_root,mask_name]),cv2.IMREAD_GRAYSCALE);
            w,h = sample_img.shape;
            sample_img = sample_img.flatten();
            sample_img = (sample_img == 255);
            p = np.sum(sample_img);
            n = np.sum((sample_img == 0))
            positive += p / (704 * w*h);
            negative += n/ (704 * w*h);
    
    print(f"Positive portion:{positive}\tNegative portion:{negative}");
    negative_bias = positive;


    pass

if __name__ == "__main__":
    dl = DatasetLoader.display_histograms("C:\\PhD\\Miscellaneous\\vet2",0);