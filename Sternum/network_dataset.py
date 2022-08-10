
from pandas.io import pickle
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
from sklearn.utils import shuffle
from glob import glob
import pickle
import matplotlib.pyplot as plt
import config

class SternumDataset(Dataset):
    def __init__(self, radiographs, masks, labels, transform, valid=False):
        self.__radiographs = radiographs;
        self.__masks = masks;
        self.__transform = transform;
        self.__labels = labels;
        self.__valid = valid;
        
            
    def __len__(self):
        return len(self.__radiographs);

    def __getitem__(self, index):
        radiograph_image_path = self.__radiographs[index];
        
        radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);

        mask =  cv2.imread(self.__masks[index], cv2.IMREAD_GRAYSCALE);
        mask = np.where(mask > 0, 1, 0).astype("uint8");

        gt_lbl = self.__labels[index];

        transformed = self.__transform(image = radiograph_image, mask = mask);
        radiograph_image = transformed["image"];
        mask = transformed["mask"];
        if self.__valid is False:
            return radiograph_image, mask, gt_lbl;
        file_name = os.path.basename(self.__radiographs[index]);
        file_name = file_name[:file_name.rfind('.')];

        return radiograph_image, mask, gt_lbl, file_name;