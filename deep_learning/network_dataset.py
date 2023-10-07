
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
from utility import draw_missing_spine, retarget_img, smooth_boundaries
from tqdm import tqdm

class CanineDataset(Dataset):
    def __init__(self, radiographs, masks, fold, train = True,  multilabel=False):
        self.__radiographs = radiographs;
        self.__masks = masks;
        self.__train = train;
        if multilabel:
            num_masks = masks.shape[1];
            temp_masks = [];
            for m in tqdm(masks):
                sr = pickle.load(open(m[0].replace('\\', '/'), 'rb'));
                d = pickle.load(open(m[1].replace('\\', '/'), 'rb'));
                h = pickle.load(open(m[2].replace('\\', '/'), 'rb'));

                d = cv2.resize(d.astype("uint8")*255, (h.shape[1], h.shape[0])) >0;

                mask = np.zeros((sr.shape[0], sr.shape[1], num_masks+1), dtype="uint8");
                mask[:,:,0] = np.where(sr == 1, 1,0).squeeze();
                mask[:,:,1] = np.where(sr == 2, 1,0).squeeze();
                mask[:,:,2] = d;
                mask[:,:,3] = h;

                if config.DEBUG_TRAIN_DATA:
                    fig,ax = plt.subplots(1,num_masks+1);
                    ax[0].imshow(mask[:,:,0]);
                    ax[1].imshow(mask[:,:,1]);
                    ax[2].imshow(mask[:,:,2]);
                    ax[3].imshow(mask[:,:,3]);
                    plt.show();

                temp_masks.append(mask);

        else:
            temp_masks = [];
            for m in tqdm(masks):
                mask = pickle.load(open(m.replace('\\', '/'), 'rb'));
                temp_masks.append(mask);
                
        temp_radiographs = [];
        c = 0;
        for rad in tqdm(radiographs):
            radiograph = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{rad}.jpeg'),cv2.IMREAD_GRAYSCALE);
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            radiograph = clahe.apply(radiograph);
            radiograph = np.expand_dims(radiograph, axis=2);
            radiograph = np.repeat(radiograph, 3,axis=2);
            temp_radiographs.append(radiograph);


        self.__masks = temp_masks;
        self.__radiographs = temp_radiographs;
            
    def __len__(self):
        return len(self.__radiographs);

    def __getitem__(self, index):
        
        radiograph = self.__radiographs[index]
        mask = self.__masks[index];
        
        
        
        if self.__train is True:
            transformed = config.train_transforms(image = radiograph, mask = mask);
        else:
            transformed = config.valid_transforms(image = radiograph, mask = mask);


        radiograph = transformed["image"];
        mask = transformed["mask"];

        return radiograph, mask;