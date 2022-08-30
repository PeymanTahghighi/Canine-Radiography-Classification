
from re import S
import numpy as np
import cv2
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorchvideo
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class Dataset()

def get_contrast(img):
    kernel = np.ones((5,5), np.uint8);

    min = cv2.erode(img, kernel, iterations=1);
    max = cv2.dilate(img, kernel, iterations=1);

    div_zero_avoid = np.ones(shape=max.shape, )

    contrast = (max - min) / (max + min + div_zero_avoid);
    contrast = np.mean(contrast);

    return contrast;

def get_std(img):
    return img.std();

if __name__ == "__main__":

    
    labeled_imgs = glob('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Exposure\\*.meta');


    stdsc = StandardScaler();

    lbl_dict = dict();
    total_lbl = [];
    total_features = [];
    for img_path in labeled_imgs:
        meta = pickle.load(open(img_path,'rb'));
        lbl = meta['misc'][0];
        if lbl not in lbl_dict:
            lbl_dict[lbl] = 1;
        else:
            lbl_dict[lbl] += 1;
        
        total_lbl.append(lbl);
        

        file_name = os.path.basename(img_path);
        file_name = file_name[:file_name.rfind('.')];

        img = cv2.imread(os.path.join('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f'{file_name}.jpeg'), cv2.IMREAD_GRAYSCALE);
        mask = cv2.threshold(img, thresh=40,maxval=255, type= cv2.THRESH_BINARY)[1];
        #cv2.imshow('m', mask);
        #cv2.waitKey();
        hist = cv2.calcHist([img], [0], mask, [256], [0,255]);
        hist = hist / hist.sum();
        plt.plot(hist);
        plt.savefig(f'res\\{file_name}_{lbl}.png');
        plt.clf();
        cv2.imwrite(f'res\\{file_name}.png', img);
        total_features.append(hist);

    
    print(lbl_dict);

    le = LabelEncoder();
    total_lbl = le.fit_transform(total_lbl);
    
    kfold = StratifiedKFold(n_splits=5);

    total_features = np.array(total_features);
    lbl_list = np.array(total_lbl);

    param_args = {
        'svc__C':[0.001,0.01,0.1,1,10,100],
        'svc__kernel': ['rbf', 'linear']
    };

    pipe = Pipeline([('scalar',StandardScaler()), ('svc',SVC(class_weight='balanced'))]);
    gs = GridSearchCV(pipe, param_args, scoring='f1_macro', n_jobs=-1, cv = 5);
    gs = gs.fit(total_features.squeeze(), total_lbl);
    print(gs.best_score_);

    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True);
    model.classifier =  nn.Linear(1024, 1, bias=True);
    opimizer = optim.Adam(model.parameters(), 1e-5);
    loss = nn.BCEWithLogitsLoss();

    



    mean_acc = 0;
    mean_prec = 0;
    mean_rec = 0;
    mean_f1 = 0;
    for train_id, valid_id in kfold.split(total_features, lbl_list):
        train_X, train_y = total_features[train_id], lbl_list[train_id];    
        valid_X, valid_y = total_features[valid_id], lbl_list[valid_id];

        #X_train_std = stdsc.fit_transform(train_X);
        #valid_X = stdsc.transform(valid_X);

        model = SVC(class_weight='balanced');
        model.fit(train_X, train_y);
        pred = model.predict(valid_X);
        prec, rec, f1, _ = precision_recall_fscore_support(valid_y, pred, average='macro');
        mean_prec += prec;
        mean_rec += rec;
        mean_f1 += f1;

    
    print(f"Average prec: {mean_prec / 20} \tAverage rec: {mean_rec / 20}\tAverage f1: {mean_f1 / 20}");


