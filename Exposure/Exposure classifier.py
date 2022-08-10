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
    df = pd.read_excel("C:\\PhD\\Thesis\\Dataset\\Data.xlsx");
    total_lbl = list(df['Unnamed: 49']);
    img_list = glob("C:\\PhD\\Thesis\\Dataset\\DVVD\\*");
    lbl_list = [];
    hist_list = [];

    stdsc = StandardScaler();


    for img_path in img_list:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
        mask = cv2.threshold(img, thresh=40, maxval=255, type=cv2.THRESH_BINARY)[1];
        hist = cv2.calcHist([img], [0], mask = mask, histSize=[256], ranges=[0,255]);

        mean = hist.mean();
        std = hist.std();
        hist = (hist - mean) / std
        
        #contrast = get_contrast(img);
        #std = get_std(img);
        hist_list.append([hist]);

        img_name = os.path.basename(img_path);
        img_name = img_name[:img_name.rfind('.')];
        if '(' in img_name:
            img_name = img_name[:img_name.rfind('(')];
        elif '-' in img_name:
            img_name = img_name[:img_name.rfind('-')];

        lbl = total_lbl[int(img_name) - 2];
        if lbl == 1 or lbl == 3 or lbl == 2:
            lbl = 0;
        elif lbl == 0:
            lbl = 1;
        elif lbl == 4:
            lbl = 2;
            #print(img_path);
        lbl_list.append(lbl);

    
    kfold = StratifiedKFold(n_splits=20);

    hist_list = np.array(hist_list).squeeze();
    lbl_list = np.array(lbl_list);

    mean_acc = 0;
    mean_prec = 0;
    mean_rec = 0;
    mean_f1 = 0;
    for train_id, valid_id in kfold.split(hist_list, lbl_list):
        train_X, train_y = hist_list[train_id], lbl_list[train_id];    
        valid_X, valid_y = hist_list[valid_id], lbl_list[valid_id];

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


