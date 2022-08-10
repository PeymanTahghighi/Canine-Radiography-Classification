import os
from glob import glob
from sklearn.svm import SVC
import pickle
import cv2
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import pandas as pd


if __name__ == "__main__":
    df = pd.read_excel("C:\PhD\Thesis\Dataset\data.xlsx");
    #print(df['Unnamed: 51']);
    obliq_no = dict();
    obliq_yes = dict();
    exp_no = dict();
    exp_yes = dict();
    cnt =2;
    for e in df['Unnamed: 51']:
        if e=='No':
            exposure = df.iloc[cnt]['Unnamed: 49'];
            obliquity = df.iloc[cnt]['DV or VD'];

            if obliquity in obliq_no:
                obliq_no[obliquity] += 1;
            else:
                obliq_no[obliquity] = 1;
            
            if exposure in exp_no:
                exp_no[exposure] += 1;
            else:
                exp_no[exposure] = 1;
        
        if e =='Yes':
            exposure = df.iloc[cnt]['Unnamed: 49'];
            obliquity = df.iloc[cnt]['DV or VD'];
            
            if obliquity in obliq_yes:
                obliq_yes[obliquity] += 1;
            else:
                obliq_yes[obliquity] = 1;
            
            if exposure in exp_yes:
                exp_yes[exposure] += 1;
            else:
                exp_yes[exposure] = 1;

        cnt += 1;
    
    print("Obliquity No: ");
    print(obliq_no);
    print("Obliquity Yes: ");
    print(obliq_yes);
    print("Exposure No: ");
    print(exp_no);
    print("Exposure Yes: ");
    print(exp_yes);