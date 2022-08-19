import os
from tkinter import Grid
from matplotlib.pyplot import contour
import numpy as np
import cv2
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def extract_features(img):
    w,h = img.shape;
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    total_area = 0;
    total_diameter = 0;
    for c in contours:
        total_area += cv2.contourArea(c);
        total_diameter += cv2.arcLength(c, True);

    total_area /= w*h;
    total_diameter /= ((w+h)*2);
    return [total_area, total_diameter];


if __name__ == "__main__":
    gt = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');

    lbl_list = list(gt['Cranial']);

    img_list = list(gt['Image']);
    img_list = list(map(str, img_list));

    total_data = [];

    for idx, img_name in enumerate(img_list):
        if os.path.exists(f"results\\{img_name}.png") is True:
            cranial_part = cv2.imread(f"results\\{img_name}.png", cv2.IMREAD_GRAYSCALE);

            total_data.append(extract_features(cranial_part));
        else:
            #drop
            lbl_list.pop(idx);

    lbl_list = np.array(lbl_list);
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

    param_grid = [
        {
            'svc__C': param_range,
            'svc__kernel': ['linear']
        }
        ,
        {
            'svc__C': param_range,
            'svc__gamma': param_range,
            'svc__kernel': ['rbf']
        }
    ];

    pipe_svc = make_pipeline(StandardScaler(), SVC(class_weight='balanced'));

    svm = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='f1', n_jobs=-1, refit=True);
    svm = svm.fit(X = total_data, y = lbl_list);

    print(svm.best_score_);
    



