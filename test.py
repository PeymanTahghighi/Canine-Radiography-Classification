from copy import deepcopy
import pickle
from turtle import color
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def centroid(img):
    M = cv2.moments(img);
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY;

def cut_top(img, ratio = 0.1):
    ret = deepcopy(img);
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_cnt = 0;
    max_bbox = None;
    for c in contours:
        area = cv2.contourArea(c);
        bbox = cv2.boundingRect(c);
        if area > max_cnt:
            max_cnt = area;
            max_bbox = bbox;

    row_to_cut = max_bbox[1] + int(max_bbox[3]*ratio);
    ret[row_to_cut:,:] = 0;

    return ret;

def min_enclosing_circle(img):

    ret = deepcopy(img);
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_area = 0;
    max_cnt = None;
    for c in contours:
        area = cv2.contourArea(c);
        bbox = cv2.boundingRect(c);
        if area > max_area:
            max_area = area;
            max_cnt = c;
    return bbox;


df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
img_list = list(df['Image']);
img_list = list(map(str, img_list));
sym = list(df['Symmetric Hemithoraces']);
total = [];
total_lbl = [];
total_img = [];

for idx, img in enumerate(img_list):
    if img != '97':
        full = cv2.imread(f'results\\train_data\\{img}.png', cv2.IMREAD_GRAYSCALE);
        left = cv2.imread(f'results\\train_data\\{img}_left.png', cv2.IMREAD_GRAYSCALE);
        right = cv2.imread(f'results\\train_data\\{img}_right.png', cv2.IMREAD_GRAYSCALE);
        spine_mask_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{img}.meta', 'rb'));
        spine_name = spine_mask_meta['Spine'][2];
        ribs_name = spine_mask_meta['Ribs'][2];
        spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{spine_name}', cv2.IMREAD_GRAYSCALE);

        spine_mask = cv2.resize(spine_mask, (1024, 1024));
        spine_mask = np.where(spine_mask > 0, 1, 0);
        full_thesh = np.where(full >0, 1, 0);
        inters = full_thesh * spine_mask;
        inters = inters.astype("uint8")*255;
        inters = cut_top(inters);
        contours = cv2.findContours(left, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
        bbox = min_enclosing_circle(left);
        
  
        rows = np.sum(left,axis = 1);
        t = np.where(rows != 0)[0];
        start = t[0];
        pr = left[start,:];
        t = np.where(pr!=0)[0];
        median = np.median(t);
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2RGB);

        left = cv2.rectangle(left, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255,255,255),2);
        #cv2.imwrite(f'tmp\\{img}.png', inters);
        cv2.imshow(f'tmp\\{img}.png', left);
        cv2.waitKey();
        # b = cv2.addWeighted(full,0.5,spine_mask.astype("uint8")*255,0.5,0.0);
        # cv2.imshow('t', b);
        # cv2.imshow('i', inters);
        # #cv2.imshow('t', spine_mask.astype("uint8")*255);
        # cv2.waitKey();
        #cv2.waitKey();
        # cX_f, cY_f = centroid(full);
        # cX_l, cY_l = centroid(left);
        # cX_r, cY_r = centroid(right);
        # left_to_center = abs(cX_f - cX_l);
        # right_to_center = abs(cX_f - cX_r);
        # ratio = min(left_to_center, right_to_center) / max(left_to_center, right_to_center);
        # total.append(ratio);

        # full = cv2.cvtColor(full, cv2.COLOR_GRAY2RGB);

        # lbl = sym[idx];
        # if lbl == 0 or lbl == 1:
        #     lbl = 0;
        # else:
        #     lbl = 1;

        # total_img.append(img);
        # total_lbl.append(lbl);

    # put text and highlight the center
    # cv2.circle(full, (cX_f, cY_f), 5, (0, 0, 255), -1);
    # cv2.circle(full, (cX_l, cY_l), 5, (255, 0, 0), -1);
    # cv2.circle(full, (cX_r, cY_r), 5, (0, 255,0), -1);

    # # display the image
    # cv2.imshow("Image", full)
    # cv2.waitKey(0)

cat = np.zeros((len(total_lbl),2));
cat[:,0] = total;
cat[:,1] = total_lbl;

df = pd.DataFrame(cat,columns=[0,1]);
sns.heatmap(df.corr(), annot=True);
plt.show();
# fig = plt.figure();
# ax = plt.axes();
# cols = ['r', 'b'];
# for i in range(len(total_img)):
#     ax.scatter(total[i],10,color=cols[total_lbl[i]]);
#     ax.annotate(total_img[i],(total[i],10));

# plt.show();