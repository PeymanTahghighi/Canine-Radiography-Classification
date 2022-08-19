from glob import glob
import cv2
import os
import numpy as np
import pandas as pd

def get_thorax_bottom(img):
    rows_sum = np.sum(img, axis = 1);
    pos_rows = np.where(np.where(rows_sum > 0, True, False) == True);
    last_row = pos_rows[0][-1];
    first_row = pos_rows[0][0];
    middle = first_row + (last_row - first_row)/2;
    return int(middle);

def generate_dataset():
    gt = pd.read_excel('C:\\PhD\\Thesis\\Data Analysis\\dvvd_list_final.xlsx');
    img_list = list(gt['Image']);
    img_list = list(map(str, img_list));
    ROOT_THORAX = "..\\Segmentation Results\\thorax";
    ROOT_SPINE = "..\\Segmentation Results\\spine";

    for m in img_list:
        if os.path.exists(os.path.join(ROOT_THORAX, f"{m}_thorax.png")) is True and os.path.exists(os.path.join(ROOT_SPINE, f"{m}_s.png")) is True:
            thorax_img = cv2.imread(os.path.join(ROOT_THORAX, f"{m}_thorax.png"), cv2.IMREAD_GRAYSCALE);
            spine_img = cv2.imread(os.path.join(ROOT_SPINE, f"{m}_s.png"), cv2.IMREAD_GRAYSCALE);
            #unnecessary and should be removed
            spine_img = cv2.resize(spine_img, (512,512));
            
            #get thorax bottom to cut it in half
            thorax_bottom = get_thorax_bottom(thorax_img);

            # cv2.imshow('thorax', thorax_img);
            # cv2.imshow('thorax_half', thorax_img[:thorax_bottom,:]);
            # cv2.waitKey();

            result = spine_img - thorax_img;
            result = result[:thorax_bottom,:];
            result = cv2.resize(result, (512,512));
            cv2.imwrite(f'results\\{m}.png', result);
        else:
            print(m);

if __name__ == "__main__":
    generate_dataset();