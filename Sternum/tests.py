from copyreg import pickle
from glob import glob
import pickle
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from network_trainer import NetworkTrainer

def match_dataset(root):
    meta_files = glob(f"{root}\\labels\\*.meta");
    df = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');
    img_list = list(df['Image']);
    img_list = list(map(str, img_list));
    gt_lbl_list = list(df['Sternum']);

    image_list = [];
    mask_list = [];
    lbl_list = [];

    for idx, m in enumerate(meta_files):
        meta_data = pickle.load(open(m, 'rb'));
        file_name = os.path.basename(m);
        file_name = file_name[:file_name.rfind('.')];
        if file_name in img_list:
            idx = img_list.index(file_name);
            if 'Sternum' in meta_data.keys():
                sternum_mask = cv2.imread(os.path.join(root, 'labels', meta_data['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
                sternum_mask = np.where(sternum_mask > 0, 255, 0).astype("uint8");
                mask_list.append(os.path.join(root, 'labels', meta_data['Sternum'][2]));
                #assign label based on summation of positive pixels
                sternum_mask_sum = sternum_mask.sum()/255;
                if sternum_mask_sum > 10:
                    if gt_lbl_list[idx] != 'Yes' and gt_lbl_list[idx] != 'Mid':
                        print(f'Check {file_name}')
                    #lbl_list.append(1);
                else:
                    if gt_lbl_list[idx] != 'No':
                        print(f'Check {file_name}')

                # #get image from images folder
                # file_name = os.path.basename(m);
                # file_name = file_name[:file_name.rfind('.')];

                # if os.path.exists(os.path.join(root,'images', f"{file_name}.jpeg")):
                #     image_list.append(os.path.join(root,'images', f"{file_name}.jpeg"));
                # elif os.path.exists(os.path.join(root,'images', f"{file_name}.png")):
                #     image_list.append(os.path.join(root,'images', f"{file_name}.png"));
                # else:
                #     print(f"{file_name} does not exists");
            else:
                print(m);
        else:
            print(file_name);

    return image_list, mask_list, lbl_list;

if __name__=="__main__":
    match_dataset('C:\\PhD\\Miscellaneous\\Sternum');