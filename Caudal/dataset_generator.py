
from glob import glob
import cv2
import os
import numpy as np
import pickle
import pandas as pd

def get_thorax_bottom(img):
    rows_sum = np.sum(img, axis = 1);
    pos_rows = np.where(np.where(rows_sum > 0, True, False) == True);
    last_row = pos_rows[0][-1];
    first_row = pos_rows[0][0];
    middle = first_row + (last_row - first_row)/2;
    return int(middle);

def generate_dataset():
    gt = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');
    img_list = list(gt['Image']);
    img_list = list(map(str, img_list));
    ROOT_THORAX = "C:\\PhD\\Thesis\\Tests\\Segmentation Results\\thorax";
    ROOT_DIAPH = "C:\\PhD\\Miscellaneous\\Diaphragm\\labels";

    for m in img_list:
        meta_file = pickle.load(open(os.path.join(ROOT_DIAPH, f'{m}.meta'), 'rb'));

        diaphragm_file_name = meta_file['Diaphragm'][2];
        diaph_img = cv2.imread(os.path.join(ROOT_DIAPH, f'{diaphragm_file_name}'), cv2.IMREAD_GRAYSCALE);
        
        thorax_img = cv2.imread(os.path.join(ROOT_THORAX, f"{m}_thorax.png"), cv2.IMREAD_GRAYSCALE);
            
            
        #get thorax bottom to cut it in half
        thorax_img = cv2.resize(thorax_img, (512,512));
        diaph_img = cv2.resize(diaph_img, (512,512));
        diaph_img = np.where(diaph_img > 0, 1, 0);
        thorax_img = np.where(thorax_img > 0, 1, 0);

        res = np.maximum(diaph_img - thorax_img,np.zeros_like(thorax_img));

        # cv2.imshow('thorax', thorax_img);
        # cv2.imshow('thorax_half', thorax_img[:thorax_bottom,:]);
        # cv2.waitKey();

        cv2.imwrite(f'Results\\{m}.png', res.astype('uint8')*255);



if __name__ == "__main__":
    generate_dataset();