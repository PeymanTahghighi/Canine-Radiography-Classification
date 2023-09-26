import cv2
import numpy as np
import pandas as pd
import pickle


if __name__ == "__main__":
    img_list = glob('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final\\*.');
    data = pickle.load(open('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\SpineandRibs.uog', 'rb'));
    for i in img_list:
        file_name = os.path.basename(i);
        file_name = file_name[:file_name.rfind('.')];

        if f'{file_name}.jpeg' not in data.keys() and f'{file_name}.png' not in data.keys():
            print(file_name);