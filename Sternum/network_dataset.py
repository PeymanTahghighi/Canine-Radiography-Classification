
from copy import deepcopy
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
from Utility import draw_missing_spine, get_max_contour, post_process, retarget_img, smooth_boundaries
from tqdm import tqdm
from musica import *

class CanineDatasetSeg(Dataset):
    def __init__(self, radiographs, lbl, train = True, exposure_labels = None):
        self.__radiographs = [];
        self.__masks = [];
        self.__train = train;
        reload = False;
        total_zeros = 0;
        total_ones = 0;
        if reload is True:
            for index in tqdm(range(len(radiographs))):
                n = radiographs[index];
                if os.path.exists(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{radiographs[index]}.jpeg')) is True:
                    radiograph = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
                elif os.path.exists(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{radiographs[index]}.jpeg')) is True:
                    radiograph = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
                else:
                    print(radiographs[index]);
                    continue;

                sternum_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Sternum\\labels\\{radiographs[index]}.meta', 'rb'));
                sternum_mask_name = sternum_meta['Sternum'][-1];
                sternum_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Sternum\\labels\\{sternum_mask_name}', cv2.IMREAD_GRAYSCALE);
                sternum_mask = np.where(sternum_mask>0, 255, 0).astype("uint8");
                

                spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{radiographs[index]}.meta', 'rb'));
                spine_mask_name = spine_meta['Spine'][-1];
                spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\{spine_mask_name}', cv2.IMREAD_GRAYSCALE);
                spine_mask = np.where(spine_mask>0, 255, 0);

                thorax_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{radiographs[index]}.png', cv2.IMREAD_GRAYSCALE);
                thorax_mask = cv2.resize(thorax_mask, (radiograph.shape[1], radiograph.shape[0]));

                sternum_mask = ((np.where(thorax_mask>0, 1, 0) * sternum_mask)).astype("uint8");


                spine_mask = smooth_boundaries(spine_mask,10);
                spine_mask = smooth_boundaries(spine_mask,25);
                spine_mask = draw_missing_spine(spine_mask);


                residual = np.maximum(np.int32(thorax_mask) - np.int32(spine_mask), np.zeros_like(spine_mask)).astype("uint8");
                # sym_line = get_symmetry_line(cv2.resize(spine_mask.astype("uint8"), (1024, 1024)));
                # residual_mask_left, residual_mask_right = divide_image_symmetry_line(cv2.resize(residual, (1024, 1024)), sym_line);
                # residual_mask_left =  cv2.resize(residual_mask_left, (radiograph.shape[1], radiograph.shape[0]));
                # residual_mask_right =  cv2.resize(residual_mask_right, (radiograph.shape[1], radiograph.shape[0]));
                radiograph = (np.int32(radiograph) * np.where(thorax_mask>0, 1, 0)).astype("uint8");
                radiograph = (np.int32(radiograph) * np.where(spine_mask>0, 0, 1)).astype("uint8");

                ret, residual_mask = retarget_img([sternum_mask, radiograph], residual);
                sternum_mask = ret[0];
                radiograph = ret[1];
                sternum_mask = post_process(sternum_mask);

                # cv2.imshow('after', cv2.resize(spine_mask, (512,512)))
                # cv2.imshow('before', cv2.resize(bef, (512,512)));
                # cv2.waitKey();

                returned_contour = np.zeros_like(residual_mask);
 
                sternum_contours = cv2.findContours(sternum_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
                for idx,c in enumerate(sternum_contours):
                    tmp = np.zeros_like(residual_mask);
                    tmp = cv2.drawContours(tmp, [c], 0, (255,255,255), -1);
                    area_before = np.sum(tmp)/255;
                    residual_tmp = ((np.where(residual_mask>0, 1, 0) * np.where(tmp>0, 1, 0))*255).astype("uint8");
                    area_after = np.sum(residual_tmp)/255;
                    #cv2.imwrite(f'tmp\\{file_name}_{idx}.png', tmp);
                    #cv2.imwrite(f'tmp\\{file_name}_resstr_{idx}.png', residual_tmp);
                    rat = (area_after / area_before);
                    # b = cv2.addWeighted(residual_tmp, 0.5, residual, 0.5, 0.0);
                    # cv2.imshow('b', cv2.resize(b, (512,512)));
                    # cv2.waitKey();
                    if rat > 0.3 :
                        #print(rat);
                        contours_res,_ = get_max_contour(residual_tmp);
                        #contours_res= cv2.findContours(residual_tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
                        rect = cv2.boundingRect(contours_res);
                        rect = list(rect);
                        returned_contour = cv2.drawContours(returned_contour, [contours_res], 0, (255, 255, 255), -1);

                radiograph = np.expand_dims(radiograph, axis=2);
                radiograph = np.repeat(radiograph, 3,axis=2);
                # if self.__train is True:
                #     transformed = config.train_transforms(image = radiograph_image, mask = mask);
                # else:
                #     transformed = config.valid_transforms(image = radiograph_image, mask = mask);


                # radiograph_image = transformed["image"];
                # mask = transformed["mask"];

                self.__radiographs.append(radiograph);
                self.__masks.append(np.where(returned_contour>0, 1, 0));
            if train is True:
                pickle.dump([self.__radiographs, self.__masks], open(f'sternum_data_train.dmp', 'wb'));
            else:
                pickle.dump([self.__radiographs, self.__masks], open(f'sternum_data_test.dmp', 'wb'));
        else:
            if train is True:
                self.__radiographs, self.__masks  = pickle.load(open(f'sternum_data_train.dmp', 'rb'))
            else:
                self.__radiographs, self.__masks  = pickle.load(open(f'sternum_data_test.dmp', 'rb'))

            
    def __len__(self):
        return len(self.__radiographs);

    def __getitem__(self, index):
        radiograph = self.__radiographs[index];
        mask =  self.__masks[index];
        fig, ax = plt.subplots(1, 2);
        ax[0].imshow(radiograph);
        ax[1].imshow(mask);
        plt.show();

        if self.__train is True:
            transformed = config.train_transforms_seg(image = radiograph, mask = mask);
        else:
            transformed = config.valid_transforms_seg(image = radiograph, mask = mask);


        radiograph = transformed["image"];
        mask = transformed["mask"];
        
        return radiograph, mask;

def preload_classification_dataset(fold_cnt, radiographs, train = True):
    if train is True:
        if os.path.exists(f'{fold_cnt}\\train\\1') is False:
            os.makedirs(f'{fold_cnt}\\train\\1');
        if os.path.exists(f'{fold_cnt}\\train\\0') is False:
            os.mkdir(f'{fold_cnt}\\train\\0');
    if train is False:
        if os.path.exists(f'{fold_cnt}\\test\\1') is False:
            os.mkdir(f'{fold_cnt}\\test\\1');
        if os.path.exists(f'{fold_cnt}\\test\\0') is False:
            os.mkdir(f'{fold_cnt}\\test\\0');
        
    for index in tqdm(range(len(radiographs))):
        n = radiographs[index];
        if os.path.exists(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{radiographs[index]}.jpeg')) is True:
            radiograph_image = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
        elif os.path.exists(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{radiographs[index]}.jpeg')) is True:
            radiograph_image = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
        else:
            print(radiographs[index]);
        if os.path.exists(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{radiographs[index]}.png') is False:
            print(n);
            continue;

        full_body_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{radiographs[index]}.png', 
        cv2.IMREAD_GRAYSCALE);
        kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
        full_body_mask = cv2.erode(full_body_mask, kernel, iterations=10);
        full_body_mask = cv2.resize(full_body_mask, (radiograph_image.shape[1], radiograph_image.shape[0]));
        radiograph_image = ((np.where(full_body_mask>0, 1, 0) * radiograph_image)).astype("uint8");

        sternum_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Sternum\\labels\\{radiographs[index]}.meta', 'rb'));
        sternum_mask_name = sternum_meta['Sternum'][-1];
        sternum_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Sternum\\labels\\{sternum_mask_name}', cv2.IMREAD_GRAYSCALE);
        sternum_mask = np.where(sternum_mask>0, 255, 0);
        sternum_mask = ((np.where(full_body_mask>0, 1, 0) * sternum_mask)).astype("uint8");
        h,w = sternum_mask.shape;

        spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{radiographs[index]}.meta', 'rb'));
        spine_mask_name = spine_meta['Spine'][-1];
        spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\{spine_mask_name}', cv2.IMREAD_GRAYSCALE);
        spine_mask = np.where(spine_mask>0, 255, 0).astype("uint8")
        bef = deepcopy(spine_mask);

        spine_mask = smooth_boundaries(spine_mask,10);
        spine_mask = smooth_boundaries(spine_mask,25);
        spine_mask = draw_missing_spine(spine_mask);
        spine_mask = np.uint8(spine_mask);
        full_body_mask = (np.int32(full_body_mask) * np.where(spine_mask>0,0,1)).astype("uint8")

        # cv2.imshow('after', cv2.resize(spine_mask, (512,512)))
        # cv2.imshow('before', cv2.resize(bef, (512,512)));
        # cv2.waitKey();

        radiograph_image = (np.int32(radiograph_image) * np.where(spine_mask>1, 0, 1)).astype("uint8");
        #mask = (np.int32(mask) * np.where(spine_mask>1, 0, 1)).astype("uint8");
        sternum_contours = cv2.findContours(sternum_mask.astype("uint8"), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];

        returned_sternums = np.zeros_like(sternum_mask);
        cnt = 0;
        for idx,c in enumerate(sternum_contours):
            tmp = np.zeros_like(sternum_mask);
            tmp = cv2.drawContours(tmp, [c], 0, (255,255,255), -1);
            area_before = np.sum(tmp)/255;
            residual_tmp = ((np.where(spine_mask>0, 0, 1) * np.where(tmp>0, 1, 0))*255).astype("uint8");
            area_after = np.sum(residual_tmp)/255;
            rat = (area_after / area_before);
            bbox = cv2.boundingRect(c);
            if rat > 0.6 and bbox[2] > 10 and bbox[3] > 10:
                start_x = int(max((bbox[1] - bbox[3] * config.ENLARGE_RATE), 0));
                end_x = int(min((bbox[1] + bbox[3] +  bbox[3] * config.ENLARGE_RATE), h));

                start_y = int(max((bbox[0] - bbox[2] * config.ENLARGE_RATE), 0));
                end_y = int(min((bbox[0] + bbox[2] +  bbox[2] * config.ENLARGE_RATE), w));

                sternum = radiograph_image[start_x:end_x, start_y:end_y];
                if train is True:
                    cv2.imwrite(f'{fold_cnt}\\train\\1\\{radiographs[index]}_{cnt}.png', sternum);
                else:
                    cv2.imwrite(f'{fold_cnt}\\test\\1\\{radiographs[index]}_{cnt}.png', sternum);

                returned_sternums = cv2.drawContours(returned_sternums, [c], 0, (255,255,255), -1);

                cnt+=1;
  
        ret, full_body_mask = retarget_img([radiograph_image,returned_sternums], full_body_mask);
        radiograph_image = ret[0];
        mask = np.where(ret[1]>0, 1, 0);


        full_body_mask = (np.int32(full_body_mask) * np.where(mask>0, 0, 1)).astype("uint8");
        poss = np.where(full_body_mask);
        cv2.imshow('f', cv2.resize(full_body_mask, (512,512)));
        cv2.waitKey();
        if train is True:
            cnt_crp = 100;
        else:
            cnt_crp = 10;
        while(cnt_crp != 0):
            start_rnd = np.random.randint(0, len(poss[0]));
            start_x = poss[0][start_rnd];
            start_y = poss[1][start_rnd];
            end_x = np.random.randint(100,300);
            ar = np.random.rand()/2 + 0.5;
            end_y = int(ar*end_x);
            w,h = full_body_mask.shape;
            crop = radiograph_image[start_x:min(end_x+start_x, w) , start_y:min(end_y+start_y, h)];
            str_crp_s = np.sum(mask[start_x:min(end_x+start_x, w) , start_y:min(end_y+start_y, h)]);
            mask_crp = full_body_mask[start_x:min(end_x+start_x, w) , start_y:min(end_y+start_y, h)];
            rat = (np.sum(mask_crp)/255) / (mask_crp.shape[0] * mask_crp.shape[1]);
            if str_crp_s <= 0 and rat > 0.7:
                cnt_crp -= 1;
                if train is True:
                    cv2.imwrite(f'{fold_cnt}\\train\\0\\{radiographs[index]}_{cnt_crp}.png', crop);
                else:
                    cv2.imwrite(f'{fold_cnt}\\test\\0\\{radiographs[index]}_{cnt_crp}.png', crop);

class CanineDatasetClass(Dataset):
    def __init__(self, fold_cnt, train = True):
        
        self.__img = [];
        self.__train = train;
        if train is True:
            pos = glob(f'{fold_cnt}\\train\\1\\*.png');
            self.__positives = pos;
            neg = np.array(glob(f'{fold_cnt}\\train\\0\\*.png'));
            r = np.random.randint(0,len(neg), len(pos));
            neg = [neg[i] for i in r];
            self.__img.extend(pos);
            self.__img.extend(neg);
            self.__fold_cnt = fold_cnt;
        else:
            pos = glob(f'{fold_cnt}\\test\\1\\*.png');
            neg = glob(f'{fold_cnt}\\test\\0\\*.png');
            self.__img.extend(pos);
            self.__img.extend(neg);
        
        self.__lbl = [];
        self.__lbl.extend(np.ones(len(pos)));
        self.__lbl.extend(np.zeros(len(neg)));

    def resample(self):
        neg = np.array(glob(f'{self.__fold_cnt}\\train\\0\\*.png'));
        r = np.random.randint(0,len(neg), len(self.__positives));
        neg = [neg[i] for i in r];
        self.__img.clear();
        self.__img.extend(self.__positives);
        self.__img.extend(neg);

    def __len__(self):
        return len(self.__img);

    def __getitem__(self, index):
        radiograph = cv2.imread(self.__img[index],cv2.IMREAD_GRAYSCALE);
        lbl =  self.__lbl[index];
        radiograph = cv2.cvtColor(radiograph, cv2.COLOR_GRAY2RGB);

        if self.__train is True:
            transformed = config.train_transforms_class(image = radiograph);
        else:
            transformed = config.valid_transforms_class(image = radiograph);

        radiograph = transformed["image"];

        return radiograph, lbl;