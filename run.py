import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from glob import glob
import cv2
import numpy as np
import config
from deep_learning.network import Unet
from deep_learning.model_trainer import NetworkTrainer, evaluate_test_data, train_caudal_model, train_cranial_model, train_full_model
from deep_learning.network_dataset import CanineDataset
from thorax import segment_thorax
from utils import create_folder, extract_cranial_features, save_test_data
from torch.utils.data import DataLoader


#---------------------------------------------------------
def update_folds(root_dataframe, num_folds = 5):

    #find intersection with spine and ribs since it hasn't been labelled yet
    spine_and_ribs = pickle.load(open('D:\\PhD\\Miscellaneous\\Spine and Ribs\\SpineandRibs.uog','rb'));
    img_list_all = list(root_dataframe['Image']);
    img_list_all = list(map(str, img_list_all));
    lbl_list_all = list(root_dataframe['Diagnosis']);
    cranial_list_all = list(root_dataframe['Cranial']);
    caudal_list_all = list(root_dataframe['Caudal']);
    caudal_list_all = list(root_dataframe['Caudal']);
    symmetry_list_all = list(root_dataframe['Symmetry Hemithoraces']);
    sternum_list_all = list(root_dataframe['Sternum']);

    img_list = [];
    lbl_list = [];
    mask_list = [];
    grain_lbl_list = [];
    features_list = [];

    create_folder('cache');

    for s in spine_and_ribs.keys():
        if spine_and_ribs[s][0]=='labeled':
            file_name = s[:s.rfind('.')];
            meta_file = pickle.load(open(f'D:\\PhD\\Miscellaneous\\Spine and Ribs\\labels\\{file_name}.meta', 'rb'));
            if 'Ribs' in meta_file.keys() and 'Spine' in meta_file.keys():
                curr_masks  = [];
                
                idx = img_list_all.index(file_name);
                img_list.append(img_list_all[idx]);
                lbl_list.append(lbl_list_all[idx]);
                grain_lbl_list.append([cranial_list_all[idx], caudal_list_all[idx], symmetry_list_all[idx], sternum_list_all[idx]]);
                spine_and_rib_mask_meta = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb')) ;

                spine_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, 'labels', spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
                spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
                ribs_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, 'labels', spine_and_rib_mask_meta['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
                ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
                rib_spine_mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
                rib_spine_mask[spine_mask] = 2;
                rib_spine_mask[ribs_mask] = 1;
                rib_spine_mask = np.int32(rib_spine_mask);
                pickle.dump(rib_spine_mask, open(f'cache\\{file_name}_SR.msk', 'wb'));

                curr_masks.append(f'cache\\{file_name}_SR.msk');

                diaphragm_mask_meta = pickle.load(open(f'{config.D_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb'));
                diaphragm_mask = cv2.imread(os.path.join(config.D_PROJECT_ROOT, 'labels', diaphragm_mask_meta['Diaphragm'][2]), cv2.IMREAD_GRAYSCALE);
                diaphragm_mask = np.where(diaphragm_mask > 0, 1, 0);
                pickle.dump(diaphragm_mask, open(f'cache\\{file_name}_D.msk', 'wb'));
                curr_masks.append(f'cache\\{file_name}_D.msk');

                sternum_mask_meta = pickle.load(open(f'{config.ST_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb'));
                sternum_mask = cv2.imread(os.path.join(config.ST_PROJECT_ROOT, 'labels', sternum_mask_meta['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
                sternum_mask = np.where(sternum_mask > 0, 1, 0);
                pickle.dump(sternum_mask, open(f'cache\\{file_name}_ST.msk', 'wb'));
                curr_masks.append(f'cache\\{file_name}_ST.msk');

                whole_thorax = segment_thorax(ribs_mask);

                #cranial
                cranial = spine_mask - whole_thorax;
                cranial_features = extract_cranial_features(cranial);

                #Caudal
                caudal = diaphragm_mask - whole_thorax;
                caudal_features = extract_cranial_features(caudal);
                #-----------------------------------------------------

                #sternum
                sternum = np.logical_and(sternum_mask.squeeze(), whole_thorax).astype(np.uint8);
                sternum_features = np.sum(sternum, (1,2));
                #-----------------------------------------------------

                features_list.append(cranial_features, caudal_features, sternum_features);


                mask_list.append(curr_masks);


    le = LabelEncoder();
    lbl_list =  le.fit_transform(lbl_list);

    skfold = StratifiedKFold(num_folds, shuffle=True, random_state=42);
    fold_cnt = 0;
    for train_idx, test_idx in skfold.split(img_list, lbl_list):
        pickle.dump([img_list[train_idx], mask_list[train_idx], lbl_list[train_idx], grain_lbl_list[train_idx], features_list[train_idx], img_list[test_idx], mask_list[test_idx], lbl_list[test_idx], 
        grain_lbl_list[train_idx]], 
        open(f'{fold_cnt}.fold', 'wb'));
        fold_cnt += 1;
#---------------------------------------------------------

#---------------------------------------------------------
def load_folds():
    fold_lst = glob('*.fold');
    folds = [];
    for f in fold_lst:
        folds.append(pickle.load(open(f, 'rb')));
    

    return folds;
#---------------------------------------------------------

if __name__ == "__main__":
    root_dataframe = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');

    #(1-1)
    update_folds(root_dataframe);
    #(1-2)
    folds = load_folds();

    newtwork_trainer = NetworkTrainer();
    spine_and_ribs_segmentation_model = Unet(3);
    diaphragm_segmentation_model = Unet(1);
    sternum_segmentation_model = Unet(1);

    #(2)
    for idx,f in enumerate(folds):
        train_imgs, train_mask, train_lbl, train_grain_lbl, train_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8];

        #create root fold folder
        create_folder(idx);

        print(f'\n================= Starting fold {idx} =================\n');
        fill_model = train_full_model(idx, train_grain_lbl, train_lbl);
        
        #(2-1)
        print('------------- Training spine and ribs model ---------------\n');
        spine_and_ribs_segmentation_model = newtwork_trainer.train('spine and ribs', spine_and_ribs_segmentation_model, idx, train_imgs, train_mask[:,0], test_imgs, test_mask[:,0]);
        #symmetry_model = train_symmetry_model();
        cranial_model = train_cranial_model(idx, train_features[:,0], train_grain_lbl[:,0]);
        caudal_model = train_caudal_model(idx, train_features[:,1], train_grain_lbl[:,1]);


        #(2-2)
        print('------------- Training Diaphragm ---------------\n');
        diaphragm_segmentation_model = newtwork_trainer.train('Diaphragm', diaphragm_segmentation_model, idx,  train_imgs, train_mask[:,1], test_imgs, test_mask[:,1]);

        #(2-3)
        print('------------- Training Sternum ---------------\n');
        sternum_segmentation_model = newtwork_trainer.train('Sternum', sternum_segmentation_model, idx,  train_imgs, train_mask[:,2], test_imgs, test_mask[:,2]);

        evaluate_test_data(idx, [spine_and_ribs_segmentation_model, diaphragm_segmentation_model, sternum_segmentation_model], test_imgs);


