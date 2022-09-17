import os
from pickletools import optimize
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from glob import glob
import cv2
import numpy as np
from utility import extract_sternum_features, scale_width, smooth_boundaries
#from optimize_models import optimize_caudal_model, optimize_cranial_model, optimize_full_model, optimize_sternum_model
from utility import divide_image_symmetry_line, get_symmetry_line
import config
from deep_learning.network import Unet
from deep_learning.model_trainer import NetworkTrainer, evaluate_test_data, train_caudal_model, train_cranial_model, train_full_model, train_sternum_model, train_symmetry_model
from Symmetry.thorax import segment_thorax
from utils import create_folder, extract_cranial_features, extract_symmetry_features
from tqdm import tqdm

#---------------------------------------------------------
def update_folds(root_dataframe, ):

    #find intersection with spine and ribs since it hasn't been labelled yet
    spine_and_ribs = pickle.load(open('C:\\Users\\Admin\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\SpineandRibs.uog','rb'));
    img_list_all = list(root_dataframe['Image']);
    img_list_all = list(map(str, img_list_all));
    lbl_list_all = list(root_dataframe['Diagnosis']);
    cranial_list_all = list(root_dataframe['Cranial']);
    caudal_list_all = list(root_dataframe['Caudal']);
    symmetry_list_all = list(root_dataframe['Symmetric Hemithoraces']);
    sternum_list_all = list(root_dataframe['Sternum']);

    img_list = [];
    lbl_list = [];
    mask_list = [];
    grain_lbl_list = [];
    sternum_features_list = [];
    cranial_features_list = [];
    caudal_features_list = [];
    symmetry_features_list = [];

    #create_folder(f'results\\train_data\\');
    #create_folder('cache');

    f = open('C:\\Users\\Admin\\Desktop\\list.txt', 'r');
    img_list_t = [];
    for l in f.readlines():
        img_list_t.append(l.strip())
    for s in tqdm(spine_and_ribs.keys()):
        if spine_and_ribs[s][0]=='labeled':
            file_name = s[:s.rfind('.')];
            meta_file = pickle.load(open(f'C:\\Users\\Admin\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{file_name}.meta', 'rb'));
            
            if False:
                curr_masks  = [];
                
                idx = img_list_all.index(file_name);
                img_list.append(img_list_all[idx]);
                lbl_list.append(lbl_list_all[idx]);
                grain_lbl_list.append([cranial_list_all[idx], caudal_list_all[idx], symmetry_list_all[idx], sternum_list_all[idx]]);
                spine_and_rib_mask_meta = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb')) ;

                spine_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, 'labels', spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
                spine_mask = cv2.resize(spine_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
                ribs_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, 'labels', spine_and_rib_mask_meta['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
                ribs_mask = cv2.resize(ribs_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
                rib_spine_mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
                rib_spine_mask[spine_mask] = 2;
                rib_spine_mask[ribs_mask] = 1;
                rib_spine_mask = np.int32(rib_spine_mask);
                spine_mask = np.int32(spine_mask);
                ribs_mask = np.int32(ribs_mask);
                pickle.dump(rib_spine_mask, open(f'cache\\{file_name}_SR.msk', 'wb'));

                curr_masks.append(f'cache\\{file_name}_SR.msk');

                diaphragm_mask_meta = pickle.load(open(f'{config.D_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb'));
                diaphragm_mask = cv2.imread(os.path.join(config.D_PROJECT_ROOT, 'labels', diaphragm_mask_meta['Diaphragm'][2]), cv2.IMREAD_GRAYSCALE);
                diaphragm_mask = cv2.resize(diaphragm_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                diaphragm_mask = np.where(diaphragm_mask > 0, 1, 0);
                pickle.dump(diaphragm_mask, open(f'cache\\{file_name}_D.msk', 'wb'));
                curr_masks.append(f'cache\\{file_name}_D.msk');

                sternum_mask_meta = pickle.load(open(f'{config.ST_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb'));
                sternum_mask = cv2.imread(os.path.join(config.ST_PROJECT_ROOT, 'labels', sternum_mask_meta['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
                sternum_mask = cv2.resize(sternum_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                sternum_mask = np.where(sternum_mask > 0, 1, 0);
                pickle.dump(sternum_mask, open(f'cache\\{file_name}_ST.msk', 'wb'));
                curr_masks.append(f'cache\\{file_name}_ST.msk');

                whole_thorax = segment_thorax(np.uint8(ribs_mask*255));
                
                #cranial
                cranial = np.uint8(spine_mask*255) - whole_thorax;
                cranial_features = extract_cranial_features(cranial);

                #Caudal
                caudal = np.uint8(diaphragm_mask*255) - whole_thorax;
                caudal_features = extract_cranial_features(caudal);
                #-----------------------------------------------------

                #sternum
                spine_mask_processed = smooth_boundaries(spine_mask,10);
                spine_mask_processed = smooth_boundaries(spine_mask_processed,25);
                spine_mask_processed = scale_width(spine_mask_processed, 3);
                sternum = np.logical_and(sternum_mask.squeeze(), np.where(whole_thorax>0,1,0)).astype(np.uint8);
                sternum_features = extract_sternum_features(sternum,spine_mask_processed);
                #-----------------------------------------------------

                #symmetry
                sym_line = get_symmetry_line(spine_mask*255);
                ribs_left, ribs_right = divide_image_symmetry_line(ribs_mask*255, sym_line);
                thorax_left = segment_thorax(ribs_left);
                thorax_right = segment_thorax(ribs_right);
                symmetry_features = extract_symmetry_features(thorax_left, thorax_right);
                #------------------------------------------------------

                sternum_features_list.append(sternum_features);
                symmetry_features_list.append(symmetry_features);
                cranial_features_list.append(cranial_features);
                caudal_features_list.append(caudal_features);


                mask_list.append(curr_masks);

                #store thorax
                cv2.imwrite(f'results\\train_data\\{file_name}.png', whole_thorax);
                cv2.imwrite(f'results\\train_data\\{file_name}_left.png', thorax_left);
                cv2.imwrite(f'results\\train_data\\{file_name}_right.png', thorax_right);
            else:
                curr_masks  = [];
                
                idx = img_list_all.index(file_name);
                img_list.append(img_list_all[idx]);
                lbl_list.append(lbl_list_all[idx]);
                grain_lbl_list.append([cranial_list_all[idx], caudal_list_all[idx], symmetry_list_all[idx], sternum_list_all[idx]]);
                spine_and_rib_mask_meta = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb')) ;

                spine_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, 'labels', spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
                spine_mask = cv2.resize(spine_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
                ribs_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, 'labels', spine_and_rib_mask_meta['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
                ribs_mask = cv2.resize(ribs_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
                rib_spine_mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
                rib_spine_mask[spine_mask] = 2;
                rib_spine_mask[ribs_mask] = 1;
                rib_spine_mask = np.int32(rib_spine_mask);
                spine_mask = np.int32(spine_mask);
                ribs_mask = np.int32(ribs_mask);
                

                curr_masks.append(f'cache\\{file_name}_SR.msk');

                diaphragm_mask_meta = pickle.load(open(f'{config.D_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb'));
                diaphragm_mask = cv2.imread(os.path.join(config.D_PROJECT_ROOT, 'labels', diaphragm_mask_meta['Diaphragm'][2]), cv2.IMREAD_GRAYSCALE);
                diaphragm_mask = cv2.resize(diaphragm_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                diaphragm_mask = np.where(diaphragm_mask > 0, 1, 0);
                curr_masks.append(f'cache\\{file_name}_D.msk');

                sternum_mask_meta = pickle.load(open(f'{config.ST_PROJECT_ROOT}\\labels\\{file_name}.meta', 'rb'));
                sternum_mask = cv2.imread(os.path.join(config.ST_PROJECT_ROOT, 'labels', sternum_mask_meta['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
                sternum_mask = cv2.resize(sternum_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
                sternum_mask = np.where(sternum_mask > 0, 1, 0);
                curr_masks.append(f'cache\\{file_name}_ST.msk');

                whole_thorax = cv2.imread(f'results\\train_data\\{file_name}.png', cv2.IMREAD_GRAYSCALE);
                
                #cranial
                cranial = np.uint8(spine_mask*255) - whole_thorax;
                cranial_features = extract_cranial_features(cranial);

                #Caudal
                caudal = np.uint8(diaphragm_mask*255) - whole_thorax;
                caudal_features = extract_cranial_features(caudal);
                #-----------------------------------------------------

                #sternum
                spine_mask_processed = smooth_boundaries(spine_mask,10);
                spine_mask_processed = smooth_boundaries(spine_mask_processed,25);
                spine_mask_processed = scale_width(spine_mask_processed, 3);
                sternum = np.logical_and(sternum_mask.squeeze(), np.where(whole_thorax>0,1,0)).astype(np.uint8);
                sternum_features = extract_sternum_features(sternum,spine_mask_processed);
                #-----------------------------------------------------

                #symmetry
                sym_line = get_symmetry_line(spine_mask*255);
                ribs_left, ribs_right = divide_image_symmetry_line(ribs_mask*255, sym_line);
                thorax_left = cv2.imread(f'results\\train_data\\{file_name}_left.png', cv2.IMREAD_GRAYSCALE);
                thorax_right = cv2.imread(f'results\\train_data\\{file_name}_right.png', cv2.IMREAD_GRAYSCALE);
                symmetry_features = extract_symmetry_features(thorax_left, thorax_right);
                #------------------------------------------------------

                sternum_features_list.append(sternum_features);
                symmetry_features_list.append(symmetry_features);
                cranial_features_list.append(cranial_features);
                caudal_features_list.append(caudal_features);


                mask_list.append(curr_masks);

                #store thorax
                # cv2.imwrite(f'results\\train_data\\{file_name}.png', whole_thorax);
                # cv2.imwrite(f'results\\train_data\\{file_name}_left.png', thorax_left);
                # cv2.imwrite(f'results\\train_data\\{file_name}_right.png', thorax_right);
                

    
    img_list = np.array(img_list);
    mask_list = np.array(mask_list);
    lbl_list = np.array(lbl_list);
    grain_lbl_list = np.array(grain_lbl_list);
    sternum_features_list = np.array(sternum_features_list);
    symmetry_features_list = np.array(symmetry_features_list);
    cranial_features_list = np.array(cranial_features_list);
    caudal_features_list = np.array(caudal_features_list);

    pickle.dump(img_list,open(f'cache\\img_list.dmp', 'wb'));
    pickle.dump(mask_list,open(f'cache\\mask_list.dmp', 'wb'));
    pickle.dump(lbl_list,open(f'cache\\lbl_list.dmp', 'wb'));
    pickle.dump(grain_lbl_list,open(f'cache\\grain_lbl_list.dmp', 'wb'));
    pickle.dump(sternum_features_list,open(f'cache\\sternum_features_list.dmp', 'wb'));
    pickle.dump(cranial_features_list,open(f'cache\\cranial_features_list.dmp', 'wb'));
    pickle.dump(caudal_features_list,open(f'cache\\caudal_features_list.dmp', 'wb'));
    pickle.dump(symmetry_features_list,open(f'cache\\symmetry_features_list.dmp', 'wb'));

    # lbl_list =  le.fit_transform(lbl_list);
    # skfold = StratifiedKFold(num_folds, shuffle=True, random_state=42);
    # fold_cnt = 0;
    # for train_idx, test_idx in skfold.split(img_list, lbl_list):
    #     pickle.dump([img_list[train_idx], 
    #     mask_list[train_idx], 
    #     lbl_list[train_idx],
    #     grain_lbl_list[train_idx], 
    #     cranial_features_list[train_idx], 
    #     caudal_features_list[train_idx], 
    #     symmetry_features_list[train_idx], 
    #     sternum_features_list[train_idx], 
    #     img_list[test_idx], 
    #     mask_list[test_idx],
    #     lbl_list[test_idx], 
    #     grain_lbl_list[test_idx]], 
    #     open(f'cache\\{fold_cnt}.fold', 'wb'));
    #     fold_cnt += 1;
#---------------------------------------------------------

def store_folds(num_folds = 5):
    img_list = [];
    lbl_list = [];
    mask_list = [];
    grain_lbl_list = [];
    sternum_features_list = [];
    cranial_features_list = [];
    caudal_features_list = [];
    symmetry_features_list = [];

    img_list = pickle.load(open(f'cache\\img_list.dmp', 'rb'));
    mask_list = pickle.load(open(f'cache\\mask_list.dmp', 'rb'));
    lbl_list = pickle.load(open(f'cache\\lbl_list.dmp', 'rb'));
    grain_lbl_list = pickle.load(open(f'cache\\grain_lbl_list.dmp', 'rb'));
    sternum_features_list = pickle.load(open(f'cache\\sternum_features_list.dmp', 'rb'));
    cranial_features_list = pickle.load(open(f'cache\\cranial_features_list.dmp', 'rb'));
    caudal_features_list = pickle.load(open(f'cache\\caudal_features_list.dmp', 'rb'));
    symmetry_features_list = pickle.load(open(f'cache\\symmetry_features_list.dmp', 'rb'));


    le = LabelEncoder();
    lbl_list =  le.fit_transform(lbl_list);
    skfold = StratifiedKFold(num_folds, shuffle=True, random_state=42);
    fold_cnt = 0;
    for train_idx, test_idx in skfold.split(img_list, lbl_list):
        pickle.dump([img_list[train_idx], 
        mask_list[train_idx], 
        lbl_list[train_idx],
        grain_lbl_list[train_idx], 
        cranial_features_list[train_idx], 
        caudal_features_list[train_idx], 
        symmetry_features_list[train_idx], 
        sternum_features_list[train_idx], 
        img_list[test_idx], 
        mask_list[test_idx],
        lbl_list[test_idx], 
        grain_lbl_list[test_idx]], 
        open(f'cache\\{fold_cnt}.fold', 'wb'));
        fold_cnt += 1;

#---------------------------------------------------------
def load_folds():
    fold_lst = glob('cache\\*.fold');
    folds = [];
    for f in fold_lst:
        folds.append(pickle.load(open(f, 'rb')));

    return folds;
#---------------------------------------------------------

if __name__ == "__main__":
    root_dataframe = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');

    #(1-1)
    #update_folds(root_dataframe);
    store_folds();
    #(1-2)
    folds = load_folds();
    #optimize_sternum_model(folds)
    #optimize_cranial_model(folds);
    #optimize_full_model(folds);

    newtwork_trainer = NetworkTrainer();
    spine_and_ribs_segmentation_model = Unet(3).to(config.DEVICE);
    diaphragm_segmentation_model = Unet(1).to(config.DEVICE);
    sternum_segmentation_model = Unet(1).to(config.DEVICE);

    total_cranial = [];
    total_caudal = [];
    total_symmetry = [];
    total_sternum = [];
    total_quality = [];

    #(2)
    for idx in range(0,len(folds)):
        train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
        caudal_features, \
        symmetry_features, \
        sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = folds[idx][0], folds[idx][1], folds[idx][2], folds[idx][3], folds[idx][4], folds[idx][5], folds[idx][6], folds[idx][7], folds[idx][8], folds[idx][9], folds[idx][10], folds[idx][11];
        
        #temp: changing labels for symmetry
        sym_lbl = np.int32(train_grain_lbl[:,2]);
        above_one = sym_lbl > 1;
        one_below = sym_lbl <=1;
        sym_lbl[one_below] = 0;
        sym_lbl[above_one] = 1;
        train_grain_lbl[:,2] = sym_lbl;

        sym_lbl = np.int32(test_grain_lbl[:,2]);
        above_one = sym_lbl > 1;
        one_below = sym_lbl <=1;
        sym_lbl[one_below] = 0;
        sym_lbl[above_one] = 1;
        test_grain_lbl[:,2] = sym_lbl;
        #=======================================

        #create root fold folder
        #create_folder(f'results\\{idx}', delete_if_exists=False);

        print(f'\n================= Starting fold {idx} =================\n');

        c_transform = ColumnTransformer([
        ('onehot', OneHotEncoder(), [3]),
        ('nothing', 'passthrough', [0,1,2])
        ]);


        full_classification_model = train_full_model(idx, c_transform.fit_transform(train_grain_lbl).astype(np.float32), train_lbl);
        
        #(2-1)
        print('------------- Training spine and ribs model ---------------\n');
        cranial_classification_model = train_cranial_model(idx, cranial_features, train_grain_lbl[:,0]);
        caudal_classification_model = train_caudal_model(idx, caudal_features, train_grain_lbl[:,1]);
        symmetry_classification_model = train_symmetry_model(idx, symmetry_features, train_grain_lbl[:,2]);
        sternum_classification_model = train_sternum_model(idx, sternum_features, train_grain_lbl[:,3]);
        spine_and_ribs_segmentation_model = newtwork_trainer.train('spine and ribs', 3, spine_and_ribs_segmentation_model, idx, train_imgs, train_mask[:,0], 
        test_imgs, test_mask[:,0], load_trained_model=True);
        
        #(2-2)
        print('------------- Training Diaphragm ---------------\n');
        diaphragm_segmentation_model = newtwork_trainer.train('Diaphragm', 1, diaphragm_segmentation_model, idx,
        train_imgs, train_mask[:,1], test_imgs, test_mask[:,1], load_trained_model=True);

        #(2-3)
        print('------------- Training Sternum ---------------\n');
        sternum_segmentation_model = newtwork_trainer.train('Sternum', 1, sternum_segmentation_model, 
        idx,  train_imgs, train_mask[:,2], test_imgs, test_mask[:,2], load_trained_model=True);

        evaluate_test_data(idx, 
        [spine_and_ribs_segmentation_model, diaphragm_segmentation_model, sternum_segmentation_model], 
        [cranial_classification_model, caudal_classification_model, symmetry_classification_model, sternum_classification_model, full_classification_model],
        test_imgs,
        test_grain_lbl,
        test_lbl,
        c_transform,
        use_saved_features=False);

        # total_cranial.append(cranial_results);
        # total_caudal.append(caudal_results);
        # total_symmetry.append(symmetry_results);
        # total_quality.append(quality_results);
        # total_sternum.append(sternum_results);

    total_cranial = np.mean(total_cranial, axis = 0);
    total_caudal = np.mean(total_caudal, axis = 0);
    total_symmetry = np.mean(total_symmetry, axis = 0);
    total_sternum = np.mean(total_sternum, axis = 0);
    total_quality = np.mean(total_quality, axis = 0);

    
    print(('\n'+'%10s'*5)%('Type', 'Precision', 'Recall', 'F1', 'Accuracy'));
    print(('\n'+'%10s'*1 + '%10f'*4)%('Cranial', total_cranial[0], total_cranial[1], total_cranial[2], total_cranial[3]));
    print(('\n'+'%10s'*1 + '%10f'*4)%('Caudal', total_caudal[0], total_caudal[1], total_caudal[2], total_caudal[3]));
    print(('\n'+'%10s'*1 + '%10f'*4)%('Symmetry', total_symmetry[0], total_symmetry[1], total_symmetry[2], total_symmetry[3]));
    print(('\n'+'%10s'*1 + '%10f'*4)%('Sternum', total_sternum[0], total_sternum[1], total_sternum[2], total_sternum[3]));
    print(('\n'+'%10s'*1 + '%10f'*4)%('Quality', total_quality[0], total_quality[1], total_quality[2], total_quality[3]));
