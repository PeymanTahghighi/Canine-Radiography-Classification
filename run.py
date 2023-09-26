import os
from matplotlib import pyplot as plt
import pandas as pd
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import StratifiedKFold
from glob import glob
import cv2
import numpy as np
from sklearn.svm import SVC
from utility import confidence_intervals
from optimize_models import optimize_caudal_model, optimize_cranial_model, optimize_sp_model
import config
from deep_learning.network import Unet
from deep_learning.model_trainer import NetworkTrainer, store_results
from Symmetry.thorax import segment_thorax
from tqdm import tqdm
import matplotlib
import torch
from torchmetrics import Precision, Recall, F1Score, Accuracy
from deep_learning.network_dataset import CanineDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from stopping_strategy import CombinedTrainValid
from torch.utils.tensorboard import SummaryWriter
from deep_learning.loss import dice_loss
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
from copy import deepcopy
RUN_NAME = 'test';

def build_thorax():
    img_list = glob('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Sternum\\images\\*.jpeg');
    for i in tqdm(range(len(img_list))):
       
        img_name = img_list[i];
        file_name = os.path.basename(img_name);
        file_name = file_name[:file_name.rfind('.')];
        #if os.path.exists(f'results\\train_data\\{file_name}.png') is False:
        if os.path.exists(os.path.join(config.SPINE_RIBS_LABEL_DIR, f'{file_name}.meta')) is True:
            spine_and_rib_mask_meta = pickle.load(open(os.path.join(config.SPINE_RIBS_LABEL_DIR, f'{file_name}.meta'), 'rb')) ;

            spine_mask = cv2.imread(os.path.join(config.SPINE_RIBS_LABEL_DIR, spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
            spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);

            ribs_mask = cv2.imread(os.path.join(config.SPINE_RIBS_LABEL_DIR, spine_and_rib_mask_meta['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
            ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
            rib_spine_mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
            rib_spine_mask[spine_mask] = 2;
            rib_spine_mask[ribs_mask] = 1;
            rib_spine_mask = np.int32(rib_spine_mask);

            whole_thorax = segment_thorax(np.uint8(ribs_mask*255));
            whole_thorax = cv2.imwrite(f'results\\train_data\\{file_name}.png', whole_thorax);
        else:
            print(f'{config.SPINE_RIBS_LABEL_DIR}\\{file_name}.meta does not exist!!!');

#---------------------------------------------------------
def update_folds(root_dataframe, ):


    img_list_all = list(root_dataframe['Image']);
    img_list_all = list(map(str, img_list_all));
    lbl_list_all = list(root_dataframe['Diagnosis']);
    cranial_list_all = list(root_dataframe['Cranial']);
    caudal_list_all = list(root_dataframe['Caudal']);
    symmetry_list_all = list(root_dataframe['Symmetric Hemithoraces']);
    sternum_list_all = list(root_dataframe['Sternum']);
    tips_list_all = list(root_dataframe['Tips outside of spine']);
    exp_list_all = list(root_dataframe['Exposure']);

    img_list = [];
    lbl_list = [];
    mask_list = [];
    grain_lbl_list = [];
    sternum_features_list = [];
    cranial_features_list = [];
    caudal_features_list = [];
    symmetry_features_list = [];
    tips_features_list = [];
    for i in tqdm(range(len(img_list_all))):
       
        img_name = img_list_all[i];
        curr_masks  = [];

        idx = img_list_all.index(img_name);

        img_list.append(img_name);
        # lbl_list.append(lbl_list_all[idx]);
        # grain_lbl_list.append([cranial_list_all[idx], caudal_list_all[idx], tips_list_all[idx], sternum_list_all[idx], exp_list_all[idx]]);
        spine_and_rib_mask_meta = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\{img_name}.meta', 'rb')) ;

        spine_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
        #spine_mask = cv2.resize(spine_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
        ribs_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT,  spine_and_rib_mask_meta['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
        #ribs_mask = cv2.resize(ribs_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
        rib_spine_mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
        rib_spine_mask[spine_mask] = 2;
        rib_spine_mask[ribs_mask] = 1;
        rib_spine_mask = np.int32(rib_spine_mask);
        #pickle.dump(rib_spine_mask, open(f'cache\\{img_name}_SR.msk', 'wb'));

        curr_masks.append(f'cache\\{img_name}_SR.msk');

        diaphragm_mask_meta = pickle.load(open(f'{config.D_PROJECT_ROOT}\\{img_list_all[idx]}.meta', 'rb'));
        diaphragm_mask = cv2.imread(os.path.join(config.D_PROJECT_ROOT, diaphragm_mask_meta['Diaphragm'][2]), cv2.IMREAD_GRAYSCALE);
        #diaphragm_mask = cv2.resize(diaphragm_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        diaphragm_mask = np.where(diaphragm_mask > 0, 1, 0);
        #pickle.dump(diaphragm_mask, open(f'cache\\{img_name}_D.msk', 'wb'));
        curr_masks.append(f'cache\\{img_name}_D.msk');


        sternum_mask_meta = pickle.load(open(f'{config.ST_PROJECT_ROOT}\\{img_name}.meta', 'rb'));
        sternum_mask = cv2.imread(os.path.join(config.ST_PROJECT_ROOT,sternum_mask_meta['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
        #sternum_mask = cv2.resize(sternum_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        sternum_mask = np.where(sternum_mask > 0, 1, 0);
        #pickle.dump(sternum_mask, open(f'cache\\{img_name}_ST.msk', 'wb'));
        curr_masks.append(f'cache\\{img_name}_ST.msk');

        spinous_process_mask_meta = pickle.load(open(f'{config.SP_PROJECT_ROOT}\\{img_name}.meta', 'rb'));
        spinous_process_mask = cv2.imread(os.path.join(config.SP_PROJECT_ROOT,spinous_process_mask_meta['Spinous process'][2]), cv2.IMREAD_GRAYSCALE);
        #spinous_process_mask = cv2.resize(spinous_process_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        spinous_process_mask = np.where(spinous_process_mask > 0, 1, 0);
        #pickle.dump(spinous_process_mask, open(f'cache\\{img_name}_SP.msk', 'wb'));
        curr_masks.append(f'cache\\{img_name}_SP.msk');

        heart_mask_meta = pickle.load(open(f'{config.H_PROJECT_ROOT}\\{img_list_all[idx]}.meta', 'rb'));
        heart_mask = cv2.imread(os.path.join(config.H_PROJECT_ROOT, heart_mask_meta['Heart'][2]), cv2.IMREAD_GRAYSCALE);
        #heart_mask = cv2.resize(heart_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        heart_mask = np.where(heart_mask > 0, 1, 0);
        #pickle.dump(heart_mask, open(f'cache\\{img_name}_H.msk', 'wb'));
        curr_masks.append(f'cache\\{img_name}_H.msk');

        a = f'{config.FULL_BODY_LABEL_DIR}\\{img_list_all[idx]}.meta';
        f = open(f'{config.FULL_BODY_LABEL_DIR}\\{img_list_all[idx]}.meta', 'rb');
        full_body_meta = pickle.load(f);
        full_body_mask = cv2.imread(os.path.join(config.FULL_BODY_LABEL_DIR, full_body_meta['Full'][2]), cv2.IMREAD_GRAYSCALE);
        #full_body_mask = cv2.resize(heart_mask, (config.IMAGE_SIZE, config.IMAGE_SIZE));
        full_body_mask = np.where(full_body_mask > 0, 1, 0);
        pickle.dump(full_body_mask, open(f'cache\\{img_name}_FB.msk', 'wb'));
        curr_masks.append(f'cache\\{img_name}_FB.msk');

        mask_list.append(curr_masks);
                
    pickle.dump(img_list,open(f'cache\\img_list.dmp', 'wb'));
    pickle.dump(mask_list,open(f'cache\\mask_list.dmp', 'wb'));
    
    
    img_list = list(np.array(img_list));
    mask_list = np.array(mask_list);

    folds = load_folds();

    fold_cnt = 0;
    for idx in range(0,len(folds)):
        train_imgs,_,test_imgs, _ = folds[idx][0], folds[idx][1], folds[idx][2], folds[idx][3];
        
        train_mask_list = [];
        test_mask_list = [];
        for t in train_imgs:
            i = img_list.index(t);
            train_mask_list.append(mask_list[i]);

        for t in test_imgs:
            i = img_list.index(t);
            test_mask_list.append(mask_list[i]);

        pickle.dump([train_imgs, 
        train_mask_list, 
        test_imgs,
        test_mask_list], 
        open(f'cache\\{fold_cnt}.fold', 'wb'));
        fold_cnt += 1;
    

    pickle.dump(img_list,open(f'cache\\img_list.dmp', 'wb'));
    pickle.dump(mask_list,open(f'cache\\mask_list.dmp', 'wb'));
    # pickle.dump(lbl_list, open(f'cache\\lbl_list.dmp', 'wb'));
    # pickle.dump(grain_lbl_list,open(f'cache\\grain_lbl_list.dmp', 'wb'));
    # pickle.dump(tips_features_list,open(f'cache\\tips_features_list.dmp', 'wb'));
    # pickle.dump(sternum_features_list,open(f'cache\\sternum_features_list.dmp', 'wb'));
    # pickle.dump(cranial_features_list,open(f'cache\\cranial_features_list.dmp', 'wb'));
    #pickle.dump(caudal_features_list,open(f'cache\\caudal_features_list.dmp', 'wb'));
    #pickle.dump(symmetry_features_list,open(f'cache\\symmetry_features_list.dmp', 'wb'));

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
    caudal_features_list = pickle.load(open(f'cache\\cranial_features_list.dmp', 'rb'));
    #symmetry_features_list = pickle.load(open(f'cache\\symmetry_features_list.dmp', 'rb'));
    tips_features_list = pickle.load(open(f'cache\\tips_features_list.dmp', 'rb'));


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
        tips_features_list[train_idx], 
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

def inclusion_results(df):
    """Displays overall inclusion classification results
        It uses outputs from Cranial Classification Model (CRCM) and Caudal Classification Model (CACM)
        Tahghighi, Peyman, et al. "Machine learning can appropriately classify the collimation of ventrodorsal and dorsoventral thoracic radiographic images of dogs and cats.
    " American Journal of Veterinary Research 1.aop (2023): 1-8.
    https://doi.org/10.2460/ajvr.23.03.0062

    :param df: data frame for ground truth labels
    """
    #cranial
    data = pickle.load(open('data_cranial.dmp', 'rb'));
    total_x_cranial, total_y_cranial, custom_cv, all_cranials = data[0], data[1], data[2], data[3];
    total_x_cranial = np.array(total_x_cranial);
    total_y_cranial = np.array(total_y_cranial, np.int32);
    p = (total_y_cranial == 1);
    total_y_cranial[p==True] = 0;
    total_y_cranial[p==False] = 1;
    #---------------------------------------------------

    #caudal
    data = pickle.load(open('data_caudal.dmp', 'rb'));
    total_x_caudal, total_y_caudal, total_imgs,  custom_cv = data[0], data[1],  data[2], data[3];
    total_x_caudal = np.array(total_x_caudal);
    total_y_caudal = np.array(total_y_caudal, np.int32);
    p = (total_y_caudal == 1);
    total_y_caudal[p==True] = 0;
    total_y_caudal[p==False] = 1;
    total_imgs = np.array(total_imgs);
    #---------------------------------------------------

    cranial = np.array(list(df['Cranial']));
    caudal = np.array(list(df['Caudal']));
    img_list = (list(map(str, list(df['Image']))));
    avg = [];
    fold_cnt = 0;
    fig,ax = plt.subplots(figsize = (6,6));
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    total_gt = [];
    total_pred = [];

    for train_id, test_id in custom_cv:
        train_x_cranial, train_y_cranial, test_x_cranial, test_y_cranial = total_x_cranial[train_id], total_y_cranial[train_id], total_x_cranial[test_id], total_y_cranial[test_id];
        train_x_caudal, train_y_caudal, test_x_caudal, test_y_caudal = total_x_caudal[train_id], total_y_caudal[train_id], total_x_caudal[test_id], total_y_caudal[test_id];
        test_imgs = total_imgs[test_id];
        
        model_caudal = make_pipeline(RobustScaler(),
        MLPClassifier(max_iter=500, activation= 'relu', 
        alpha=0.001, hidden_layer_sizes=60, learning_rate='constant',
        solver='adam'));
        model_caudal.fit(train_x_caudal, train_y_caudal);

        model_cranial = make_pipeline(RobustScaler(),
        SVC(kernel='linear', C=0.1));
        model_cranial.fit(train_x_cranial, train_y_cranial);
        
        total_pred_caudal = model_caudal.predict(test_x_caudal);
        total_pred_cranial = model_cranial.predict(test_x_cranial);
        

        for i in range(len(total_pred_cranial)):
            idx = img_list.index(test_imgs[i]);
            cranial_gt_lbl = cranial[idx];
            caudal_gt_lbl = caudal[idx];

            pred_caudal = total_pred_caudal[i];
            pred_caudal = 0 if pred_caudal == 1 else 1;

            pred_cranial = total_pred_cranial[i];
            pred_cranial = 0 if pred_cranial == 1 else 1;

            gt_lbl = cranial_gt_lbl and caudal_gt_lbl;
            pred_lbl = pred_cranial and pred_caudal;

            total_gt.append(0 if gt_lbl==1 else 1);
            total_pred.append(0 if pred_lbl ==1 else 1);
        
        viz = RocCurveDisplay.from_predictions(total_gt,
        total_pred,
            name=f'ROC fold{fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr);
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        
        prec, rec, f1, _ = precision_recall_fscore_support(total_gt, total_pred, average='binary');
        avg.append([prec, rec, f1]);
        fold_cnt+=1;
    
    std = np.std(avg, axis = 0)*100;
    avg = np.mean(avg, axis = 0)*100;

    cm = confusion_matrix(total_gt, total_pred);
    a = cm[0][0];
    cm[0][0] = cm[1][1];
    cm[1][1] = a;
    cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    total_gt = np.array(total_gt);
    total_pred = np.array(total_pred);

    disp = ConfusionMatrixDisplay.from_predictions(total_gt,total_pred, display_labels=['Accept', 'Reject'], cmap=plt.cm.Blues);
    disp.ax_.set_title('Confusion matrix for overal inclusion classification')

    mean_tpr = np.mean(tprs, axis = 0);
    mean_tpr[-1] = 1.0;
    mean_auc = np.mean(aucs);
    std_auc = np.std(aucs);

    ax.plot(
        mean_fpr,
        mean_tpr,
        color = 'b',
        label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw = 2,
        alpha = 0.8
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.plot(np.arange(0.0,1.05,0.05), np.arange(0.0,1.05,0.05), '--', color='green', )
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.set_title('ROC curves for overal inclusion classification')
    ax.legend(loc='best');
    plt.show();

def positioning_results():
    """Displays overall positioning classification results
        It uses outputs from Spinous Process Classification Model (SPCM) and Shifted Sternum Classification Model (SSCM)
    """

    #sp
    data = pickle.load(open('data_sp.dmp', 'rb'));
    total_x_sp, total_y_sp, total_imgs_sp, custom_cv = data[0], data[1], data[2], data[3];
    total_x_sp = np.array(total_x_sp);
    total_y_sp = np.array(total_y_sp, np.int32);
    total_imgs_sp = np.array(total_imgs_sp);
    #---------------------------------------------------

    #sternum
    data = pickle.load(open('data_sternum.dmp', 'rb'));
    total_train_sternum , total_test_sternum = data[0], data[1];
    #---------------------------------------------------

    avg = [];
    fold_cnt = 0;
    fig,ax = plt.subplots(figsize = (6,6));
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    total_gt = [];
    total_pred = [];

    for train_id, test_id in custom_cv:
        train_x_sp, train_y_sp, train_imgs, test_x_sp, test_y_sp, test_imgs = total_x_sp[train_id], total_y_sp[train_id], total_imgs_sp[train_id], total_x_sp[test_id], total_y_sp[test_id], total_imgs_sp[test_id];
        train_x_sternum = [];
        train_y_sternum = [];
        test_x_sternum = [];
        test_y_sternum = [];
        for t in train_imgs:
            train_x_sternum.append(total_train_sternum[t][0])
            train_y_sternum.append(total_train_sternum[t][1])

        for t in test_imgs:
            test_x_sternum.append(total_test_sternum[t][0])
            test_y_sternum.append(total_test_sternum[t][1])
        
        
        model_sp = make_pipeline(RobustScaler(),
        SVC(kernel="rbf", gamma=0.1, C=10.0));
        model_sp.fit(train_x_sp, train_y_sp);

        model_sternum = Pipeline([('scalar', RobustScaler()), ('svc', SVC(C= 0.01, kernel='linear'))])
        model_sternum.fit(train_x_sternum, train_y_sternum);
        
        total_pred_sp = model_sp.predict(test_x_sp);
        total_pred_sternum = model_sternum.predict(test_x_sternum);
        

        for i in range(len(total_pred_sp)):
            #idx = img_list.index(test_imgs[i]);
            sp_gt_lbl = test_y_sp[i];
            sternum_gt_lbl = test_y_sternum[i];

            pred_sp = total_pred_sp[i];

            pred_sternum = total_pred_sternum[i];

            gt_lbl = sp_gt_lbl or sternum_gt_lbl;
            pred_lbl = pred_sternum or pred_sp;

            total_gt.append(gt_lbl);
            total_pred.append(pred_lbl);
        
        viz = RocCurveDisplay.from_predictions(total_gt,
        total_pred,
            name=f'ROC fold{fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr);
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        
        prec, rec, f1, _ = precision_recall_fscore_support(total_gt, total_pred, average='binary');
        avg.append([prec, rec, f1]);
        fold_cnt+=1;
    
    confidence_intervals(avg);
    confidence_intervals(aucs);

    cm = confusion_matrix(total_gt, total_pred);
    a = cm[0][0];
    cm[0][0] = cm[1][1];
    cm[1][1] = a;
    cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    total_gt = np.array(total_gt);
    total_pred = np.array(total_pred);
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    disp = ConfusionMatrixDisplay.from_predictions(total_gt,total_pred, display_labels=['Accept', 'Reject'], cmap=plt.cm.Blues, colorbar=False);
    disp.ax_.set_title('PCM Confusion matrix')

    mean_tpr = np.mean(tprs, axis = 0);
    mean_tpr[-1] = 1.0;
    mean_auc = np.mean(aucs);
    std_auc = np.std(aucs);

    ax.plot(
        mean_fpr,
        mean_tpr,
        color = 'b',
        label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw = 2,
        alpha = 0.8
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ %0.2f std. dev." %(std_auc),
    )

    ax.plot(np.arange(0.0,1.05,0.05), np.arange(0.0,1.05,0.05), '--', color='green', )
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.set_title('PCM ROC curves')
    ax.legend(loc='best');
    plt.show();

def exposure_results():
    """Calculate and display overall exposure model results.
        It uses output from Underexposure Classification Model (UCM) and Overexposure Classification Model (OCM)
    """

    #sp
    data = pickle.load(open('data_sp.dmp', 'rb'));
    _, _, total_imgs_sp, custom_cv = data[0], data[1], data[2], data[3];
    total_imgs_sp = np.array(total_imgs_sp);
    #---------------------------------------------------

    #underexposure
    total_test_underexposed = pickle.load(open('data_underexposure.dmp', 'rb'));
    #---------------------------------------------------

    #overexposure
    total_features_overexposure, total_imgs_overexposure, total_lbl_overexposure = pickle.load(open('data_overexposure.dmp', 'rb'));
    total_imgs_overexposure = list(total_imgs_overexposure);

    avg = [];
    fold_cnt = 0;
    fig,ax = plt.subplots(figsize = (6,6));
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    total_gt = [];
    total_pred = [];
    num_models = 19;
    for train_id, test_id in custom_cv:
        fold_gt = [];
        fold_pred = [];
        train_imgs, test_imgs = total_imgs_sp[train_id], total_imgs_sp[test_id];
        total_pred_underexposed = [];
        total_gt_underexposed = [];
        
        train_x_overexposure = [];
        train_y_overexposure = [];
        test_x_overexposure = [];
        test_y_overexposure = [];
        for t in train_imgs:
            idx = total_imgs_overexposure.index(t);
            train_x_overexposure.append(total_features_overexposure[idx])
            train_y_overexposure.append(total_lbl_overexposure[idx])
            
        
        for t in test_imgs:
            idx = total_imgs_overexposure.index(t);
            test_x_overexposure.append(total_features_overexposure[idx])
            test_y_overexposure.append(total_lbl_overexposure[idx])
            if t not in total_test_underexposed.keys():
                total_pred_underexposed.append(1);
                total_gt_underexposed.append(1);
            else:
                total_pred_underexposed.append(int(total_test_underexposed[t][0][0]));
                total_gt_underexposed.append(total_test_underexposed[t][1]);

        
        total_pred_overexposed = [];
        for i in range(num_models):
            pipe = Pipeline([('scalar',RobustScaler()), \
            ('bc', BaggingClassifier(n_estimators=np.random.randint(20,50), max_samples=np.random.rand()*0.2 + 0.7, max_features=np.random.rand()*0.2 + 0.75))]);

            pipe.fit(train_x_overexposure, train_y_overexposure);
            pred = pipe.predict_proba(test_x_overexposure);
            total_pred_overexposed.append(pred[:,1]);

        total_pred_overexposed = np.array(total_pred_overexposed).T;
        total_pred_overexposed = np.mean(total_pred_overexposed,axis = 1);
        total_pred_overexposed = total_pred_overexposed > 0.4;

        

        for i in range(len(total_pred_overexposed)):
            #idx = img_list.index(test_imgs[i]);
            overexposed_gt_lbl = int(test_y_overexposure[i]);
            underexposed_gt_lbl = int(total_gt_underexposed[i]);

            pred_overexposed = int(total_pred_overexposed[i]);
            pred_underexposed = int(total_pred_underexposed[i]);

            gt_lbl = overexposed_gt_lbl or underexposed_gt_lbl;
            pred_lbl = pred_overexposed or pred_underexposed;

            total_gt.append(gt_lbl);
            total_pred.append(pred_lbl);
            fold_gt.append(gt_lbl);
            fold_pred.append(pred_lbl);
        
        viz = RocCurveDisplay.from_predictions(fold_gt,
        fold_pred,
            name=f'ROC fold{fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr);
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        
        prec, rec, f1, _ = precision_recall_fscore_support(fold_gt, fold_pred, average='binary');
        avg.append([prec, rec, f1]);
        fold_cnt+=1;
    
    confidence_intervals(avg);
    confidence_intervals(aucs);

    cm = confusion_matrix(total_gt, total_pred);
    a = cm[0][0];
    cm[0][0] = cm[1][1];
    cm[1][1] = a;
    cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    total_gt = np.array(total_gt);
    total_pred = np.array(total_pred);

    MEDIUM_SIZE = 30
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    disp = ConfusionMatrixDisplay.from_predictions(total_gt,total_pred, display_labels=['Normal', 'Overexposed or Underexposed'], cmap=plt.cm.Blues, colorbar=False);
    disp.ax_.set_title('ECM Confusion matrix')

    mean_tpr = np.mean(tprs, axis = 0);
    mean_tpr[-1] = 1.0;
    mean_auc = np.mean(aucs);
    std_auc = np.std(aucs);

    ax.plot(
        mean_fpr,
        mean_tpr,
        color = 'b',
        label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw = 2,
        alpha = 0.8
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ %0.2f std. dev." %(std_auc),
    )

    ax.plot(np.arange(0.0,1.05,0.05), np.arange(0.0,1.05,0.05), '--', color='green', )
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.set_title('ECM ROC curves')
    ax.legend(loc='best');
    plt.show();

def overal_results():
    """Displays overall quality classification results
        It uses outputs from Spinous Process Classification Model (SPCM), Shifted Sternum Classification Model (SSCM), Exposure Classification Model (ECM),
        Cranial Classification Model (CRCM) and Caudal Classification Model (CACM)
    """

    #cranial
    data = pickle.load(open('data_cranial.dmp', 'rb'));
    total_x_cranial, total_y_cranial, custom_cv, all_cranials = data[0], data[1], data[2], data[3];
    total_x_cranial = np.array(total_x_cranial);
    total_y_cranial = np.array(total_y_cranial, np.int32);
    p = (total_y_cranial == 1);
    total_y_cranial[p==True] = 0;
    total_y_cranial[p==False] = 1;
    #---------------------------------------------------

    #caudal
    data = pickle.load(open('data_caudal.dmp', 'rb'));
    total_x_caudal, total_y_caudal, total_imgs,  custom_cv = data[0], data[1],  data[2], data[3];
    total_x_caudal = np.array(total_x_caudal);
    total_y_caudal = np.array(total_y_caudal, np.int32);
    p = (total_y_caudal == 1);
    total_y_caudal[p==True] = 0;
    total_y_caudal[p==False] = 1;
    total_imgs = np.array(total_imgs);
    #---------------------------------------------------

    #sp
    data = pickle.load(open('data_sp.dmp', 'rb'));
    total_x_sp, total_y_sp, total_imgs_sp, _ = data[0], data[1], data[2], data[3];
    total_x_sp = np.array(total_x_sp);
    total_y_sp = np.array(total_y_sp, np.int32);
    total_imgs_sp = np.array(total_imgs_sp);
    #---------------------------------------------------

    #sternum
    data = pickle.load(open('data_sternum.dmp', 'rb'));
    total_train_sternum , total_test_sternum = data[0], data[1];
    #---------------------------------------------------

    #underexposure
    total_test_underexposed = pickle.load(open('data_underexposure.dmp', 'rb'));
    #---------------------------------------------------

    #overexposure
    total_features_overexposure, total_imgs_overexposure, total_lbl_overexposure = pickle.load(open('data_overexposure.dmp', 'rb'));
    total_imgs_overexposure = list(total_imgs_overexposure);

    df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    cranial = np.array(list(df['Cranial']));
    caudal = np.array(list(df['Caudal']));
    img_list = (list(map(str, list(df['Image']))));
    num_models = 19;
    total_gt =[];
    total_pred = [];
    fig,ax = plt.subplots(figsize = (6,6));
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    avg = [];
    fold_cnt = 0;
    for train_id, test_id in custom_cv:
        fold_gt = [];
        fold_pred = [];
        train_x_cranial, train_y_cranial, test_x_cranial, test_y_cranial = total_x_cranial[train_id], total_y_cranial[train_id], total_x_cranial[test_id], total_y_cranial[test_id];
        train_x_caudal, train_y_caudal, test_x_caudal, test_y_caudal = total_x_caudal[train_id], total_y_caudal[train_id], total_x_caudal[test_id], total_y_caudal[test_id];
        test_imgs = total_imgs[test_id];

        model_caudal = make_pipeline(RobustScaler(),
        MLPClassifier(max_iter=500, activation= 'relu', 
        alpha=0.001, hidden_layer_sizes=60, learning_rate='constant',
        solver='adam'));
        model_caudal.fit(train_x_caudal, train_y_caudal);

        model_cranial = make_pipeline(RobustScaler(),
        SVC(kernel='linear', C=0.1));
        model_cranial.fit(train_x_cranial, train_y_cranial);
        
        total_pred_caudal = model_caudal.predict(test_x_caudal);
        total_pred_cranial = model_cranial.predict(test_x_cranial);

        train_x_sp, train_y_sp, train_imgs, test_x_sp, test_y_sp = total_x_sp[train_id], total_y_sp[train_id], total_imgs_sp[train_id], total_x_sp[test_id], total_y_sp[test_id];

        train_x_sternum = [];
        train_y_sternum = [];
        test_x_sternum = [];
        test_y_sternum = [];
        for t in train_imgs:
            train_x_sternum.append(total_train_sternum[t][0])
            train_y_sternum.append(total_train_sternum[t][1])

        for t in test_imgs:
            test_x_sternum.append(total_test_sternum[t][0])
            test_y_sternum.append(total_test_sternum[t][1])
        
        
        model_sp = make_pipeline(RobustScaler(),
        SVC(kernel="rbf", gamma=0.1, C=10.0));
        model_sp.fit(train_x_sp, train_y_sp);

        model_sternum = Pipeline([('scalar', RobustScaler()), ('svc', SVC(C= 0.01, kernel='linear'))])
        model_sternum.fit(train_x_sternum, train_y_sternum);
        
        total_pred_sp = model_sp.predict(test_x_sp);
        total_pred_sternum = model_sternum.predict(test_x_sternum);

        total_pred_underexposed = [];
        total_gt_underexposed = [];
        
        train_x_overexposure = [];
        train_y_overexposure = [];
        test_x_overexposure = [];
        test_y_overexposure = [];
        for t in train_imgs:
            idx = total_imgs_overexposure.index(t);
            train_x_overexposure.append(total_features_overexposure[idx])
            train_y_overexposure.append(total_lbl_overexposure[idx])
            
        
        for t in test_imgs:
            idx = total_imgs_overexposure.index(t);
            test_x_overexposure.append(total_features_overexposure[idx])
            test_y_overexposure.append(total_lbl_overexposure[idx])
            if t not in total_test_underexposed.keys():
                total_pred_underexposed.append(1);
                total_gt_underexposed.append(1);
            else:
                total_pred_underexposed.append(int(total_test_underexposed[t][0][0]));
                total_gt_underexposed.append(total_test_underexposed[t][1]);

        
        total_pred_overexposed = [];
        for i in range(num_models):
            pipe = Pipeline([('scalar',RobustScaler()), \
            ('bc', BaggingClassifier(n_estimators=np.random.randint(20,50), max_samples=np.random.rand()*0.2 + 0.7, max_features=np.random.rand()*0.2 + 0.75))]);

            pipe.fit(train_x_overexposure, train_y_overexposure);
            pred = pipe.predict_proba(test_x_overexposure);
            total_pred_overexposed.append(pred[:,1]);

        total_pred_overexposed = np.array(total_pred_overexposed).T;
        total_pred_overexposed = np.mean(total_pred_overexposed,axis = 1);
        total_pred_overexposed = total_pred_overexposed > 0.4;

        for i in range(len(total_pred_cranial)):
            idx = img_list.index(test_imgs[i]);

            cranial_gt_lbl = cranial[idx];
            cranial_gt_lbl = 0 if cranial_gt_lbl==1 else 1;

            caudal_gt_lbl = caudal[idx];
            caudal_gt_lbl = 0 if caudal_gt_lbl==1 else 1;

            pred_caudal = total_pred_caudal[i];
            #pred_caudal = 0 if pred_caudal == 1 else 1;

            pred_cranial = total_pred_cranial[i];
            #pred_cranial = 0 if pred_cranial == 1 else 1;

            sp_gt_lbl = test_y_sp[i];
            sternum_gt_lbl = test_y_sternum[i];

            pred_sp = total_pred_sp[i];
            pred_sternum = total_pred_sternum[i];

            overexposed_gt_lbl = int(test_y_overexposure[i]);
            underexposed_gt_lbl = int(total_gt_underexposed[i]);

            pred_overexposed = int(total_pred_overexposed[i]);
            pred_underexposed = int(total_pred_underexposed[i]);

            gt_lbl = cranial_gt_lbl or caudal_gt_lbl or sp_gt_lbl or sternum_gt_lbl or overexposed_gt_lbl or underexposed_gt_lbl;
            pred_lbl = pred_cranial or pred_caudal or pred_sternum or pred_sp or pred_overexposed or pred_underexposed;

            fold_gt.append(gt_lbl);
            fold_pred.append(pred_lbl);
            total_gt.append(gt_lbl);
            total_pred.append(pred_lbl);
        
        viz = RocCurveDisplay.from_predictions(fold_gt,
        fold_pred,
            name=f'ROC fold{fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr);
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        
        prec, rec, f1, _ = precision_recall_fscore_support(fold_gt, fold_pred, average='binary');
        avg.append([prec, rec, f1]);
        fold_cnt+=1;
    
    confidence_intervals(avg);
    confidence_intervals(aucs);

    MEDIUM_SIZE = 30
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    cm = confusion_matrix(total_gt, total_pred);
    a = cm[0][0];
    cm[0][0] = cm[1][1];
    cm[1][1] = a;
    cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    total_gt = np.array(total_gt);
    total_pred = np.array(total_pred);

    disp = ConfusionMatrixDisplay.from_predictions(total_gt,total_pred, display_labels=['Accepted', 'Rejected'], cmap=plt.cm.Blues, colorbar=False);
    disp.ax_.set_title('Confusion matrix for overall quality classification')

    mean_tpr = np.mean(tprs, axis = 0);
    mean_tpr[-1] = 1.0;
    mean_auc = np.mean(aucs);
    std_auc = np.std(aucs);

    ax.plot(
        mean_fpr,
        mean_tpr,
        color = 'b',
        label = r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw = 2,
        alpha = 0.8
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ %0.2f std. dev." %(std_auc),
    )

    ax.plot(np.arange(0.0,1.05,0.05), np.arange(0.0,1.05,0.05), '--', color='green', )
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.set_title('ROC curves for overall quality classification')
    ax.legend(loc='best');
    plt.show();

def train(task_name, 
          num_classes,
          model, 
          fold_cnt,
          train_imgs, 
          train_mask, 
          test_imgs, 
          test_mask,
           ckpt,
           param,
           multilabel = False):
          

    if os.path.exists(os.path.join('runs',task_name, RUN_NAME, 'results', str(fold_cnt))) is False:
        os.makedirs(os.path.join('runs',task_name, RUN_NAME, 'results', str(fold_cnt)));

    def loss_func(output, gt):
        if num_classes > 1:
            if multilabel is False:
                f_loss = F.cross_entropy(output, gt.squeeze(dim=3).long(), reduction='mean');
                t_loss = dice_loss(output.squeeze(dim=1), gt.squeeze(dim=3), sigmoid=False)
            else:
                f_loss = sigmoid_focal_loss(output.float(), gt.permute(0,3,1,2).float(), reduction='mean');
                t_loss = dice_loss(output.squeeze(dim=1), gt.permute(0,3,1,2), sigmoid=True, multilabel=True);

        else:

            f_loss = sigmoid_focal_loss(output.squeeze(dim=1), gt.float(), reduction="mean");
            t_loss = dice_loss(output.squeeze(dim=1), gt, sigmoid=True)
        return  t_loss + f_loss;

    def train_one_epoch(epoch, loader, model, optimizer):
        epoch_loss = [];
        step = 0;
        pbar = enumerate(loader);
        
        print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for batch_idx, (radiograph, mask) in pbar:
            if config.DEBUG_TRAIN_DATA is True:
                B = radiograph.shape[0];
                radiograph_np = radiograph.permute(0,2,3,1).detach().cpu().numpy();
                mask_np = mask.detach().cpu().numpy();
                for i in range (B):
                    rad = radiograph_np[i];
                    mak = mask_np[i];
                    rad = rad*(0.229, 0.224, 0.225) +  (0.485, 0.456, 0.406)
                    rad = (rad*255).astype("uint8");
                    cv2.imshow('r', rad);
                    cv2.imshow('m', mak.astype("uint8")*125);
                    cv2.waitKey();
            radiograph, mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE)

            with torch.cuda.amp.autocast_mode.autocast():
                pred = model(radiograph.float());
                loss = loss_func(pred, mask) / config.VIRTUAL_BATCH_SIZE;

            scaler.scale(loss).backward();
            epoch_loss.append(loss.item());

            if ((batch_idx+1) % config.VIRTUAL_BATCH_SIZE == 0 or (batch_idx+1) == len(loader)):
                scaler.step(optimizer);
                scaler.update();
                model.zero_grad(set_to_none = True);
            step += 1;

            pbar.set_description(('%10s' + '%10.4g') %(epoch, np.mean(epoch_loss)));
        
        return np.mean(epoch_loss);

    def eval_one_epoch(epoch, loader, model):
        epoch_loss = [];
        total_prec = [];
        total_rec = [];
        total_f1 = [];
        
        pbar = enumerate(loader);
        print(('\n' + '%10s'*5) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i ,(radiograph, mask) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = model(radiograph);
                loss = loss_func(pred, mask);

                epoch_loss.append(loss.item());
                
                if num_classes > 1:
                    if multilabel is False:
                        pred = (torch.softmax(pred, dim = 1)).permute(0,2,3,1);
                        pred = torch.argmax(pred, dim = 3);
                    else:
                        pred = torch.sigmoid(pred)>0.5
                else:
                    pred = torch.sigmoid(pred) > 0.5;
                if multilabel is False:
                    prec = precision_estimator(pred.flatten(), mask.flatten().long());
                    rec = recall_estimator(pred.flatten(), mask.flatten().long());
                    f1 = f1_esimator(pred.flatten(), mask.flatten().long());
                else:
                    prec = precision_estimator(pred, mask.permute(0,3,1,2).long());
                    rec = recall_estimator(pred, mask.permute(0,3,1,2).long());
                    f1 = f1_esimator(pred, mask.permute(0,3,1,2).long());
                
                
                total_prec.append(prec.item());
                total_rec.append(rec.item());
                total_f1.append(f1.item());

                pbar.set_description(('%10s' + '%10.4g'*4) % (epoch, np.mean(epoch_loss),
                np.mean(total_prec), np.mean(total_rec), np.mean(total_f1)))

        return np.mean(epoch_loss), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);
    
    if config.RESUME:
        model.load_state_dict(ckpt['model']);
    else:
        model.reset_weights();

    scaler = torch.cuda.amp.grad_scaler.GradScaler();

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE);

    stopping_strategy = CombinedTrainValid(3.0,10);
    best_loss = 100;
    best_metrics = None;
    epoch = 0;

    if config.RESUME:
        optimizer.load_state_dict(ckpt['optim']);
        stopping_strategy.load_state_dict(ckpt['stopping_strategy']);
        epoch = ckpt['epoch'];
        best_metrics = ckpt['best_metrics'];
        best_loss = best_metrics[0];
        config.RESUME = False;
        print(f'resuming from epoch: {epoch} on fold: {fold_cnt}');
        
    precision_estimator = Precision('binary' if num_classes ==1 else 'multiclass' if multilabel is False else 'multilabel', num_labels=5, num_classes=num_classes, average='macro').to(config.DEVICE);
    recall_estimator = Recall('binary' if num_classes ==1 else 'multiclass' if multilabel is False else 'multilabel', num_labels=5,num_classes=num_classes, average='macro').to(config.DEVICE);
    f1_esimator = F1Score('binary' if num_classes ==1 else 'multiclass' if multilabel is False else 'multilabel', num_labels=5,num_classes=num_classes, average='macro').to(config.DEVICE);

    train_dataset = CanineDataset(train_imgs, train_mask,  train = True, fold=fold_cnt, multilabel=True, cache=True);
    valid_dataset = CanineDataset(test_imgs, test_mask, train = False, fold=fold_cnt, multilabel=True, cache=True);

    train_loader = DataLoader(train_dataset, 
    batch_size= config.BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True, drop_last=False);

    valid_loader = DataLoader(valid_dataset, 
    batch_size= config.BATCH_SIZE, shuffle=False);

    summary = SummaryWriter(os.path.join('runs', 'experiments', RUN_NAME, str(idx)));
    
    while(True):

        # if e<= config.WARMUP_EPOCHS:
        #     for p in optimizer.param_groups:
        #         p['lr'] = self.get_lr(config.LEARNING_RATE,e);

        model.train();
        train_loss = train_one_epoch(epoch, train_loader, model, optimizer);

        model.eval();
        valid_loss, valid_precision, valid_recall, valid_f1 = eval_one_epoch(epoch, valid_loader, model);

        print(f"Valid \tTrain: {train_loss}\tValid {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tF1: {valid_f1}");
        summary.add_scalar('train/train_loss', train_loss, epoch);
        summary.add_scalar('valid/valid_loss', valid_loss, epoch);
        summary.add_scalar('valid/valid_f1', valid_f1, epoch);

        if(valid_loss < best_loss):
            best_loss = valid_loss;
            best_metrics = [valid_loss, valid_precision, valid_recall, valid_f1];
            ckpt = {
                'model': model.state_dict(),
            }
            torch.save(ckpt, os.path.join('runs',task_name, RUN_NAME, 'results', str(fold_cnt), f'{task_name}_best_{param}.ckpt'));
            print(f'New best model found! best f1: {valid_f1}\n saving checkpoint to: ckpt.pt');
        ckpt = {
            'model': model.state_dict(),
            'optim': optimizer.state_dict(),
            'epoch': epoch+1,
            'best_metrics': best_metrics,
            'stopping_strategy': stopping_strategy.state_dict(),
            'fold': fold_cnt,
        }
        torch.save(ckpt, f'ckpt_resume.ckpt');
        
        epoch += 1;
        if epoch > param['epochs']:
            break;
    with open(os.path.join('runs',task_name, RUN_NAME, 'results', str(fold_cnt), f'{task_name}_results_{param}.txt'), 'w') as f:
        f.write(f"Precision: {best_metrics[1]}\tRecall: {best_metrics[2]}\tF1: {best_metrics[3]}");
    return best_metrics[3];

class Parameters:
    def __init__(self, params, folds) -> None:
        self.__params = params;
        self.__folds = folds;
        self.__build_parameter_space();
        self.__iteration = 0;
        self.__current_param = None;
        pass

    def __build_parameter_space(self,):
        self.__param_space = [];
        for k in self.__params.keys():
            for i in range(len(self.__params[k])):
                self.__param_space.append({k:self.__params[k][i]});
        self.__results = np.zeros((len(self.__param_space) , self.__folds));
    def __iter__(self):
        return self;
        pass
    
    def add_results(self, res, fold):
        self.__results[self.__iteration-1][fold] = res;
        self.save_checkpoint();
    
    def __next__(self):
        if self.__iteration < len(self.__param_space):
            x = self.__param_space[self.__iteration];
            self.__current_param = x;
            self.__iteration += 1;
            self.save_checkpoint();
            return x;
        else:
            raise StopIteration;

    def state_dict(self):
        return {'results' : self.__results,
                'param' : self.__current_param};

    def load_state_dict(self, dict):
        param = dict['param'];
        res = dict['results'];
        self.__results = res;
        if param not in self.__param_space:
            print('Parameters state dict loading error, param not found!');
            return;
        self.__iteration = self.__param_space.index(param);

    def get_final_results(self):
        res = deepcopy(self.__results);
        res = np.mean(res, axis = 1);
        print(f'best result with: {self.__param_space[np.argmax(res)]} with f1 of: {np.max(res)}');

    def save_checkpoint(self):
        pickle.dump(self.state_dict(), open('parameters.ckpt', 'wb'));
    
    def load_checkpoint(self):
        dict = pickle.load(open('parameters.ckpt', 'rb'));
        self.load_state_dict(dict);




if __name__ == "__main__":

    #build_thorax();
    if config.REBUILD_THORAX:
        build_thorax();
    
    root_dataframe = pd.read_excel(config.ROOT_DATAFRAME_PATH);


    #update_folds(root_dataframe);
    #store_folds();

    folds = load_folds();

    #optimize_cranial_model(folds, root_dataframe, rebuild_features=True);
    #optimize_caudal_model(folds, root_dataframe, rebuild_features=True);
    #optimize_sp_model(folds, root_dataframe, rebuild_features = False);

    #exposure_results();
    #positioning_results();
    #inclusion_results(root_dataframe);
    #overal_results();
#
    #spine_and_ribs_segmentation_model = UNETR(3, 3, config.IMAGE_SIZE, spatial_dims=2).to(config.DEVICE);
    #spine_and_ribs_segmentation_model = UNETRC(3, config.IMAGE_SIZE,16, 768, 12, 0, 4, 3).to(config.DEVICE);
    #spine_and_ribs_segmentation_model = SwinUNETRC(3, 512, 2, 0.25, 3, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48), feature_size=192).to(config.DEVICE);
    #spine_and_ribs_segmentation_model = SwinUNETR(img_size=512, in_channels=3, out_channels=3, spatial_dims=2).to(config.DEVICE);
    modified_dict = dict();
    # state_dict = pickle.load(open(f'ckpt.pt', 'rb'));
    # for s in state_dict:
    #     if 'encoder' in s:
    #         modified_dict[s[8:]] = state_dict[s];
    
    spine_and_ribs_segmentation_model = Unet(3).to(config.DEVICE);
    diaphragm_segmentation_model = Unet(1).to(config.DEVICE);
    sternum_segmentation_model = Unet(1).to(config.DEVICE);
    spinous_process_segmentation_model = Unet(1).to(config.DEVICE);
    heart_model = Unet(1).to(config.DEVICE);
    full_body_model = Unet(1).to(config.DEVICE);
    multimodel = Unet(3).to(config.DEVICE);

    total_cranial = [];
    total_caudal = [];
    total_symmetry = [];
    total_sternum = [];
    total_quality = [];
    total_tips= [];

    
    start_fold = 0;

    parameters =  Parameters({'epochs':[1,2]}, 5);
    if config.RESUME is True:
        ckpt = torch.load('ckpt_resume.ckpt');
        parameters.load_checkpoint('parameters.ckpt');
        start_fold = ckpt['fold'];

    tune_results = [];

    for param in parameters:
        print(f'optimizing hyperparameters with:\n{param}');
        parameters.save_checkpoint();
        cur_exp_results = [];
        for idx in range(2,len(folds)):
            train_imgs,train_mask, test_imgs, test_mask,= folds[idx][0], folds[idx][1], folds[idx][2], folds[idx][3];
            train_mask = np.array(train_mask);
            test_mask = np.array(test_mask);

            print(f'\n================= Starting fold {idx} =================\n');
                            
            # multimodel.load_state_dict(torch.load((f'results\\{idx}\\multimodel.ckpt'))['model'], strict=False)
            # multimodel.eval();
            diaphragm_segmentation_model.load_state_dict(torch.load((f'results\\{idx}\\Diaphragm.ckpt'))['model'], strict=False)
            spine_and_ribs_segmentation_model.load_state_dict(torch.load((f'results\\{idx}\\spineandribs.ckpt'))['model'], strict=False)
            heart_model.load_state_dict(torch.load((f'results\\{idx}\\heart.ckpt'))['model'], strict=False)
            #full_body_model.load_state_dict(torch.load((f'results\\{idx}\\fullbody.ckpt'))['model'], strict=False)
            store_results(idx, [spine_and_ribs_segmentation_model, diaphragm_segmentation_model, heart_model], test_imgs);

            
            #(2-1)
            # print('------------- Training Diaphragm ---------------\n');
            # train_mask = np.take_along_axis(train_mask, np.repeat(np.array([[0,1,4,5]]),repeats=train_mask.shape[0], axis=0) , axis=1);
            # test_mask = np.take_along_axis(test_mask, np.repeat(np.array([[0,1,4,5]]),repeats=test_mask.shape[0], axis=0) , axis=1);
            # valid_f1 = train('heart', 5, multimodel, idx, train_imgs, train_mask, 
            # test_imgs, test_mask, ckpt if config.RESUME else None, param, multilabel=True);
            # parameters.add_results(valid_f1, idx);

            
            
            #(2-2)
            # print('------------- Training Diaphragm ---------------\n');
            # diaphragm_segmentation_model = newtwork_trainer.train('Diaphragm', 1, diaphragm_segmentation_model, idx,
            # train_imgs, train_mask[:,1], test_imgs, test_mask[:,1], load_trained_model=True);

            # #(2-3)
            # print('------------- Training Sternum ---------------\n');
            # sternum_segmentation_model = newtwork_trainer.train('Sternum', 1, sternum_segmentation_model, 
            # idx,  train_imgs, train_mask[:,2], test_imgs, test_mask[:,2], load_trained_model=False, exposure_labels=train_grain_lbl[:,-1]);

            # #(2-3)
            # print('------------- Training Spinous process ---------------\n');
            # spinous_process_segmentation_model = newtwork_trainer.train('Spinous process', 1, spinous_process_segmentation_model, 
            # idx,  train_imgs, train_mask[:,3], test_imgs, test_mask[:,3], load_trained_model=True);

            # cranial_results, caudal_results, tips_results, sternum_results,  quality_results = evaluate_test_data(idx, 
            # [spine_and_ribs_segmentation_model, diaphragm_segmentation_model, sternum_segmentation_model, spinous_process_segmentation_model], 
            # [cranial_classification_model, caudal_classification_model, sternum_classification_model, full_classification_model],
            # test_imgs,
            # test_grain_lbl,
            # test_lbl,
            # None,
            # use_saved_features=False);

            # # total_cranial.append(cranial_results);
            # # total_caudal.append(caudal_results);
            # # total_tips.append(tips_results);
            # # total_sternum.append(sternum_results);
            # # total_quality.append(quality_results);
    
    parameters.get_final_results();
    
    
    

    

        # total_cranial = np.mean(total_cranial, axis = 0);
        # total_caudal = np.mean(total_caudal, axis = 0);
        # total_tips = np.mean(total_tips, axis = 0);
        # total_sternum = np.mean(total_sternum, axis = 0);
        # total_quality = np.mean(total_quality, axis = 0);

        
        # print(('\n'+'%10s'*5)%('Type', 'Precision', 'Recall', 'F1', 'Accuracy'));
        # print(('\n'+'%10s'*1 + '%10f'*4)%('Cranial', total_cranial[0], total_cranial[1], total_cranial[2], total_cranial[3]));
        # print(('\n'+'%10s'*1 + '%10f'*4)%('Caudal', total_caudal[0], total_caudal[1], total_caudal[2], total_caudal[3]));
        # print(('\n'+'%10s'*1 + '%10f'*4)%('Tips', total_tips[0], total_tips[1], total_tips[2], total_tips[3]));
        # print(('\n'+'%10s'*1 + '%10f'*4)%('Sternum', total_sternum[0], total_sternum[1], total_sternum[2], total_sternum[3]));
        # print(('\n'+'%10s'*1 + '%10f'*4)%('Quality', total_quality[0], total_quality[1], total_quality[2], total_quality[3]));
