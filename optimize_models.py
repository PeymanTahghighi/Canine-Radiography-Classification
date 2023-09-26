
from copy import deepcopy
import os
import cv2
import numpy as np
from sklearn.metrics import RocCurveDisplay, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from deep_learning.network import Unet
import pickle
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import chain, combinations
import pandas as pd
from sklearn.utils._testing import ignore_warnings
from itertools import chain, combinations
from utility import extract_cranial_features
import config
from utility import confidence_intervals, draw_missing_spine, extract_caudal_features, extract_sp_feature, extract_sternum_features, postprocess_sternum, retarget_img, scale_width, smooth_boundaries
from utility import get_symmetry_line, divide_image_symmetry_line

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def optimize_sternum_model(folds):
    best_class_thresh = 0;
    best_pix_thresh = 0;
    thresholds = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9,0.95];
    best_f1 = 0;
    df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    sternum_list = list(df['Sternum']);
    img_list = list(map(str, list(df['Image'])));

  
    for idx,f in enumerate(folds):
        train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
        caudal_features, \
        tips_features,\
        sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11];
        
        spine_model = Unet(3);
        sternum_model = Unet(1);
        spine_model.load_state_dict(pickle.load(open(f'results\\{idx}\\spine and ribs.pt', 'rb')));
        sternum_model.load_state_dict(pickle.load(open(f'results\\{idx}\\sternum.pt', 'rb')));
        spine_model = spine_model.to(config.DEVICE);
        sternum_model = sternum_model.to(config.DEVICE);

        best_threshold_sternum = 0;
        best_threshold_num= 0;
        for thresh in thresholds:
            total_X = [];
            total_Y = [];
            train_fold_indices = [];
            test_fold_indices = [];
            data_idx = 0;
            pred_dict = dict();
            #test
            total_gt = [];
            total_pred = [];
            avg = [];
            thresh_num_list = [0,5,10,20,25,30];
            best_thresh_num = -1;
            best_thresh_val = 0;
            
            for thresh_num in thresh_num_list:
                for index, t in (enumerate(test_imgs)):
                    if t not in pred_dict.keys():
                        radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{t}.jpeg'),cv2.IMREAD_GRAYSCALE);
                        
                        full_body_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{t}.png', 
                        cv2.IMREAD_GRAYSCALE);
                        kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
                        full_body_mask = cv2.erode(full_body_mask, kernel, iterations=10);
                        full_body_mask = cv2.resize(full_body_mask, (radiograph_image.shape[1], radiograph_image.shape[0]));
                        radiograph_image = ((np.where(full_body_mask>0, 1, 0) * radiograph_image)).astype("uint8");

                        spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{t}.meta', 'rb'));
                        spine_mask_name = spine_meta['Spine'][-1];
                        spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\{spine_mask_name}', cv2.IMREAD_GRAYSCALE);
                        spine_mask = np.where(spine_mask>0, 255, 0);
                        
                        spine_mask = smooth_boundaries(spine_mask,10);
                        spine_mask = smooth_boundaries(spine_mask,25);
                        spine_mask = draw_missing_spine(spine_mask);
                        spine_mask = scale_width(spine_mask,2);

                        radiograph_image = (np.int32(radiograph_image) * np.where(spine_mask>1, 0, 1)).astype("uint8");
                        ret, full_body_mask = retarget_img([radiograph_image], full_body_mask);
                        radiograph_image = deepcopy(ret[0]);

                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                        radiograph_image = clahe.apply(radiograph_image);
                        radiograph_image = np.expand_dims(radiograph_image, axis=2);
                        radiograph_image = np.repeat(radiograph_image, 3,axis=2);
                        transformed = config.valid_transforms(image = radiograph_image);
                        radiograph_image = transformed["image"];
                        radiograph_image = radiograph_image.to(config.DEVICE);

                        #sternum
                        sternum = sternum_model(radiograph_image.unsqueeze(dim=0));
                        sternum = torch.sigmoid(sternum)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
                        sternum = sternum > 0.95;
                        sternum = np.uint8(sternum)*255;
                        #----------------------------------------------------

                        pred_dict[t] = sternum;
                    else:
                        sternum = pred_dict[t];

                    total_pixels = np.sum(sternum)/255;
                    if total_pixels> 0:
                        pred = 1;
                    else:
                        pred = 0;
                        
                    
                    lbl = sternum_list[img_list.index(t)];
                    total_gt.append(1 if lbl == 2 else 0);
                    total_pred.append(pred);
                    lbl = 1 if lbl ==2 else  0;
                    if pred != lbl:
                        print(f"{t}: pred: {pred}\tlbl: {lbl}");

                    
                    cv2.imwrite(f'tmp\\{t}_sternum.png', sternum);
                    cv2.imwrite(f'tmp\\{t}_rad.png', ret[0]);
                prec, rec, f1, _ = precision_recall_fscore_support(total_gt, total_pred, average='binary');
                if f1 > best_thresh_val:
                    best_thresh_val = f1;
                    best_thresh_num = thresh_num;

            if best_thresh_val > best_threshold_sternum:
                best_threshold_sternum = best_thresh_val;
                best_threshold_num = thresh;
        
        print(f'best: {best_threshold_sternum}');

    
    print(f'best_f1: {best_f1}\t param: {best_class_thresh}\t {best_pix_thresh}');

def optimize_cranial_model(folds, df, rebuild_features = False):
    """Optimize Cranial Classification Model (CRCM) and display results.
        Tahghighi, Peyman, et al. Machine learning can appropriately classify the collimation of ventrodorsal and dorsoventral thoracic radiographic images of dogs and cats.
        American Journal of Veterinary Research 1.aop (2023): 1-8.
        https://doi.org/10.2460/ajvr.23.03.0062
    
    :param folds: each fold data which should include train/test images and labels
    :param df: data frame for ground truth labels
    :param rebuild_features: Indicated wether to build features or use already saved ones
    """
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];
    #all_cranials = [];
    cranial_list = list(df['Cranial']);
    img_list = list(map(str, list(df['Image'])));

    if rebuild_features:
        for idx,f in enumerate(folds):
            train_imgs,train_mask, test_imgs, test_mask= f[0], f[1], f[2], f[3]

            train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
            test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));
            
            #train
            for i in tqdm(range(len(train_imgs))):
                index = img_list.index(train_imgs[i]);
                spine_and_rib_mask_meta = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\{train_imgs[i]}.meta', 'rb'));

                spine_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
                ribs_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, spine_and_rib_mask_meta['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
                spine_mask = draw_missing_spine(spine_mask);
                
                feat = extract_cranial_features(spine_mask, ribs_mask);
                total_X.append(feat);
                lbl_in_list = cranial_list[index];
                total_Y.append(lbl_in_list);
            # #--------------------------------------------------------------------------------------------------------

            #test
            cranials = [];
            for index,t in tqdm(enumerate(test_imgs)):
                
                indx = img_list.index(t);
                spine_mask = cv2.imread(f'results\\{idx}\\outputs\\{t}_spine_orig.png', cv2.IMREAD_GRAYSCALE);
                ribs_mask = cv2.imread(f'results\\{idx}\\outputs\\{t}_ribs_orig.png', cv2.IMREAD_GRAYSCALE);
                spine_mask = draw_missing_spine(spine_mask);

                feat = extract_cranial_features(spine_mask, ribs_mask);
                #cranials.append([t,c]);
                total_X.append(feat);
                lbl_in_list = cranial_list[indx];

                total_Y.append(lbl_in_list);
            #----------------------------------------------------------------------------------------
            #all_cranials.append(cranials);

        custom_cv = zip(train_fold_indices, test_fold_indices);
        pickle.dump([total_X, total_Y, custom_cv], open('data_cranial1.dmp', 'wb'));

    data = pickle.load(open('data_cranial1.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];
    total_x = np.array(total_x);
    total_y = np.array(total_y, np.int32);
    p = (total_y == 1);
    total_y[p==True] = 0;
    total_y[p==False] = 1;
    fig, ax = plt.subplots(1,2);
    total_x = np.expand_dims(total_x, axis = -1);


    # idx = 0;
    # for train_id, test_id in custom_cv:
    #     f = folds[idx];
    #     train_imgs,train_mask, test_imgs, test_mask = f[0], f[1], f[2], f[3]
    #     train_x, train_y, test_x, test_y = total_x[train_id], total_y[train_id], total_x[test_id], total_y[test_id];
    #     for i in range(len(train_x)):
    #         ax[0].scatter(train_x[i,0], 0, color='r' if train_y[i] == 0 else 'b');
    #         ax[0].text(train_x[i,0], 0, train_imgs[i])
        
    #     for i in range(len(test_x)):
    #         ax[1].scatter(test_x[i,0], 0, color='r' if test_y[i] == 0 else 'b');
    #         ax[1].text(test_x[i,0], 0, test_imgs[i])
    #     plt.show();
    #     idx = idx+1;
    

    param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1];

    param_grid = [
                {'svc__C' : param_range,
                'svc__kernel' : ['linear']},
                {
                    'svc__C': param_range,
                    'svc__gamma' : param_range,
                    'svc__kernel' : ['rbf']
                }
            ];

    pipe = Pipeline([('scalar',RobustScaler()), ('svc', SVC())]); 
    tmp_cv = deepcopy(custom_cv);
    svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = tmp_cv);

    svm = svm.fit(total_x, total_y);

    print(f'Best score of {svm.best_score_} achieved with: {svm.best_params_}');

    fig,ax = plt.subplots(figsize = (6,6));
    avg = [];
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    fold_cnt = 0;
    total_pred = [];
    total_gt = [];
    for train_id, test_id in custom_cv:
        train_x, train_y, test_x, test_y = total_x[train_id], total_y[train_id], total_x[test_id], total_y[test_id];

        model = make_pipeline(RobustScaler(),
        SVC(kernel='linear', C=0.1));

        model.fit(train_x, train_y);
        pickle.dump(model, open(os.path.join('results', str(fold_cnt), 'cranial.mlm'), 'wb'));
        pred = model.predict(test_x);
        # for i,p in enumerate(pred):
        #     if p != test_y[i]:
        #         cv2.imwrite(f'tmp\\{all_cranials[fold_cnt][i][0]}_{p}_{test_y[i]}.png', all_cranials[fold_cnt][i][1]);

        viz = RocCurveDisplay.from_estimator(
            model,
            test_x,
            test_y,
            name = f'ROC fold {fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )

        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        prec, rec, f1, _ = precision_recall_fscore_support(test_y, pred, average='binary');
        total_gt.extend(test_y);
        total_pred.extend(pred);
        avg.append([prec, rec, f1]);

        fold_cnt += 1;
    print(avg);
    std = np.std(avg, axis = 0)*100;
    avg = np.mean(avg, axis = 0)*100;
    print(f'prec: {avg[0]}\trecall: {avg[1]}\tf1: {avg[2]}');
    cm = confusion_matrix(total_gt, total_pred);
    a = cm[0][0];
    cm[0][0] = cm[1][1];
    cm[1][1] = a;
    cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    disp = ConfusionMatrixDisplay.from_predictions(total_gt,total_pred, display_labels=['Accept', 'Reject'], cmap=plt.cm.Blues);
    disp.ax_.set_title('Confusion matrix for cranial inclusion classification')


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
    ax.set_title('ROC curves for cranial inclusion classification')
    ax.legend(loc='best');
    plt.show();

def optimize_caudal_model(folds, df, rebuild_features):
    """Optimize Caudal Classification Model (CACM) and display results.
    Tahghighi, Peyman, et al. "Machine learning can appropriately classify the collimation of ventrodorsal and dorsoventral thoracic radiographic images of dogs and cats.
    " American Journal of Veterinary Research 1.aop (2023): 1-8.
    https://doi.org/10.2460/ajvr.23.03.0062
    
    :param folds: each fold data which should include train/test images and labels
    :param df: data frame for ground truth labels
    :param rebuild_features: Indicated wether to build features or use already saved ones
    """
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];

    caudal_list = list(df['Caudal']);
    img_list = list(map(str, list(df['Image'])));
    total_imgs = [];
    
    if rebuild_features:
        for idx,f in enumerate(folds):
            
            train_imgs,train_mask, test_imgs, test_mask = f[0], f[1], f[2], f[3]

            train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
            test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));
            
            # #train
            for i in tqdm(range(len(train_imgs))):
                index = img_list.index(train_imgs[i]);
                abdomen_mask_path = pickle.load(open(f'{config.D_PROJECT_ROOT}\\{train_imgs[i]}.meta', 'rb'));
                heart_mask_path = pickle.load(open(f'{config.H_PROJECT_ROOT}\\{train_imgs[i]}.meta', 'rb'));
                ribs_mask_path = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\{train_imgs[i]}.meta', 'rb'));
                radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{train_imgs[i]}.jpeg'),cv2.IMREAD_GRAYSCALE);

                abdomen_mask = cv2.imread(os.path.join(config.D_PROJECT_ROOT,  abdomen_mask_path['Diaphragm'][2]), cv2.IMREAD_GRAYSCALE);
                heart_mask = cv2.imread(os.path.join(config.H_PROJECT_ROOT,  heart_mask_path['Heart'][2]), cv2.IMREAD_GRAYSCALE);
                ribs_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT,  ribs_mask_path['Ribs'][2]), cv2.IMREAD_GRAYSCALE);

                thorax = cv2.imread(f'results\\train_data\\{train_imgs[i]}.png', cv2.IMREAD_GRAYSCALE);

                feat = extract_caudal_features(abdomen_mask, thorax, heart_mask, ribs_mask);
                
                total_X.append(feat);
                lbl_in_list = caudal_list[index];
                total_Y.append(lbl_in_list);
                total_imgs.append(train_imgs[i]);
            # #--------------------------------------------------------------------------------------------------------

            #test
            for index,t in tqdm(enumerate(test_imgs)):
                indx = img_list.index(t);

                radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{t}.jpeg'),cv2.IMREAD_GRAYSCALE);
                abdomen_mask = cv2.imread(f'results\\{idx}\\outputs\\{t}_diaph.png', cv2.IMREAD_GRAYSCALE)

                heart_mask_path = pickle.load(open(f'{config.H_PROJECT_ROOT}\\{t}.meta', 'rb'));
                heart_mask = cv2.imread(os.path.join(config.H_PROJECT_ROOT, heart_mask_path['Heart'][2]), cv2.IMREAD_GRAYSCALE);

                thorax = cv2.imread(f'results\\{idx}\\outputs\\{t}_thorax.png', cv2.IMREAD_GRAYSCALE);
                ribs = cv2.imread(f'results\\{idx}\\outputs\\{t}_ribs_orig.png', cv2.IMREAD_GRAYSCALE);

                
                feat = extract_caudal_features(abdomen_mask, thorax, heart_mask, ribs);

                total_X.append(feat);
                lbl_in_list = caudal_list[indx];
                total_Y.append(lbl_in_list);
                total_imgs.append(t);
            #----------------------------------------------------------------------------------------

        custom_cv = zip(train_fold_indices, test_fold_indices);
        pickle.dump([total_X, total_Y, total_imgs, custom_cv], open('data_caudal.dmp', 'wb'));
    

    data = pickle.load(open('data_caudal.dmp', 'rb'));
    total_x, total_y, total_imgs,  custom_cv = data[0], data[1],  data[2], data[3];
    total_x = np.array(total_x);
    total_y = np.array(total_y, np.int32);
    total_y = np.array(total_y);
    p = (total_y == 1);
    total_y[p==True] = 0;
    total_y[p==False] = 1;

    total_imgs = np.array(total_imgs);
    avg = [];
    fold_cnt = 0;
    
    fig,ax = plt.subplots(figsize = (6,6));
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    total_pred = [];
    total_gt = [];
    for train_id, test_id in custom_cv:
        train_x, train_y, test_x, test_y, tes_imgs = total_x[train_id], total_y[train_id], total_x[test_id], total_y[test_id], total_imgs[test_id];

        model = make_pipeline(RobustScaler(),
        MLPClassifier(max_iter=500, activation= 'relu', 
        alpha=0.001, hidden_layer_sizes=60, learning_rate='constant',
        solver='adam'));
        model.fit(train_x, train_y);
        pickle.dump(model, open(os.path.join('results', str(fold_cnt), 'caudal.mlm'), 'wb'));
        pred = model.predict(test_x);

        viz = RocCurveDisplay.from_estimator(
            model,
            test_x,
            test_y,
            name=f'ROC fold{fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )

        for i in range(len(test_y)):
            if test_y[i]!=pred[i]:
                print(f'{tes_imgs[i]}\t pred: {pred[i]}\tTrue: {test_y[i]}');
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr);
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        
        
        prec, rec, f1, _ = precision_recall_fscore_support(test_y, pred, average='binary');
        avg.append([prec, rec, f1]);
        total_gt.extend(test_y);
        total_pred.extend(pred);

        fold_cnt += 1;
    
    std = np.std(avg, axis = 0)*100;
    avg = np.mean(avg, axis = 0)*100;
    print(f'prec: {avg[0]}\trecall: {avg[1]}\tf1: {avg[2]}');

    cm = confusion_matrix(total_gt, total_pred);
    a = cm[0][0];
    cm[0][0] = cm[1][1];
    cm[1][1] = a;
    cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    disp = ConfusionMatrixDisplay.from_predictions(total_gt,total_pred, display_labels=['Accept', 'Reject'], cmap=plt.cm.Blues);
    disp.ax_.set_title('Confusion matrix for caudal inclusion classification')

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
    ax.set_title('ROC curves for caudal inclusion classification')
    ax.legend(loc='best');
    plt.show();

def optimize_full_model(folds):
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];

    for idx,f in enumerate(folds):
        train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
        caudal_features, \
        symmetry_features, \
        tips_features,\
        sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11], f[12];

        train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
        test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));

        #train
        for i in tqdm(range(len(train_imgs))):    
            total_X.append([*cranial_features[i], *caudal_features[i], tips_features[i], sternum_features[i][1]]);
            total_Y.append(train_lbl[i]);
        #--------------------------------------------------------------------------------------------------------

        #test
        for i in tqdm(range(len(test_imgs))):
            cranial_feat = pickle.load(open(f'results\\{idx}\\outputs\\{test_imgs[i]}_cranial.feat', 'rb'));
            caudal_feat = pickle.load(open(f'results\\{idx}\\outputs\\{test_imgs[i]}_caudal.feat', 'rb'));
            sp_feat = pickle.load(open(f'results\\{idx}\\outputs\\{test_imgs[i]}_sp.feat', 'rb'));
            sternum_feat = pickle.load(open(f'results\\{idx}\\outputs\\{test_imgs[i]}_sternum.feat', 'rb'));
            total_X.append([*cranial_feat, *caudal_feat, sp_feat, sternum_feat[1]]);
            total_Y.append(test_lbl[i]);
        #----------------------------------------------------------------------------------------

    custom_cv = zip(train_fold_indices, test_fold_indices);
    pickle.dump([total_X, total_Y, custom_cv], open('data.dmp', 'wb'));

    data = pickle.load(open('data.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];

    # c_transform = ColumnTransformer([
    #     ('onehot', OneHotEncoder(), [3]),
    #     ('nothing', 'passthrough', [0,1,2])
    #     ]);

    total_x = np.array(total_x);
    total_y = np.array(total_y, np.int32);
    
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

    param_grid = [
        {
            'mlp__hidden_layer_sizes': [10,20,30,40,50,60,70,80,90,100],
            'mlp__activation':['relu', 'tanh'],
            'mlp__solver':['sgd','adam'],
            'mlp__learning_rate': ['constant','adaptive','invscaling'],
            'mlp__alpha': param_range

        }
    ];

    pipe = Pipeline([('scalar',RobustScaler()), ('mlp', MLPClassifier())]);

    svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = custom_cv);

    svm = svm.fit(total_x, total_y);

    print(f'Best score of {svm.best_score_} achieved with: {svm.best_params_}');

def optimize_sp_model(folds, df, rebuild_features = False):
    """Optimize Spinous Process Classification Model (SPCM) and display results
    
    :param folds: each fold data which should include train/test images and labels
    :param df: data frame for ground truth labels
    :param rebuild_features: Indicated wether to build features or use already saved ones
    """

    sp_list = list(df['Tips outside of spine']);
    img_list = list(map(str, list(df['Image'])));
    best_f1 = 0;
    best_t = None;
    train_features = dict();
    train_fold_indices = [];
    test_fold_indices = [];

    total_X = [];
    total_Y = [];
    total_imgs = [];

    reoptimize = True;

    if rebuild_features is True:
        for idx,f in enumerate(folds):
            train_imgs,train_mask,test_imgs, test_mask = f[0], f[1], f[2], f[3]
            model_rs = Unet(3);
            model_rs.load_state_dict(pickle.load(open(f'results\\{idx}\\spine and ribs.pt', 'rb')));
            model_rs = model_rs.to(config.DEVICE)
            model_sp = Unet(1);
            model_sp.load_state_dict(pickle.load(open(f'results\\{idx}\\Spinous process.pt', 'rb')));
            model_sp = model_sp.to(config.DEVICE)

            train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
            test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));
            for t in tqdm(range(len(train_imgs))):
                if train_imgs[t] not in train_features.keys():
                    spine_and_rib_mask_meta = pickle.load(open(f'{config.SR_PROJECT_ROOT}\\{train_imgs[t]}.meta', 'rb'));

                    spine_mask = cv2.imread(os.path.join(config.SR_PROJECT_ROOT, spine_and_rib_mask_meta['Spine'][2]), cv2.IMREAD_GRAYSCALE);
                    spinous_process_meta = pickle.load(open(f'{config.SP_PROJECT_ROOT}\\{train_imgs[t]}.meta', 'rb'));
                    spinous_process = cv2.imread(os.path.join(config.SP_PROJECT_ROOT, spinous_process_meta['Spinous process'][2]), cv2.IMREAD_GRAYSCALE);
                    whole_thorax = cv2.imread(f'results\\train_data\\{train_imgs[t]}.png', cv2.IMREAD_GRAYSCALE);

                    sp_features = extract_sp_feature(spine_mask, spinous_process, whole_thorax, train_imgs[t], False);
                    
                    
                    train_features[train_imgs[t]] = sp_features;
                else:
                    sp_features = train_features[train_imgs[t]];

                total_X.append(sp_features);
                lbl = sp_list[img_list.index(train_imgs[t])];
                total_Y.append(0 if lbl == 'No' else 1);
                total_imgs.append(train_imgs[t]);


            for t in tqdm(range(len(test_imgs))):

            
                # radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{test_imgs[t]}.jpeg'),cv2.IMREAD_GRAYSCALE);
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                # radiograph_image = clahe.apply(radiograph_image);
                # radiograph_image = np.expand_dims(radiograph_image, axis=2);
                # radiograph_image = np.repeat(radiograph_image, 3,axis=2);


                # transformed = config.valid_transforms(image = radiograph_image);
                # radiograph_image = transformed["image"];
                # radiograph_image = radiograph_image.to(config.DEVICE);
                
                spine = cv2.imread(f'results\\{idx}\\outputs\\{test_imgs[t]}_spine_orig.png', cv2.IMREAD_GRAYSCALE);
                whole_thorax = cv2.imread(f'results\\{idx}\\outputs\\{test_imgs[t]}_thorax.png', cv2.IMREAD_GRAYSCALE);

                # spinous_process = model_sp(radiograph_image.unsqueeze(dim=0));
                # spinous_process = torch.sigmoid(spinous_process)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
                # spinous_process = spinous_process > 0.6;
                # spinous_process = np.uint8(spinous_process)*255;
                spinous_process = cv2.imread(f'results\\{idx}\\outputs\\{test_imgs[t]}_spinous_prcess.png', cv2.IMREAD_GRAYSCALE);
            
                sp_features = extract_sp_feature(spine, spinous_process, whole_thorax, test_imgs[t], True);
                
                total_X.append(sp_features);
                lbl = sp_list[img_list.index(test_imgs[t])];
                total_Y.append(0 if lbl == 'No' else 1);
                total_imgs.append(test_imgs[t]);

            #folds_data.append(fold_data);
        
        custom_cv = zip(train_fold_indices, test_fold_indices);
        pickle.dump([total_X, total_Y, total_imgs, custom_cv], open('data_sp1.dmp', 'wb'));

    data = pickle.load(open('data_sp1.dmp', 'rb'));
    total_x, total_y, total_imgs, custom_cv = data[0], data[1], data[2], data[3];
    total_x = np.array(total_x);
    total_y = np.array(total_y);
    total_imgs = np.array(total_imgs);

    
    if reoptimize is True:
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

        param_grid = [
            {'svc__C' : param_range,
            'svc__kernel' : ['linear']},
            {
                'svc__C': param_range,
                'svc__gamma' : param_range,
                'svc__kernel' : ['rbf']
            }
        ];

        pipe = Pipeline([('scalar',RobustScaler()), ('svc', SVC())]);

        svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = deepcopy(custom_cv));

        svm = svm.fit(total_x, total_y);

        print(f'best score: {best_f1} with t: {best_t}')

    fig,ax = plt.subplots(figsize=(6,6));
    avg = [];
    fold_cnt = 0; 
    total_gt = [];
    total_pred = [];
    mean_fpr = np.linspace(0,1,100);
    tprs = [];
    aucs = [];

    for train_id, test_id in custom_cv:
        train_x, train_y, test_x, test_y, test_imgs = total_x[train_id], total_y[train_id], total_x[test_id], total_y[test_id], total_imgs[test_id];


        model = make_pipeline(RobustScaler(),
        SVC(kernel="rbf", gamma=0.1, C=10.0));
        model.fit(train_x, train_y);
        pickle.dump(model, open(os.path.join('results', str(fold_cnt), 'sp.mlm'), 'wb'));
        pred = model.predict(test_x);
        for i in range(len(pred)):
            if pred[i]!=test_y[i]:
                print(f'{test_imgs[i]}\tpred: {pred[i]}\tGT: {test_y[i]}')
        total_gt.extend(test_y);
        total_pred.extend(pred);

        roc_curve = RocCurveDisplay.from_estimator(
            model,
            test_x,
            test_y,
            lw = 1,
            alpha = 0.3,
            name = f'ROC fold {fold_cnt}',
            ax = ax   
        )

        interp_tpr = np.interp(mean_fpr, roc_curve.fpr, roc_curve.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_curve.roc_auc)

        prec, rec, f1, _ = precision_recall_fscore_support(test_y, pred, average='binary');
        avg.append([prec, rec, f1]);

        fold_cnt += 1;
    
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 20

    plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    confidence_intervals(avg);
    confidence_intervals(aucs);
    prec, rec, f1, _ = precision_recall_fscore_support(total_gt, total_pred, average='binary');
    disp = ConfusionMatrixDisplay.from_predictions(total_gt, total_pred, display_labels=['Inside spine', 'Outside of spine'],  cmap=plt.cm.Blues, colorbar=False);
    disp.ax_.set_title('SPCM Confusion matrix')
    disp.plot();

    mean_tpr = np.mean(tprs, axis = 0);
    mean_tpr[-1] = 1;
    mean_auc = np.mean(aucs);
    std_auc = np.std(aucs);
    ax.plot(mean_fpr, mean_tpr, color='b',
    label = r'Mean Roc (AUC = %0.2f $\pm$ %0.2f)' %(mean_auc, std_auc), lw = 2, alpha = 0.8);

    std_tpr = np.std(tprs);
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1);
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0);
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color='gray',
        alpha = 0.2,
        label = r"$\pm$ %0.2f std. dev." %(std_auc)
    )

    #ax.plot(np.arange(0.0,1.05,0.05), np.arange(0.0,1.05,0.05), '--', color='green', )
    ax.set_xlabel('FPR');
    ax.set_ylabel('TPR');
    ax.set_title('SPCM ROC curves')
    ax.legend(loc='best');
    plt.show();