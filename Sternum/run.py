from copyreg import pickle
from glob import glob
import pickle
from re import L
import cv2
import matplotlib
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from tqdm import tqdm
from Utility import confidence_intervals, draw_missing_spine, get_max_contour, post_process, retarget_img, scale_width, smooth_boundaries
from network_trainer import NetworkTrainer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.inspection import DecisionBoundaryDisplay

TRAIN = False;
def test(root):
    meta_files = glob(f"{root}\\labels\\*.meta");

    image_list = [];
    mask_list = [];
    lbl_list = [];
    all_sternum_area = [];
    mi = 100;
    for idx, m in enumerate(meta_files):
        meta_data = pickle.load(open(m, 'rb'));
        if 'Sternum' in meta_data.keys():
            sternum_mask = cv2.imread(os.path.join(root, 'labels', meta_data['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
            sternum_mask = np.where(sternum_mask > 0, 255, 0).astype("uint8");
            contours = cv2.findContours(sternum_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
            w,h = sternum_mask.shape;
            for c in contours:
                area = cv2.contourArea(c);
                if area < mi:
                    mi = area;
                all_sternum_area.append(area/(w*h));
    print(f'min:{mi}')
    sns.displot(all_sternum_area, kind='kde');
    plt.show();



def preload_dataset():
    labels_file = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    extra_imgs = os.listdir(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\additionalDVVD\\');
    for i in range(len(extra_imgs)):
        extra_imgs[i] = extra_imgs[i][:extra_imgs[i].rfind('.')];
    
    img_lst = list(labels_file['Image']);
    img_lst = list(map(str, img_lst));
    #img_lst.extend(extra_imgs);
    img_lst = np.array(img_lst);
    sternum_lbl = np.array(list(labels_file['Sternum']));
    pos = (sternum_lbl == 2);
    sternum_lbl[pos==False] = 0; 
    sternum_lbl[pos] = 1; 
    sternum_lbl= list(sternum_lbl);
    #sternum_lbl.extend(np.ones(len(extra_imgs)));
    sternum_lbl = np.array(sternum_lbl);

    kfold = StratifiedKFold(shuffle=True, random_state=42);
    fold_cnt = 0;
    for train_id, test_id in kfold.split(img_lst, sternum_lbl):
        train_img, train_lbl, test_img, test_lbl = img_lst[train_id], sternum_lbl[train_id], img_lst[test_id], sternum_lbl[test_id];
        pickle.dump([train_img, train_lbl, test_img, test_lbl], open(f'{fold_cnt}.dmp', 'wb'))
        fold_cnt+=1;

def store_labels():
    #946, 916, 945, 913, 950, 951, 995, 933, 931, 921, 949, 903, 942, 943, 944, 967->0
    neg_list = [946,916,913,950,951,933,949,903,942,943,944,967,951,933, 996, 952, 947, 920, 933, 926,949, 958, 932,922, 909, 910, 984, 934, 940, 986, 978];
    labels_file = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    extra_imgs = os.listdir(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\additionalDVVD\\');
    img_lst = list(labels_file['Image']);
    img_lst = list(map(str, img_lst));
    for i in range(len(extra_imgs)):
        extra_imgs[i] = extra_imgs[i][:extra_imgs[i].rfind('.')];
    #extra_imgs.remove('917');
    img_lst.extend(extra_imgs);

    sternum_lbl = np.array(list(labels_file['Sternum']));
    pos = (sternum_lbl == 2);
    sternum_lbl[pos==False] = 0; 
    sternum_lbl[pos] = 1; 
    sternum_lbl= list(sternum_lbl);
    sternum_lbl.extend(np.ones(len(extra_imgs)));
    for n in neg_list:
        sternum_lbl[img_lst.index(str(n))] = 0;
        #print(sternum_lbl[img_lst.index(str(n))]);


    
    total_data = [];
    total_gt = [];
    
    for index in (range(len(img_lst))):
        gt_lbl = sternum_lbl[index];
        n = img_lst[index];
        if os.path.exists(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{img_lst[index]}.jpeg')) is True:
            radiograph = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{img_lst[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
        elif os.path.exists(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{img_lst[index]}.jpeg')) is True:
            radiograph = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{img_lst[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
        else:
            print(img_lst[index]);
        if os.path.exists(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{img_lst[index]}.png') is False:
            print(n);
            continue;

        sternum_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Sternum\\labels\\{img_lst[index]}.meta', 'rb'));
        sternum_mask_name = sternum_meta['Sternum'][-1];
        sternum_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Sternum\\labels\\{sternum_mask_name}', cv2.IMREAD_GRAYSCALE);
        sternum_mask = np.where(sternum_mask>0, 255, 0).astype("uint8");
        

        spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{img_lst[index]}.meta', 'rb'));
        spine_mask_name = spine_meta['Spine'][-1];
        spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\{spine_mask_name}', cv2.IMREAD_GRAYSCALE);
        spine_mask = np.where(spine_mask>0, 255, 0);

        thorax_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{img_lst[index]}.png', cv2.IMREAD_GRAYSCALE);
        kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
        thorax_mask = cv2.erode(thorax_mask, kernel, iterations=10);

        thorax_mask = cv2.resize(thorax_mask, (radiograph.shape[1], radiograph.shape[0]));

        sternum_mask = ((np.where(thorax_mask>0, 1, 0) * sternum_mask)).astype("uint8");


        spine_mask = smooth_boundaries(spine_mask,10);
        spine_mask = smooth_boundaries(spine_mask,25);
        spine_mask = draw_missing_spine(spine_mask);
        spine_mask = scale_width(spine_mask,1.5);


        residual = np.maximum(np.int32(thorax_mask) - np.int32(spine_mask), np.zeros_like(spine_mask)).astype("uint8");
        # sym_line = get_symmetry_line(cv2.resize(spine_mask.astype("uint8"), (1024, 1024)));
        # residual_mask_left, residual_mask_right = divide_image_symmetry_line(cv2.resize(residual, (1024, 1024)), sym_line);
        # residual_mask_left =  cv2.resize(residual_mask_left, (radiograph.shape[1], radiograph.shape[0]));
        # residual_mask_right =  cv2.resize(residual_mask_right, (radiograph.shape[1], radiograph.shape[0]));
        radiograph = (np.int32(radiograph) * np.where(thorax_mask>0, 1, 0)).astype("uint8");
        radiograph = (np.int32(radiograph) * np.where(spine_mask>0, 0, 1)).astype("uint8");

        ret, residual_mask = retarget_img([sternum_mask, radiograph, spine_mask, thorax_mask], residual);
        sternum_mask = ret[0];
        radiograph = ret[1];
        spine_mask = ret[2];
        thorax_mask = ret[3];


        sternum_mask = cv2.resize(sternum_mask, (1024, 1024));
        sternum_mask = post_process(sternum_mask);
        thorax_mask = cv2.resize(thorax_mask, (1024, 1024));
        radiograph = cv2.resize(radiograph, (1024, 1024));
        spine_mask = cv2.resize(spine_mask.astype("uint8"), (1024, 1024));
        before = np.sum(np.where(sternum_mask> 0, 1, 0));
        sternum_mask = (sternum_mask * np.where(spine_mask>0, 0, 1)).astype("uint8");
        after = np.sum(np.where(sternum_mask> 0, 1, 0));
        #cv2.imshow('t', thorax_mask);
        #cv2.waitKey();

        s = after/(before+1e-6);
        rat = after / np.sum(np.where(thorax_mask>0, 1, 0));
        # if s < 1e-10 and gt_lbl == 1:
        #     print(img_lst[index]);
        total_data.append([s, rat]);
        total_gt.append(gt_lbl);

        

        # cv2.imshow('after', cv2.resize(spine_mask, (512,512)))
        # cv2.imshow('before', cv2.resize(bef, (512,512)));
        # cv2.waitKey();

        # returned_contour = np.zeros_like(residual_mask);
        # lbl = 0;
        # sternum_contours = cv2.findContours(sternum_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
        # for idx,c in enumerate(sternum_contours):
        #     tmp = np.zeros_like(residual_mask);
        #     tmp = cv2.drawContours(tmp, [c], 0, (255,255,255), -1);
        #     area_before = np.sum(tmp)/255;
        #     residual_tmp = ((np.where(residual_mask>0, 1, 0) * np.where(tmp>0, 1, 0))*255).astype("uint8");
        #     area_after = np.sum(residual_tmp)/255;
        #     #cv2.imwrite(f'tmp\\{file_name}_{idx}.png', tmp);
        #     #cv2.imwrite(f'tmp\\{file_name}_resstr_{idx}.png', residual_tmp);
        #     rat = (area_after / area_before);
        #     # b = cv2.addWeighted(residual_tmp, 0.5, residual, 0.5, 0.0);
        #     # cv2.imshow('b', cv2.resize(b, (512,512)));
        #     # cv2.waitKey();
        #     if rat > 0.8:
        #         #print(rat);
        #         lbl = 1;
        #         contours_res,_ = get_max_contour(residual_tmp);
        #         #contours_res= cv2.findContours(residual_tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
        #         rect = cv2.boundingRect(contours_res);
        #         rect = list(rect);
        #         returned_contour = cv2.drawContours(returned_contour, [contours_res], 0, (255, 255, 255), -1);
        
        b = cv2.addWeighted(radiograph, 0.5, sternum_mask, 0.5, 0.0);
        cv2.imwrite(f'tmp\\{img_lst[index]}.png', b);
        # if lbl != gt_lbl:
        #     print(f'{n}\tfrom: {gt_lbl}\tto{lbl}');
        # labels_file.at[index, 'Sternum'] = lbl;
        # labels_file.at[index, 'Image'] = n;
    
    pickle.dump([total_data, total_gt, img_lst] ,open('d1.dmp','wb'))
    # total_data, total_gt, total_img = pickle.load(open('d.dmp','rb'))
    # for ix, i in enumerate(img_lst):
    #     idx = total_img.index(i);
    #     loaded = total_gt[idx];
    #     t = sternum_lbl[ix];

    #     if t != loaded:
    #         print(i);
    # total_gt  = np.array(total_gt, dtype=np.int32);
    # #total_data = np.expand_dims(np.array(total_data), axis=1)
    # #total_data = np.concatenate([total_data, total_data], axis = 1);
    # total_data = np.array(total_data);
    # rs = RobustScaler();
    # total_data = rs.fit_transform(total_data);
    # qda = LogisticRegression();
    # #s = cross_val_score(lr, total_data, total_gt, scoring='f1', n_jobs=-1);
    # qda.fit(total_data, total_gt);
    # DecisionBoundaryDisplay.from_estimator(qda, total_data, cmap=plt.cm.RdBu, alpha=0.8,  eps=0.5);
    # color = ['g','r'];
    # for i in range(len(total_data)):
    #     plt.scatter(total_data[i][0], total_data[i][1], c = color[total_gt[i]]);
    #     plt.text(total_data[i][0], total_data[i][1], img_lst[i]);
    # plt.show();
    #labels_file.to_excel('test1.xlsx', index=False);
        


if __name__ == "__main__":
    preload_dataset()
    #store_labels();

    font = {'family' : 'normal',
        'size'   : 14}

    #matplotlib.rc('font', **font)

    nt = NetworkTrainer();
    kfold = KFold(5);
    train_ids = [];
    test_ids = [];
    total_x = [];
    total_y = [];
    total_train = dict();
    total_test = dict();
    params = [[10,100], [100, 10]];
    best_f1 = 0;
    best_param = None;
    #for hyp in params:
    total_f1 = [];
    total_train_sternum1, total_test_sternum1 = pickle.load(open('data_sternum1.dmp', 'rb'));
    # for curr_fold in range(0,5):
    #     print(f'===============Starting fold: {curr_fold}==================')
    #     img_train,lbl_train, img_test,lbl_test = pickle.load(open(f'{curr_fold}.dmp', 'rb'));

        
    #     f1 = nt.train(curr_fold, [img_train, lbl_train], [img_test, lbl_test]);
        #total_f1.append(f1);
        #nt.train_classifier(curr_fold, [img_train, lbl_train], [img_test, lbl_test]);
    #train_x, train_y, test_x, test_y = nt.store_results(curr_fold, [img_train, img_test, lbl_test], total_test_sternum1);
    #     train_ids.append(np.arange(len(total_x), len(train_x) + len(total_x)));
    #     test_ids.append(np.arange(len(total_x) + len(train_x), len(test_x) + len(total_x) + len(train_x)));
    #     total_x.extend(train_x);
    #     total_x.extend(test_x);
    #     total_y.extend(train_y);
    #     total_y.extend(test_y);
    #     for idx, img in enumerate(img_train):
    #         if img not in total_train.keys():
    #             total_train[img] = [train_x[idx], train_y[idx]];
        
    #     for idx, img in enumerate(img_test):
    #         if img not in total_test.keys():
    #             total_test[img] = [test_x[idx], test_y[idx]];
        
    #     if np.average(total_f1) > best_f1:
    #         best_f1 = np.average(total_f1);
    #         best_param = hyp;
    
    # print(f'Best f1: {best_f1}\tHyp: {best_param}');

    # custom_cv = zip(train_ids, test_ids);
    # #pickle.dump([total_x, total_y, custom_cv], open("data.dmp", "wb"));
    # pickle.dump([total_train, total_test], open("data_sternum1.dmp", "wb"));
    # total_train_sternum1, total_test_sternum1 = pickle.load(open('data_sternum1.dmp', 'rb'));
    total_train_sternum, total_test_sternum = pickle.load(open('data_sternum.dmp', 'rb'));
    # for k in total_train_sternum1.keys():
    #     diff = np.mean(np.array(total_train_sternum1[k][0]) - np.array(total_train_sternum[k][0]));
    #     if diff !=0:
    #         print(i);
    
    # for k in total_test_sternum.keys():
    #     diff = np.mean(np.array(total_test_sternum1[k][0]) - np.array(total_test_sternum[k][0]));
    #     if diff !=0:
    #         print(k);
    data = pickle.load(open('data_sp.dmp', 'rb'));
    total_x_sp, total_y_sp, total_imgs_sp, custom_cv = data[0], data[1], data[2], data[3];
    total_x_sp = np.array(total_x_sp);
    total_y_sp = np.array(total_y_sp, np.int32);
    total_imgs_sp = np.array(total_imgs_sp);

    total_x= np.array(total_x);
    total_y= np.array(total_y);

    
    # param_range = [0.00001, 0.0001, 0.001, 0.01, 0.1];
    # param_grid = [
    #             {'svc__C' : param_range,
    #             'svc__kernel' : ['linear']},
    #             {
    #                 'svc__C': param_range,
    #                 'svc__gamma' : param_range,
    #                 'svc__kernel' : ['rbf']
    #             }
    #         ];
    # scoring = {'f1': 'f1',
    #        'prec': 'precision',
    #        'rec': 'recall'}
    # cv = cross_validate(model, total_x, total_y, n_jobs=-1, cv = custom_cv, scoring='auc');
    fold_cnt = 0;
    fig,ax = plt.subplots(figsize = (6,6));
    tprs = [];
    aucs = [];
    mean_fpr = np.linspace(0,1,100);
    total_pred = [];
    total_gt = [];
    avg = [];
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
        

        model = Pipeline([('scalar', RobustScaler()), ('svc', SVC(C= 0.01, kernel='linear'))]);
        model.fit(train_x_sternum, train_y_sternum);
        pred = model.predict(test_x_sternum);

        viz = RocCurveDisplay.from_estimator(
            model,
            test_x_sternum,
            test_y_sternum,
            name=f'ROC fold{fold_cnt}',
            alpha = 0.3,
            lw = 1,
            ax = ax
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr);
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
        total_pred.extend(pred);
        total_gt.extend(test_y_sternum);

        prec, rec, f1, _ = precision_recall_fscore_support(test_y_sternum, pred, average='binary');
        avg.append([prec, rec, f1]);

        fold_cnt+=1;
    
    confidence_intervals(avg);
    confidence_intervals(aucs);
    # cm = confusion_matrix(total_gt, total_pred);
    # a = cm[0][0];
    # cm[0][0] = cm[1][1];
    # cm[1][1] = a;
    # cm[0][1], cm[1][0] = cm[1][0], cm[0][1];
    ##prec, rec, f1, _ = precision_recall_fscore_support(total_gt, total_pred, average='binary');
    #print(f'prec: {prec}\trec: {rec}\tf1: {f1}');

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
    disp.ax_.set_title('SSCM Confusion matrix')

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
    ax.set_title('SSCM ROC curves')
    ax.legend(loc='best');
    plt.show();



    #ConfusionMatrixDisplay.from_estimator(cv)
    #print(cv);

       
        

