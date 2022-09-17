from copy import deepcopy
import os
from tkinter import Label
import cv2
import matplotlib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from utility import extract_sternum_features, postprocess_sternum, scale_width
from utility import remove_blobs_spine
import config
from deep_learning.network import Unet
import pickle
import torch
from glob import glob
from utils import create_folder
from tqdm import tgrange, tqdm
import matplotlib.pyplot as plt

def optimize_sternum_model(folds):

    # create_folder('optimization_cache');
    

    best_class_thresh = 0;
    best_pix_thresh = 0;
    thresholds = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9,0.95];
    best_f1 = 0;
    for thresh in thresholds:
        total_X = [];
        total_Y = [];
        train_fold_indices = [];
        test_fold_indices = [];
        data_idx = 0;
        #for idx,f in enumerate(folds):
            
        #     train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
        #     caudal_features, \
        #     symmetry_features, \
        #     sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11];
            
        #     spine_model = Unet(3);
        #     sternum_model = Unet(1);
        #     spine_model.load_state_dict(pickle.load(open(f'results\\{idx}\\spine and ribs.pt', 'rb')));
        #     sternum_model.load_state_dict(pickle.load(open(f'results\\{idx}\\sternum.pt', 'rb')));
        #     spine_model = spine_model.to(config.DEVICE);
        #     sternum_model = sternum_model.to(config.DEVICE);
        #     train_curr_fold_indices = [];
        #     test_curr_fold_indices = [];

        #     #train
        #     for i in tqdm(range(len(train_mask))):
        #         if train_grain_lbl[i][3] == 'Yes' or  train_grain_lbl[i][3] == 'Mid' or  train_grain_lbl[i][3] == 'No':
        #             sternum_mask = pickle.load(open(train_mask[i][2],'rb'))*255;
        #             spine_mask = pickle.load(open(train_mask[i][0],'rb'));
        #             spine_mask = spine_mask==2
        #             spine_mask = np.uint8(spine_mask)*255;
        #             spine_mask = remove_blobs_spine(spine_mask);
        #             spine_mask = scale_width(spine_mask, 3);
        #             total_X.append(extract_sternum_features(sternum_mask, spine_mask));
        #             total_Y.append(train_grain_lbl[i][3]);
        #             train_curr_fold_indices.append(data_idx);
        #             data_idx += 1;
        #     #--------------------------------------------------------------------------------------------------------

        #     #test
        #     for index,t in tqdm(enumerate(test_imgs)):
        #         if test_grain_lbl[index][3] == 'Yes' or  test_grain_lbl[index][3] == 'Mid' or  test_grain_lbl[index][3] == 'No':
        #             radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{t}.jpeg'),cv2.IMREAD_GRAYSCALE);
        #             clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #             radiograph_image = clahe.apply(radiograph_image);
        #             radiograph_image = np.expand_dims(radiograph_image, axis=2);
        #             radiograph_image = np.repeat(radiograph_image, 3,axis=2);


        #             transformed = config.valid_transforms(image = radiograph_image);
        #             radiograph_image = transformed["image"];
        #             radiograph_image = radiograph_image.to(config.DEVICE);
                    
        #             #spine and ribs
        #             out = spine_model(radiograph_image.unsqueeze(dim=0));
        #             out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
        #             out = np.argmax(out,axis = 2);

        #             spine = (out == 2).astype("uint8")*255;

        #             spine = remove_blobs_spine(spine).astype("uint8");
        #             spine = scale_width(spine, 3);
        #             #----------------------------------------------------

        #             #sternum
        #             sternum = sternum_model(radiograph_image.unsqueeze(dim=0));
        #             sternum = torch.sigmoid(sternum)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
        #             sternum = sternum > thresh;
        #             sternum = np.uint8(sternum)*255;
        #             sternum = postprocess_sternum(sternum);
        #             #----------------------------------------------------

        #             total_X.append(extract_sternum_features(sternum, spine));
        #             total_Y.append(test_grain_lbl[index][3]);

        #             test_curr_fold_indices.append(data_idx);
        #             data_idx += 1;
                
                
        #     train_fold_indices.append(train_curr_fold_indices);
        #     test_fold_indices.append(test_curr_fold_indices);
        #     #train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
        #     #test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));

        # #     #----------------------------------------------------------------------------------------
        
        #custom_cv = zip(train_fold_indices, test_fold_indices);
        #pickle.dump([total_X,total_Y, custom_cv], open('data.dmp', 'wb'));

        data = pickle.load(open('data.dmp', 'rb'));
        total_x, total_y, custom_cv = data[0], data[1], data[2];
        total_y =  np.array(total_y);
        total_x =  np.array(total_x);
        h,w = total_x.shape;
        total_y[total_y=='No'] = 0;
        total_y[total_y=='Mid'] = 1;
        total_y[total_y=='Yes'] = 1;
        total_y = np.array(total_y, np.int32)
        fig, ax = plt.subplots(1,2);
        # for tr,te in custom_cv:
        #     train_x = total_x[tr];
        #     train_y = total_y[tr];

        #     test_x = total_x[te];
        #     test_y = total_y[te];

        #     train_x = np.array(train_x, dtype=np.int32);
        #     colors = ['r', 'g', 'b']
        #     for idx,t in enumerate(train_x):

        #         ax[0].scatter(t[0], t[1], c = colors[train_y[idx]]);
            
        #     for idx,t in enumerate(test_x):

        #         ax[1].scatter(t[0], t[1], c = colors[test_y[idx]]);
            
        #     #plt.plot(np.arange(0,25e+4)*0.05);
        #     plt.show();
        #     plt.waitforbuttonpress();
        #     plt.clf();




        
        tranges = [0,10,20,30];

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

        pipe = Pipeline([('scalar',RobustScaler()), ('svc', SVC(class_weight='balanced'))]);

        svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = custom_cv);

        svm = svm.fit(total_x, total_y);

        print(f'Best score of {svm.best_score_} achieved with: {svm.best_params_}');
        
        # for t in tranges:
        #     total_prec = [];
        #     total_rec = [];
        #     total_f1 = [];
        #     total_cm = [];
        #     tmp_cv = deepcopy(custom_cv)
        #     for train_data, test_data in tmp_cv:
        #         #train
        #         train_X = total_x[train_data];
        #         train_y = total_y[train_data];
        #         train_y_no = np.where(train_y==0)[0];
        #         train_y = np.delete(train_y, train_y_no);
        #         train_X = np.delete(train_X, train_y_no, axis = 0);

        #         train_lbl_mid = train_y==1;
        #         train_y[train_lbl_mid] = 0;
        #         train_y[train_lbl_mid==False] = 1;

        #         params = {'svc__C': 1.0, 'svc__gamma': 0.1, 'svc__kernel': 'rbf'}
        #         model = make_pipeline(RobustScaler(),
        #                 SVC(class_weight='balanced', C=params['svc__C'], gamma=params['svc__gamma'], kernel = params['svc__kernel']));
        #         train_X = list(train_X);
        #         train_X = np.array(train_X);
        #         model.fit(train_X, np.array(train_y,np.int32));
        #         #----------------------------------------

        #         pred = [];
        #         labels = [];
        #         for idx, td in enumerate(test_data):
        #             feat = total_x[td];
        #             gt = total_y[td];
        #             # if gt == 1 or gt == 2:
        #             #     gt=1;
        #             # else:
        #             #     gt = 0;

        #             labels.append(gt);

        #             if feat[0] > t:
        #                 out_sternum = model.predict(np.array(feat).reshape(1,-1));
        #                 if out_sternum[0] == 1:
        #                     sternum_lbl = 2
        #                 else:
        #                     sternum_lbl = 1
        #                 #sternum_lbl = 1;
        #             else:
        #                 sternum_lbl = 0;
                    
        #             pred.append(sternum_lbl);

        #         prec, rec, f1, _ = precision_recall_fscore_support(labels, pred,average='macro');
        #         cm = confusion_matrix(labels, pred);
        #         total_f1.append(f1);
        #         total_prec.append(rec);
        #         total_rec.append(prec);
        #         total_cm.append(cm);
            
        #     total_cm = np.array(total_cm);
        #     total_cm = np.sum(total_cm, axis = 0);
        #     #disp = ConfusionMatrixDisplay(total_cm, display_labels=['No', 'Yes']);
        #     #disp.plot();
        #     #plt.show();
        #     if np.mean(total_f1) > best_f1:
        #         best_f1 = np.mean(total_f1);
        #         best_class_thresh = thresh;
        #         best_pix_thresh = t;

    
    print(f'best_f1: {best_f1}\t param: {best_class_thresh}\t {best_pix_thresh}');
        

#     total_x = np.array(total_x);
#     total_y = np.array(total_y);

#     #lbl_list = np.expand_dims(lbl_list, axis = 1);
#     best_idx = -1;
#     best_score = 0;
    
#     #features = StandardScaler().fit_transform(features);
    
#     param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

#     param_grid = [
#         {'svc__C' : param_range,
#         'svc__kernel' : ['linear']},
#         {
#             'svc__C': param_range,
#             'svc__gamma' : param_range,
#             'svc__kernel' : ['rbf']
#         }
#     ];

#     parameter_space = {
#     'mlp__hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'mlp__activation': ['tanh', 'relu'],
#     'mlp__solver': ['sgd', 'adam'],
#     'mlp__alpha': [0.0001, 0.05],
#     'mlp__learning_rate': ['constant','adaptive'],
# }

#     parameter_space_ada = {
#         'n_estimators': [10,20,30,40,50,60,70,80,90,100]
#     }

#     # param_grid = {'decisiontreeclassifier__learning_rate' : param_range,
#     # 'decisiontreeclassifier__n_estimators': [1, 2, 4, 8, 16, 32, 64, 100, 200,300,500,1000]}

#     pipeline = Pipeline([('scalar', RobustScaler()),
#         #PCA(n_components=20),
#             ('svc',SVC(class_weight='balanced'))]);

#     # iris = datasets.load_iris()
#     # parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
#     # svc = SVC()
#     # clf = GridSearchCV(svc, parameters, scoring='f1', refit=True, n_jobs=-1)
#     # clf.fit(iris.data, iris.target, )
#     svm = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = custom_cv);
#     #mlp = GridSearchCV(estimator=MLPClassifier(), param_grid=parameter_space, scoring='f1', cv = 10, refit=True, n_jobs=-1);
#     #ada = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=parameter_space_ada, scoring='f1', cv = 10, refit=True, n_jobs=-1);

#     #lbl_list = preprocessing.label_binarize(lbl_list, classes=[0,1,2]);

#     svm = svm.fit(total_x, np.array(total_y, np.int32));
#     #mlp = mlp.fit(total_data, lbl_list);

#     # vc = VotingClassifier(estimators=[('svm', svm), ('mlp', mlp), ('ada', ada)], voting='hard');
#     #scores = cross_val_score(svm, total_data, lbl_list, scoring='f1', cv=10)
#     #print(scores.mean());

   
#     print(svm.best_params_);


def optimize_cranial_model(folds):
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];

    for idx,f in enumerate(folds):
        train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
        caudal_features, \
        symmetry_features, \
        sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11];

        train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
        test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));

        #train
        for i in tqdm(range(len(cranial_features))):    
            total_X.append(cranial_features[1]);
            total_Y.append(train_grain_lbl[i][0]);
        #--------------------------------------------------------------------------------------------------------

        #test
        for index,t in tqdm(enumerate(test_imgs)):
            feat_file = pickle.load(open(f'results\\{idx}\\outputs\\{t}_cranial.feat','rb'));

            total_X.append(feat_file);
            total_Y.append(test_grain_lbl[index][1]);
        #----------------------------------------------------------------------------------------

    custom_cv = zip(train_fold_indices, test_fold_indices);
    pickle.dump([total_X, total_Y, custom_cv], open('data.dmp', 'wb'));

    data = pickle.load(open('data.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];

    total_x = np.array(total_x);
    total_y = np.array(total_y, np.int32);
    
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

    pipe = Pipeline([('scalar',RobustScaler()), ('svc', SVC(class_weight='balanced'))]);

    svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = custom_cv);

    svm = svm.fit(total_x, total_y);

    print(f'Best score of {svm.best_score_} achieved with: {svm.best_params_}');

def optimize_caudal_model(folds):
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];

    for idx,f in enumerate(folds):
        train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
        caudal_features, \
        symmetry_features, \
        sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10], f[11];

        train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
        test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));

        #train
        for i in tqdm(range(len(caudal_features))):    
            total_X.append(caudal_features[i]);
            total_Y.append(train_grain_lbl[i][1]);
        #--------------------------------------------------------------------------------------------------------

        #test
        for index,t in tqdm(enumerate(test_imgs)):
            feat_file = pickle.load(open(f'results\\{idx}\\outputs\\{t}_caudal.feat','rb'));

            total_X.append(feat_file);
            total_Y.append(test_grain_lbl[index][1]);
        #----------------------------------------------------------------------------------------

    custom_cv = zip(train_fold_indices, test_fold_indices);
    pickle.dump([total_X, total_Y, custom_cv], open('data.dmp', 'wb'));

    data = pickle.load(open('data.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];

    total_x = np.array(total_x);
    total_y = np.array(total_y, np.int32);
    
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

    pipe = Pipeline([('scalar',RobustScaler()), ('svc', SVC(class_weight='balanced'))]);

    svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = custom_cv);

    svm = svm.fit(total_x, total_y);

    print(f'Best score of {svm.best_score_} achieved with: {svm.best_params_}');

def optimize_full_model(folds):
    total_X = [];
    total_Y = [];
    train_fold_indices = [];
    test_fold_indices = [];

    for idx,f in enumerate(folds):
        train_imgs, train_mask, train_lbl, train_grain_lbl, train_features, test_imgs, test_mask, test_lbl, test_grain_lbl = f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8];

        train_fold_indices.append(np.arange(len(total_X),len(total_X)+len(train_mask)));
        test_fold_indices.append(np.arange(len(total_X)+len(train_mask),len(total_X)+len(train_mask)+len(test_mask)));

        #train
        for i in tqdm(range(len(train_features))):    
            total_X.append(train_grain_lbl[i]);
            total_Y.append(train_lbl[i]);
        #--------------------------------------------------------------------------------------------------------

        #test
        for i in tqdm(range(len(test_grain_lbl))):
            total_X.append(test_grain_lbl[i]);
            total_Y.append(test_lbl[i]);
        #----------------------------------------------------------------------------------------

    custom_cv = zip(train_fold_indices, test_fold_indices);
    pickle.dump([total_X, total_Y, custom_cv], open('data.dmp', 'wb'));

    data = pickle.load(open('data.dmp', 'rb'));
    total_x, total_y, custom_cv = data[0], data[1], data[2];

    c_transform = ColumnTransformer([
        ('onehot', OneHotEncoder(), [3]),
        ('nothing', 'passthrough', [0,1,2])
        ]);

    total_x = np.array(total_x);
    total_x = c_transform.fit_transform(total_x);
    total_y = np.array(total_y, np.int32);
    
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

    pipe = Pipeline([('scalar',RobustScaler()), ('svc', SVC(class_weight='balanced'))]);

    svm = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1', n_jobs=-1, cv = custom_cv);

    svm = svm.fit(total_x, total_y);

    print(f'Best score of {svm.best_score_} achieved with: {svm.best_params_}');
