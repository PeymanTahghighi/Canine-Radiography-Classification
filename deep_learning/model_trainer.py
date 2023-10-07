from copy import deepcopy
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils import shuffle
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from torch.nn.modules.loss import L1Loss
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
from utility import extract_sternum_features, get_histogram, postprocess_sternum, remove_outliers, remove_outliers_hist_hor, remove_outliers_hist_ver, scale_width, smooth_boundaries, draw_missing_spine
import config
from deep_learning.network_dataset import CanineDataset
from deep_learning.network import Unet
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
from PIL import Image
from glob import glob
from torchvision.utils import save_image
import albumentations as A
from sklearn.svm import SVC
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import *
#import ptvsd
from deep_learning.stopping_strategy import *
from deep_learning.loss import dice_loss
from utility import divide_image_symmetry_line, get_symmetry_line, remove_blobs, remove_blobs_spine
from Symmetry.thorax import segment_thorax
from utility import extract_cranial_features
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torchvision.ops.focal_loss import sigmoid_focal_loss
from datetime import datetime

def train_cranial_model(fold_cnt, train_features, train_lbl):
    params =  {'mlp__activation':'tanh','mlp__alpha':1e-05, 'mlp__hidden_layer_sizes':10, 'mlp__learning_rate':'invscaling', 'mlp__solver':'adam'};
    model = make_pipeline(RobustScaler(),
            MLPClassifier(activation=params['mlp__activation'], alpha = params['mlp__alpha'], hidden_layer_sizes=params['mlp__hidden_layer_sizes'], learning_rate='invscaling', solver='adam'));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\cranial_model.pt', 'wb'));
    return model;

def train_caudal_model(fold_cnt, train_features, train_lbl):
    params =  {'mlp__activation':'tanh','mlp__alpha':0.1, 'mlp__hidden_layer_sizes':60, 'mlp__learning_rate':'adaptive', 'mlp__solver':'adam'};
    model = make_pipeline(RobustScaler(),
            MLPClassifier(activation=params['mlp__activation'], alpha = params['mlp__alpha'], hidden_layer_sizes=params['mlp__hidden_layer_sizes'], 
            learning_rate=params['mlp__learning_rate'], solver='adam'));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\caudal_model.pt', 'wb'));
    return model;

def train_symmetry_model(fold_cnt, train_features, train_lbl):
    model = make_pipeline(RobustScaler(),
            SVC(C=1.0, kernel = 'rbf'));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\symmetry_model.pt', 'wb'));
    return model;

def train_sternum_model(fold_cnt, train_features, train_lbl):

    params = {'svc__C': 0.01, 'svc__gamma': 1.0, 'svc__kernel': 'rbf'}
    model = make_pipeline(RobustScaler(),
            SVC(class_weight='balanced', C=params['svc__C'], gamma=params['svc__gamma'], kernel = params['svc__kernel']));
    train_features = list(train_features);
    train_features = np.array(train_features);
    model.fit(np.expand_dims(train_features,axis = 1), np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\sternum_model.pt', 'wb'));
    return model;

def train_full_model(fold_cnt, train_features, train_lbl):
    params = {'gradientboostingclassifier__learning_rate': 0.001, 'gradientboostingclassifier__max_depth': 5, 'gradientboostingclassifier__n_estimators': 500}
    model = make_pipeline(RobustScaler(),
            GradientBoostingClassifier(learning_rate=params['gradientboostingclassifier__learning_rate'], max_depth=params['gradientboostingclassifier__max_depth'], 
            n_estimators = params['gradientboostingclassifier__n_estimators']));
    model.fit(train_features, np.array(train_lbl,np.int32));
    pickle.dump(model, open(f'results\\{fold_cnt}\\full_model.pt', 'wb'));
    return model;
    

def store_results(fold_cnt, segmentation_models, test_imgs):
    for s in segmentation_models:
        s.eval();
    for idx in tqdm(range(len(test_imgs))):
       
        radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{test_imgs[idx]}.jpeg'),cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);


        transformed = config.valid_transforms(image = radiograph_image);
        radiograph_image = transformed["image"];
        radiograph_image = radiograph_image.to(config.DEVICE);
        
        #spine and ribs
        # out = segmentation_models[0](radiograph_image.unsqueeze(dim=0));
        # out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
        # out = np.argmax(out,axis = 2);

        # ribs = (out == 1).astype("uint8")*255;
        # spine = (out == 2).astype("uint8")*255;

        # ribs = remove_blobs(ribs);
        # hist_hor, hist_ver = get_histogram(ribs,config.IMAGE_SIZE);

        # spine = remove_blobs_spine(spine).astype("uint8");
        # spine_missing_drawn = draw_missing_spine(spine);
        
        # #----------------------------------------------------

        # abdomen
        # abdomen_mask = segmentation_models[1](radiograph_image.unsqueeze(dim=0));
        # abdomen_mask = torch.sigmoid(abdomen_mask)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
        # abdomen_mask = abdomen_mask > 0.5;
        # abdomen_mask = (abdomen_mask*255).astype("uint8")
        # #----------------------------------------------------

        # heart
        # heart_mask = segmentation_models[2](radiograph_image.unsqueeze(dim=0));
        # heart_mask = torch.sigmoid(heart_mask)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
        # heart_mask = heart_mask > 0.5;
        # heart_mask = (heart_mask*255).astype("uint8")
        # #----------------------------------------------------


        # # #sternum
        sternum = segmentation_models[0](radiograph_image.unsqueeze(dim=0));
        sternum = torch.sigmoid(sternum)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
        sternum = sternum > 0.7;
        sternum = np.uint8(sternum)*255;
        sternum = postprocess_sternum(sternum);
        # # #----------------------------------------------------
        
        #ribs_new = remove_outliers_hist_ver(hist_ver, ribs);
        #ribs_new = remove_outliers_hist_hor(hist_hor, ribs_new);

        #start = datetime.now();
        #whole_thorax = segment_thorax(ribs_new);
        #duration = (datetime.now() - start).microseconds*1e-6;
        #print(f'thorax segmentation took: {duration}')
        #whole_thorax = cv2.imread(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_thorax.png', cv2.IMREAD_GRAYSCALE);
        # thorax_left = segment_thorax(ribs_left);
        # thorax_right = segment_thorax(ribs_right);
        #     symmetry_features = extract_symmetry_features(thorax_left, thorax_right);
        #     symmetry_features = np.array(symmetry_features);
        
        # # #----------------------------------------------------

        # # #Cranial
        #cranial_features = extract_cranial_features(spine, whole_thorax);
        # cranial_features = np.array(cranial_features);
        # #-----------------------------------------------------

        # # #Caudal
        #caudal_features = extract_caudal_features(abdomen_mask, whole_thorax);
        #caudal_features = np.array(caudal_features, whole_thorax)
        
        # # #-----------------------------------------------------

        # # # #Sternum
        # spine_smoothed = smooth_boundaries(spine,10);
        # spine_smoothed = smooth_boundaries(spine_smoothed,25);
        # spine_smoothed = draw_missing_spine(spine_smoothed);
        # spine_scaled = scale_width(spine_smoothed,2);
        # # sternum = np.logical_and(sternum.squeeze(), np.where(whole_thorax>0, 1, 0)).astype(np.uint8);
        # sternum_features = extract_sternum_features(sternum, spine_scaled);

        # # # Spinous process
        # spine_mask_processed = smooth_boundaries(spine.astype("uint8")*255,10);
        # spine_mask_5 = draw_missing_spine(spine_mask_processed);
        # spine_mask_5 = scale_width(spine_mask_5,10);
        # spinous_process = segmentation_models[3](radiograph_image.unsqueeze(dim=0));
        # spinous_process = torch.sigmoid(spinous_process)[0].permute(1,2,0).detach().cpu().numpy().squeeze();
        # spinous_process = spinous_process > 0.5;
        # spinous_process = np.uint8(spinous_process)*255;

        # overlap = np.maximum(np.where(spinous_process>0,1,0) - np.where(spine_mask_5>0,1,0), np.zeros_like(spinous_process));
        # sp_features = np.sum(overlap);

        # #store results
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_spine.png', spine_smoothed);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_spine_orig.png', spine);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_spine_scaled.png', spine_scaled);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_ribs_orig.png', ribs_new);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_diaph.png', abdomen_mask);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_heart.png', heart_mask);
        cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_sternum.png', sternum);
        
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_thorax.png', whole_thorax);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_spinous_prcess.png', spinous_process);
        #cv2.imwrite(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_sp_overlap.png', overlap.astype("uint8")*255);
            
    #     else:
    #         tips_features = pickle.load(open(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_sp.feat','rb'));
    #         cranial_features = pickle.load(open(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_cranial.feat','rb'));
    #         caudal_features = pickle.load(open(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_caudal.feat','rb'));
    #         sternum_features = pickle.load( open(f'results\\{fold_cnt}\\outputs\\{test_imgs[idx]}_sternum.feat','rb'));

    #     if skip_results is True:
    #         continue;
    #     cranial_lbl = classification_models[0].predict(cranial_features.reshape(1,-1));
    #     caudal_lbl = classification_models[1].predict(np.array(caudal_features).reshape(1,-1));
    #     sternum_lbl = classification_models[3].predict(np.array(sternum_features[1]).reshape(1,-1));
    #     if tips_features > 10:
    #         tips_lbl = 1;
    #     else:
    #         tips_lbl = 0;
            
        
    #     grain_lbls = [cranial_lbl[0], caudal_lbl[0], tips_lbl, sternum_lbl[0]];

    #     #grain_lbls = transformer.transform(np.array(grain_lbls).reshape(1,-1));
    #     #-----------------------------------------------------

    #     # cv2.imshow('spine', spine);
    #     # cv2.waitKey();

    #     quality_lbl = classification_models[4].predict(np.array(grain_lbls).reshape(1,-1));
    #     if quality_lbl[0] != test_lbl[idx]:
    #         print(f'{test_imgs[idx]}\t\tgrain: {grain_lbls} \ttrue grain: {test_grain_lbl[idx]}\tpred: {quality_lbl[0]}\ttrue: {test_lbl[idx]}');

    #     all_predictions.append([cranial_lbl[0], caudal_lbl[0], tips_lbl, sternum_lbl[0], quality_lbl[0]]);
    

    # #get performance metrics
    # if skip_results is True:
    #     return [0, 0, 0, 0],\
    #        [0, 0, 0, 0],\
    #        [0, 0, 0, 0],\
    #        [0, 0, 0, 0],\
    #        [0, 0, 0, 0];
    # all_predictions = np.array(all_predictions);
    # cranial_precision, cranial_recall, cranial_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,0],np.int32), np.array(all_predictions[:,0],np.int32), average='binary');
    # cranial_accuracy = accuracy_score(np.array(test_grain_lbl[:,0],np.int32), np.array(all_predictions[:,0],np.int32));

    # caudal_precision, caudal_recall, caudal_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,1],np.int32), np.array(all_predictions[:,1],np.int32), average = 'binary');
    # caudal_accuracy = accuracy_score(np.array(test_grain_lbl[:,1],np.int32), np.array(all_predictions[:,1],np.int32));

    # tips_precision,tips_recall, tips_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,2],np.int32), np.array(all_predictions[:,2],np.int32), average = 'binary');
    # tips_accuracy = accuracy_score(np.array(test_grain_lbl[:,2],np.int32), np.array(all_predictions[:,2],np.int32));

    # sternum_precision, sternum_recall, sternum_f1,_ = precision_recall_fscore_support(np.array(test_grain_lbl[:,3],np.int32), np.array(all_predictions[:,3],np.int32), average='binary');
    # sternum_accuracy = accuracy_score(np.array(test_grain_lbl[:,3],np.int32), np.array(all_predictions[:,3],np.int32));

    # quality_precision, quality_recall, quality_f1,_ = precision_recall_fscore_support(test_lbl, np.array(all_predictions[:,4],np.int32), average='binary');
    # quality_accuracy = accuracy_score(test_lbl, np.array(all_predictions[:,4],np.int32));
    # # #--------------------------------------------------

    # return [cranial_precision, cranial_recall, cranial_f1, cranial_accuracy],\
    #        [caudal_precision, caudal_recall, caudal_f1, caudal_accuracy],\
    #        [tips_precision, tips_recall, tips_f1, tips_accuracy],\
    #        [sternum_precision, sternum_recall, sternum_f1, sternum_accuracy],\
    #        [quality_precision, quality_recall, quality_f1, quality_accuracy];


class NetworkTrainer():

    def __init__(self):
        pass

    def __loss_func(self, output, gt):
        if self.num_classes > 1:
            f_loss = F.cross_entropy(output, gt.squeeze(dim=3).long(), reduction='mean');
            t_loss = dice_loss(output.squeeze(dim=1), gt.squeeze(dim=3), sigmoid=False)
            return  t_loss + f_loss;
        else:

            f_loss = sigmoid_focal_loss(output.squeeze(dim=1), gt.float(), reduction="mean");
            t_loss = dice_loss(output.squeeze(dim=1), gt, sigmoid=True)
            return  t_loss + f_loss;
        
    def __train_one_epoch(self, epoch, loader, model, optimizer):
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
                loss = self.__loss_func(pred, mask) / config.VIRTUAL_BATCH_SIZE;

            self.scaler.scale(loss).backward();
            epoch_loss.append(loss.item());

            if ((batch_idx+1) % config.VIRTUAL_BATCH_SIZE == 0 or (batch_idx+1) == len(loader)):
                self.scaler.step(optimizer);
                self.scaler.update();
                model.zero_grad(set_to_none = True);
            step += 1;

            pbar.set_description(('%10s' + '%10.4g') %(epoch, np.mean(epoch_loss)));
        return np.mean(epoch_loss);

    def __eval_one_epoch(self, epoch, loader, model):
        epoch_loss = [];
        total_prec = [];
        total_rec = [];
        total_f1 = [];
        total_acc = [];
        
        pbar = enumerate(loader);
        print(('\n' + '%10s'*6) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i ,(radiograph, mask) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = model(radiograph);
                loss = self.__loss_func(pred, mask);

                epoch_loss.append(loss.item());
                
                if self.num_classes > 1:
                    pred = (torch.softmax(pred, dim = 1)).permute(0,2,3,1);
                    pred = torch.argmax(pred, dim = 3);
                else:
                    pred = torch.sigmoid(pred) > 0.5;
                prec = self.precision_estimator(pred.flatten(), mask.flatten().long());
                rec = self.recall_estimator(pred.flatten(), mask.flatten().long());
                acc = self.accuracy_esimator(pred.flatten(), mask.flatten().long());
                f1 = self.f1_esimator(pred.flatten(), mask.flatten().long());
                
                
                total_prec.append(prec.item());
                total_rec.append(rec.item());
                total_f1.append(f1.item());
                total_acc.append(acc.item());

                pbar.set_description(('%10s' + '%10.4g'*5) % (epoch, np.mean(epoch_loss),
                np.mean(total_prec), np.mean(total_rec), np.mean(total_f1), np.mean(total_acc)))

        return np.mean(epoch_loss), np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);

    def get_lr(self, lr, e):
        return min(1,e/config.WARMUP_EPOCHS)*lr;

    def train(self, task_name, num_classes, model, fold_cnt, train_imgs, train_mask, test_imgs, test_mask, load_trained_model = False, exposure_labels = None):

        if load_trained_model is True:
            model.load_state_dict(pickle.load( open(f'results\\{fold_cnt}\\{task_name}.pt', 'rb')));
            return model;

        self.num_classes = num_classes;
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.precision_estimator = Precision(num_classes=num_classes, multiclass=False if num_classes ==1 else True, average='macro').to(config.DEVICE);
        self.recall_estimator = Recall(num_classes=num_classes, multiclass=False if num_classes ==1 else True , average='macro').to(config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=num_classes, multiclass=False if num_classes ==1 else True , average='macro').to(config.DEVICE);
        self.f1_esimator = F1Score(num_classes=num_classes, multiclass=False if num_classes ==1 else True , average='macro').to(config.DEVICE);

        train_dataset = CanineDataset(train_imgs, train_mask, exposure_labels=exposure_labels, train = True);
        valid_dataset = CanineDataset(test_imgs, test_mask, exposure_labels=None, train = False);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

        model.reset_weights();
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-6);

        stopping_strategy = CombinedTrainValid(2.0,5);
        summary = SummaryWriter(f'experiments\\{fold_cnt}\\swin-large');
        #scheduler = CosineAnnealingLR(optimizer, 200);
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.9,verbose=True,);

        best = 100;
        e = 1;
        best_model = None;
        best_prec = 0;
        best_recall = 0;
        best_f1 = 0;
        best_acc = 0;

        print(f'Started training task: {task_name}');

        while(True):

            # if e<= config.WARMUP_EPOCHS:
            #     for p in optimizer.param_groups:
            #         p['lr'] = self.get_lr(config.LEARNING_RATE,e);

            model.train();
            train_loss = self.__train_one_epoch(e, train_loader, model, optimizer);

            model.eval();

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(e, valid_loader, model);

            print(f"Valid \tTrain: {train_loss}\tValid {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");
            summary.add_scalar('train/train_loss', train_loss, e);
            summary.add_scalar('valid/valid_loss', valid_loss, e);
            summary.add_scalar('valid/valid_f1', valid_f1, e);

            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(model.state_dict());
                pickle.dump(best_model, open(f'results\\{fold_cnt}\\{task_name}.pt', 'wb'));
                best_prec = valid_precision;
                best_recall = valid_recall;
                best_f1 = valid_f1;
                best_acc = valid_acc;

            if stopping_strategy(valid_loss, train_loss) is False:
                break;
            e += 1;
            # #if e > config.WARMUP_EPOCHS:
            # scheduler.step(train_loss);
        f = open(f'results\\{fold_cnt}\\res_{task_name}_pretrain.txt', 'w');
        f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        f.close();
        pickle.dump(best_model, open(f'results\\{fold_cnt}\\{task_name}.pt', 'wb'));

        #load model with best weights to save outputs
        model.load_state_dict(best_model);
        return model;