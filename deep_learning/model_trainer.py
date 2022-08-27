from copy import deepcopy
import pickle
from sklearn.utils import shuffle
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from torch.nn.modules.loss import L1Loss
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
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
import torchvision.transforms.functional as F
from ignite.contrib.handlers.tensorboard_logger import *
from torch.utils.data import DataLoader
from torchmetrics import *
#import ptvsd
from deep_learning.stopping_strategy import *
from deep_learning.loss import dice_loss, focal_loss, tversky_loss
from utility import divide_image_symmetry_line, get_symmetry_line, remove_blobs, remove_blobs_spine
from Symmetry.thorax import segment_thorax
from utils import extract_cranial_features, extract_symmetry_features

def train_cranial_model(fold_cnt, train_features, train_lbl):
    model = SVC();
    model.fit(train_features, train_lbl);
    pickle.dump(model, open(f'results\\{fold_cnt}\\cranial_model.pt', 'wb'));
    return model;

def train_caudal_model(fold_cnt, train_features, train_lbl):
    model = SVC();
    model.fit(train_features, train_lbl);
    pickle.dump(model, open(f'results\\{fold_cnt}\\caudal_model.pt', 'wb'));
    return model;

def train_symmetry_model(fold_cnt, train_features, train_lbl):
    model = SVC();
    model.fit(train_features, train_lbl);
    pickle.dump(model, open(f'results\\{fold_cnt}\\caudal_model.pt', 'wb'));
    return model;

def train_full_model(fold_cnt, train_features, train_lbl):
    model = SVC();
    model.fit(train_features, train_lbl);
    pickle.dump(model, open(f'results\\{fold_cnt}\\full_model.pt', 'wb'));
    return model;
    

def evaluate_test_data(fold_cnt, segmentation_models, classification_models, test_imgs, test_grain_lbl, test_lbl):
    all_predictions = [];
    for radiograph_image_path in test_imgs:
        file_name = os.path.basename(radiograph_image_path);
        file_name = file_name[:file_name.rfind('.')];

        file_name = os.path.basename(radiograph_image_path);
        file_name = file_name[:file_name.rfind('.')];

        radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);


        transformed = config.valid_transforms(image = radiograph_image);
        radiograph_image = transformed["image"];
        radiograph_image = radiograph_image.to(config.DEVICE);
        
        #spine and ribs
        out = segmentation_models[0](radiograph_image.unsqueeze(dim=0));
        out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
        out = np.argmax(out,axis = 2);

        ribs = (out == 1).astype("uint8")*255;
        spine = (out == 2).astype("uint8")*255;

        ribs = remove_blobs(ribs);
        spine = remove_blobs_spine(spine);
        #----------------------------------------------------

        #diaphragm
        diaphragm = segmentation_models[1](radiograph_image.unsqueeze(dim=0));
        diaphragm = torch.sigmoid(diaphragm)[0].permute(1,2,0).detach().cpu().numpy();
        diaphragm = diaphragm > 0.5;
        diaphragm = np.uint8(diaphragm)*255;
        #----------------------------------------------------

        #sternum
        sternum = segmentation_models[2](radiograph_image.unsqueeze(dim=0));
        sternum = torch.sigmoid(sternum)[0].permute(1,2,0).detach().cpu().numpy();
        sternum = sternum > 0.5;
        sternum = np.uint8(sternum);
        #----------------------------------------------------

        #Symmetry
        sym_line = get_symmetry_line(spine);
        ribs_left, ribs_right = divide_image_symmetry_line(ribs, sym_line);
        thorax_left = segment_thorax(ribs_left);
        thorax_right = segment_thorax(ribs_right);
        whole_thorax = segment_thorax(ribs);
        symmetry_features = extract_symmetry_features(thorax_left, thorax_right);
        symmetry_lbl = classification_models[2].predict(symmetry_features);
        #----------------------------------------------------

        #Cranial
        cranial = spine - whole_thorax;
        cranial_features = extract_cranial_features(cranial);
        cranial_lbl = classification_models[0].predict(cranial_features);
        #-----------------------------------------------------

        #Caudal
        caudal = diaphragm - whole_thorax;
        caudal_features = extract_cranial_features(caudal);
        caudal_lbl = classification_models[1].predict(cranial_features);
        #-----------------------------------------------------

        #Sternum
        sternum = np.logical_and(sternum.squeeze(), whole_thorax).astype(np.uint8);
        sternum_features = np.sum(sternum, (1,2));
        if sternum_features > 32:
            sternum_lbl = 1;
        else:
            sternum_lbl = 0;
        #-----------------------------------------------------

        quality_lbl = classification_models[3].predict([cranial_lbl, caudal_lbl, symmetry_lbl, sternum_lbl]);

        all_predictions.append([cranial_lbl, caudal_lbl, symmetry_lbl, sternum_lbl, quality_lbl]);


        pickle.dump([cranial_features, caudal_features, symmetry_features, sternum_features], f'results\\{fold_cnt}\\test\\{file_name}.feat');
    

    #get performance metrics

    cranial_precision, cranial_recall, cranial_f1,_ = precision_recall_fscore_support(test_grain_lbl[:,0], all_predictions[:,0]);
    cranial_accuracy = accuracy_score(test_grain_lbl[:,0], all_predictions[:,0]);

    caudal_precision, caudal_recall, caudal_f1,_ = precision_recall_fscore_support(test_grain_lbl[:,1], all_predictions[:,1]);
    caudal_accuracy = accuracy_score(test_grain_lbl[:,1], all_predictions[:,1]);

    symmetry_precision, symmetry_recall, symmetry_f1,_ = precision_recall_fscore_support(test_grain_lbl[:,2], all_predictions[:,2]);
    symmetry_accuracy = accuracy_score(test_grain_lbl[:,0], all_predictions[:,2]);

    sternum_precision, sternum_recall, sternum_f1,_ = precision_recall_fscore_support(test_grain_lbl[:,3], all_predictions[:,3]);
    sternum_accuracy = accuracy_score(test_grain_lbl[:,0], all_predictions[:,3]);

    quality_precision, quality_recall, quality_f1,_ = precision_recall_fscore_support(test_lbl, all_predictions[:,4]);
    quality_accuracy = accuracy_score(test_lbl, all_predictions[:,4]);
    #--------------------------------------------------

    print(('\n'+'%10s'*5)%('Type', 'Precision', 'Recall', 'F1', 'Accuracy'));
    print(('\n'+'%10s'*1 + '%10d'*4)%('Cranial',cranial_precision, cranial_recall, cranial_f1, cranial_accuracy));
    print(('\n'+'%10s'*1 + '%10d'*4)%('Caudal',caudal_precision, caudal_recall, caudal_f1, caudal_accuracy));
    print(('\n'+'%10s'*1 + '%10d'*4)%('Symmetry',symmetry_precision, symmetry_recall, symmetry_f1, symmetry_accuracy));
    print(('\n'+'%10s'*1 + '%10d'*4)%('Sternum',sternum_precision, sternum_recall, sternum_f1, sternum_accuracy));
    print(('\n'+'%10s'*1 + '%10d'*4)%('Quality',quality_precision, quality_recall, quality_f1, quality_accuracy));

class NetworkTrainer():

    def __init__(self):
        self.__initialize();
        pass

    def __loss_func(self, output, gt):
        f_loss = focal_loss(output, gt,  arange_logits=True, mutual_exclusion=True);
        t_loss = tversky_loss(output, gt, sigmoid=False, arange_logits=True, mutual_exclusion=True)
        return  t_loss + f_loss;
        
    def __train_one_epoch(self, epoch, loader, model, optimizer):
        epoch_loss = [];
        step = 0;
        update_step = 1;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (radiograph, mask) in pbar:
            radiograph, mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE)
            model.zero_grad(set_to_none = True);

            with torch.cuda.amp.autocast_mode.autocast():
                pred = model(radiograph);
                loss = self.__loss_func(pred, mask);

            self.scaler.scale(loss).backward();
            epoch_loss.append(loss.item());
            step += 1;

            if step % update_step == 0:
                self.scaler.step(optimizer);
                self.scaler.update();

            pbar.set_description(('%10s' + '%10.4g') %(epoch, np.mean(epoch_loss)));

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
                
                pred = (torch.softmax(pred, dim = 1)).permute(0,2,3,1);
                pred = torch.argmax(pred, dim = 3);
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


    def train(self, task_name, num_classes, model, fold_cnt, train_imgs, train_mask, test_imgs, test_mask):

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.precision_estimator = Precision(num_classes=num_classes).to(config.DEVICE);
        self.recall_estimator = Recall(num_classes=num_classes).to(config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=num_classes).to(config.DEVICE);
        self.f1_esimator = F1Score(num_classes=num_classes).to(config.DEVICE);

        train_dataset = CanineDataset(train_imgs, train_mask, config.train_transforms);
        valid_dataset = CanineDataset(test_imgs, test_mask, config.valid_transforms);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

        model.reset_weights();
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5);

        stopping_strategy = CombinedTrainValid(0.7,2);

        best = 100;
        e = 1;
        best_model = None;
        best_prec = 0;
        best_recall = 0;
        best_f1 = 0;
        best_acc = 0;

        print(f'Started training task: {task_name}');

        while(True):
            self.model.train();
            self.__train_one_epoch(e, train_loader,model, optimizer);

            model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.__eval_one_epoch(e, train_loader, model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(e, valid_loader, model);

            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");


            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(self.model.state_dict());
                best_prec = valid_precision;
                best_recall = valid_recall;
                best_f1 = valid_f1;
                best_acc = valid_acc;

            if stopping_strategy(valid_loss, train_loss) is False:
                break;
            e += 1;
        f = open(f'res{fold_cnt}.txt', 'w');
        f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        f.close();
        pickle.dump(best_model, open(f'results\\{fold_cnt}\\{task_name}.pt', 'wb'));

        #load model with best weights to save outputs
        model.load_state_dict(best_model);
        return model;
    

    def eval(self, checkpoint_path, test_data):

        self.model.load_state_dict(pickle.load(open(checkpoint_path,'rb')));

        valid_dataset = SternumDataset(test_data[0], test_data[1], test_data[2], config.valid_transforms);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

        #while(True):
        pbar = enumerate(valid_loader);
        print(('\n' + '%10s'*6) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc'));
        pbar = tqdm(pbar, total= len(valid_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        all_data = [];
        all_lbl = [];
        cnt = 0;
        with torch.no_grad():
            for i ,(radiograph, mask, gt_lbl) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = self.model(radiograph);
                pred = torch.sigmoid(pred) > 0.5;
                pred_np = pred.permute(0,2,3,1).detach().cpu().numpy();
                pred_np = np.uint8(pred_np)*255;

                positives = torch.sum(pred == 1, [1,2,3]).detach().cpu().numpy();

                all_data.extend(positives);

                radiograph_np = radiograph.permute(0,2,3,1).detach().cpu().numpy();
                radiograph_np = radiograph_np *  [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406];
                radiograph_np = np.uint8(radiograph_np*255);
                for b in range((pred.shape[0])):
                    cv2.imwrite(f'tests\\{cnt}_{gt_lbl[b]}.png', radiograph_np[b]);
                    cv2.imwrite(f'tests\\{cnt}_{gt_lbl[b]}_seg.png', pred_np[b]);
                    cnt += 1;

        plt.scatter(all_data, all_lbl);
        plt.show();
    
    def save_samples(self, fold_cnt, task_name, test_img):
        
        os.mkdir(f'{task_name}\\test\\{fold_cnt}');
        
        for radiograph_image_path in test_img:

            file_name = os.path.basename(radiograph_image_path);
            file_name = file_name[:file_name.rfind('.')];

            radiograph_image = cv2.imread(radiograph_image_path,cv2.IMREAD_GRAYSCALE);
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            radiograph_image = clahe.apply(radiograph_image);
            radiograph_image = np.expand_dims(radiograph_image, axis=2);
            radiograph_image = np.repeat(radiograph_image, 3,axis=2);


            transformed = config.valid_transforms(image = radiograph_image);
            radiograph_image = transformed["image"];
            radiograph_image = radiograph_image.to(config.DEVICE);
            
            out = self.model(radiograph_image.unsqueeze(dim=0));
            out = (torch.softmax(out, dim= 1)[0].permute(1,2,0)).detach().cpu().numpy();
            out = np.argmax(out,axis = 2);
            ribs = (out == 1).astype("uint8")*255;
            spine = (out == 2).astype("uint8")*255;
            ribs_proc = remove_blobs(ribs);
            spine_proc = remove_blobs_spine(spine);

            
            sym_line = get_symmetry_line(spine_proc);
            ribs_left, ribs_right = divide_image_symmetry_line(ribs_proc, sym_line);
            thorax_left = segment_thorax(ribs_left);
            thorax_right = segment_thorax(ribs_right);
            #total_thorax = segment_thorax(ribs_proc);

            cv2.imwrite(f'{fold_cnt}\\{file_name}_left.png', thorax_left);
            cv2.imwrite(f'{fold_cnt}\\{file_name}_right.png', thorax_right);
