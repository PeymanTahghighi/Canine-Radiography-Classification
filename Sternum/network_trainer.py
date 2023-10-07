import pickle
from re import L
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from torch.nn.modules.loss import L1Loss
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
from Utility import draw_missing_spine, get_max_contour, retarget_img, scale_width, smooth_boundaries
import config
from network_dataset import CanineDatasetClass, CanineDatasetSeg, preload_classification_dataset
from network import *
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, precision_recall_curve
import os
from PIL import Image
from glob import glob
from torchvision.utils import save_image
import albumentations as A
import PIL.ImageColor
import torchvision.transforms.functional as F
from ignite.contrib.handlers.tensorboard_logger import *
from torch.utils.data import DataLoader
from torchmetrics import *
#import ptvsd
from stopping_strategy import *
from loss import dice_loss, focal_loss, tversky_loss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torch.utils.tensorboard import SummaryWriter
from transformer import SwinUNETR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd




class NetworkTrainer():

    def __init__(self):
        self.__initialize();
        pass

    #This function should be called once the program starts
    def __initialize(self,):

        #self.model = SwinUNETR((config.IMAGE_SIZE, config.IMAGE_SIZE),3,1,num_heads=(6,12,24,48), feature_size=48).to(config.DEVICE);
        #self.model = UNet(2,3,1,(64,64,128,128,256,256,512,512,1024,), (1,2,1,2,1,2,1,2), num_res_units=3).to(config.DEVICE);
        self.model = Unet(1).to(config.DEVICE);
        self.init_weights = deepcopy(self.model.state_dict());
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.precision_estimator = Precision(task='binary', num_classes=1).to(config.DEVICE);
        self.recall_estimator = Recall(task='binary',num_classes=1).to(config.DEVICE);
        self.accuracy_esimator = Accuracy(task='binary',num_classes=1).to(config.DEVICE);
        self.f1_esimator = F1Score(task='binary',num_classes=1).to(config.DEVICE);

        pass

    def __loss_func(self, output, gt):
        #f_loss = torctorch.binary_cross_entropy_with_logits(output.squeeze(dim=1), gt.float(), pos_weight=torch.tensor(134.95))
        f_loss = sigmoid_focal_loss(output.squeeze(dim=1), gt.float(), reduction="mean");
        t_loss = dice_loss(output.squeeze(dim=1), gt, sigmoid=True)
        return  t_loss + f_loss;
        


    def __train_one_epoch_seg(self, epoch, loader, model, optimizer):
        epoch_loss = [];
        step = 0;
        update_step = 2;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (radiograph, mask) in pbar:
            radiograph, mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE)
            
            with torch.cuda.amp.autocast_mode.autocast():
                pred = model(radiograph);
                loss = self.__loss_func(pred, mask) / update_step;

            self.scaler.scale(loss).backward();
            epoch_loss.append(loss.item());

            if step % update_step == 0 or (step) == len(loader):
                self.scaler.step(optimizer);
                self.scaler.update();
                model.zero_grad(set_to_none = True);
            step += 1;

            pbar.set_description(('%10s' + '%10.4g') %(epoch, np.mean(epoch_loss)));
        return np.mean(epoch_loss);

    def __train_one_epoch_class(self, epoch, loader, model, optimizer, loss_func):
        epoch_loss = [];
        step = 0;
        update_step = 1;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (radiograph, lbl) in pbar:
            if step == 1:
                model.zero_grad(set_to_none = True);

            radiograph, lbl = radiograph.to(config.DEVICE), lbl.to(config.DEVICE)
            rad_np = radiograph.permute(0,2,3,1).detach().cpu().numpy();
            # for i in range(2):
            #     rad = rad_np[i];
            #     rad = rad * (0.229, 0.224, 0.225) +  (0.485, 0.456, 0.406);
            #     rad = rad*255;
            #     rad = np.uint8(rad);
            #     cv2.imshow('rad', rad);
            #     cv2.waitKey();

            with torch.cuda.amp.autocast_mode.autocast():
                pred = model(radiograph);
                loss = loss_func(pred.squeeze(), lbl) / update_step;

            self.scaler.scale(loss).backward();
            epoch_loss.append(loss.item());

            if step % update_step == 0 or (step) == len(loader):
                self.scaler.step(optimizer);
                self.scaler.update();
                model.zero_grad(set_to_none = True);
            step += 1;

            pbar.set_description(('%10s' + '%10.4g') %(epoch, np.mean(epoch_loss)));
        return np.mean(epoch_loss);

    def __eval_one_epoch_class(self, epoch, loader, model, loss_func, metrics):
        epoch_loss = [];
        total_prec = [];
        total_rec = [];
        total_f1 = [];
        total_acc = [];
        cnt = 0;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*6) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i ,(radiograph, lbl) in pbar:
                radiograph,lbl = radiograph.to(config.DEVICE), lbl.to(config.DEVICE);

                pred = model(radiograph);
                loss = loss_func(pred.squeeze(), lbl);

                epoch_loss.append(loss.item());
                
                
                pred = torch.sigmoid(pred.squeeze()) > 0.5;
                prec = metrics[0](pred.flatten(),lbl.long());
                rec = metrics[1](pred.flatten(), lbl.long());
                acc = metrics[2](pred.flatten(), lbl.long());
                f1 = metrics[3](pred.flatten(), lbl.long());
                
                total_prec.append(prec.item());
                total_rec.append(rec.item());
                total_f1.append(f1.item());
                total_acc.append(acc.item());

                pbar.set_description(('%10s' + '%10.4g'*5) % (epoch, np.mean(epoch_loss),
                np.mean(total_prec), np.mean(total_rec), np.mean(total_f1), np.mean(total_acc)))
            
            print(f'{cnt/len(loader)}');



        return np.mean(epoch_loss), np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);
    
    def __eval_one_epoch_seg(self, epoch, loader, model):
        epoch_loss = [];
        total_prec = [];
        total_rec = [];
        total_f1 = [];
        total_acc = [];
        cnt = 0;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*6) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i ,(radiograph, mask) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = model(radiograph);
                loss = self.__loss_func(pred, mask);

                epoch_loss.append(loss.item());
                
                
                pred = torch.sigmoid(pred) > 0.5;
                # pred = ;
                #mask_np = mask.flatten().detach().cpu().numpy();
                if torch.sum(mask)== 0:
                    cnt+=1;

                    #prec_s, rec_s, f1_s, _ = precision_recall_fscore_support(mask.flatten().detach().cpu().numpy(), pred.flatten().detach().cpu().numpy(), average='binary');
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
            
            print(f'{cnt/len(loader)}');



        return np.mean(epoch_loss), np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);

    def get_lr(self, lr, e):
        return min(1,e/config.WARMUP_EPOCHS)*lr;

    def train(self, fold_cnt, train_data, test_data):

        self.model.load_state_dict(self.init_weights);
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE);
        scheduler = CosineAnnealingWarmRestarts(optimizer,50,2);

        stopping_strategy_stop = 20;
        curr_stop = stopping_strategy_stop;

        train_dataset = CanineDatasetSeg(train_data[0], train_data[1],train=True);
        valid_dataset = CanineDatasetSeg(test_data[0], test_data[1],  train = False);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);
        summary = SummaryWriter(f'exp\\{fold_cnt}');

        best = 100;
        e = 1;
        best_model = None;
        best_acc = 0;
        best_prec = 0;
        best_f1 = 0;
        best_recall = 0;
        best_class_acc = 0;

        while(True):

            self.model.train();
            train_loss = self.__train_one_epoch_seg(e, train_loader,self.model, optimizer);

            self.model.eval();

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch_seg(e, valid_loader, self.model);

            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

            summary.add_scalar('train/loss', train_loss, e);
            summary.add_scalar('valid/loss', valid_loss, e);
            summary.add_scalar('valid/f1', valid_f1, e);


            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(self.model.state_dict());
                best_acc = valid_acc;
                best_prec = valid_precision;
                best_f1 = valid_f1;
                best_recall = valid_recall;
                pickle.dump(best_model, open(f'ckpt{fold_cnt}.pt', 'wb'));
                curr_stop = stopping_strategy_stop;

            else:
                curr_stop -=1;
            
            if curr_stop == 0:
                break;

            
            #scheduler.step(e);
            e += 1;
        f = open(f'res{fold_cnt}.txt', 'w');
        f.write(f"Precision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        f.close();
        pickle.dump(best_model, open(f'ckpt{fold_cnt}.pt', 'wb'));
        return best_f1;
    
    def train_classifier(self, fold_cnt, train_data, test_data):
        #preload_classification_dataset(fold_cnt, train_data[0], True);
        #preload_classification_dataset(fold_cnt, test_data[0], False);
        model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True);
        model.classifier =  nn.Linear(1024, 1, bias=True);
        model = model.to(config.DEVICE);

        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE);
        scheduler = CosineAnnealingWarmRestarts(optimizer,50,2);

        stopping_strategy_stop = 20;
        curr_stop = stopping_strategy_stop;

        train_dataset = CanineDatasetClass(fold_cnt,  train=True);
        valid_dataset = CanineDatasetClass(fold_cnt,train = False);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE*8, shuffle=True);
        summary = SummaryWriter(f'exp\\{fold_cnt}_cls');
        loss = nn.BCEWithLogitsLoss();

        precision_estimator = Precision(num_classes=1, multiclass=False).to(config.DEVICE);
        recall_estimator = Recall(num_classes=1, multiclass=False).to(config.DEVICE);
        accuracy_esimator = Accuracy(num_classes=1, multiclass=False).to(config.DEVICE);
        f1_esimator = F1Score(num_classes=1, multiclass=False).to(config.DEVICE);

        best = 100;
        e = 1;
        best_model = None;
        best_acc = 0;
        best_prec = 0;
        best_f1 = 0;
        best_recall = 0;
        best_class_acc = 0;

        while(True):

            model.train();
            train_loss = self.__train_one_epoch_class(e, train_loader,model, optimizer, loss);

            model.eval();

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch_class(e, valid_loader, model, loss, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);

            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

            summary.add_scalar('train/loss', train_loss, e);
            summary.add_scalar('valid/loss', valid_loss, e);
            summary.add_scalar('valid/f1', valid_f1, e);


            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(model.state_dict());
                best_acc = valid_acc;
                best_prec = valid_precision;
                best_f1 = valid_f1;
                best_recall = valid_recall;
                pickle.dump(best_model, open(f'ckpt{fold_cnt}_class.pt', 'wb'));
                curr_stop = stopping_strategy_stop;

            else:
                curr_stop -=1;
            
            if curr_stop == 0:
                break;
            
            #scheduler.step(e);
            e += 1;
        f = open(f'res{fold_cnt}.txt', 'w');
        f.write(f"Precision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        f.close();
        pickle.dump(best_model, open(f'ckpt{fold_cnt}.pt', 'wb'));
        return best_f1;
    

    def store_results(self, fold_cnt, test_data, total_test_sternum1):

        if os.path.exists(f'{fold_cnt}\\test') is False:
            os.makedirs(f'{fold_cnt}\\test');
        

        self.model.load_state_dict(pickle.load(open(f'ckpt{fold_cnt}.pt','rb')));
        self.model.eval();

        labels_file = pd.read_excel('test.xlsx');
        img_lst = list(labels_file['Image']);
        lbl_lst = list(labels_file['Sternum']);

        total_data, total_gt, total_img = pickle.load(open('d.dmp','rb'))
       # total_data1, total_gt1, total_img1 = pickle.load(open('d1.dmp','rb'))
        total_gt  = np.array(total_gt, dtype=np.int32);
        qda = LogisticRegression();

        train_imgs = test_data[0];
        train_x = [];
        train_y = [];
        test_x = [];
        test_y = [];

        fig,ax = plt.subplots(1,3);
        color = ['g', 'r'];
        

        for i in range(len(train_imgs)):
            if train_imgs[i] in total_img:
                train_x.append([total_data[total_img.index(train_imgs[i])][0], total_data[total_img.index(train_imgs[i])][1]])
                train_y.append(total_gt[total_img.index(train_imgs[i])])
        #rs = RobustScaler();
        #rs = rs.fit(train_x);
        #train_x = rs.transform(train_x);
        
        # for i in range(len(train_x)):
        #     ax[0].scatter(train_x[i][0], train_x[i][1], c = color[train_y[i]]);
        #     ax[0].text(train_x[i][0], train_x[i][1], train_imgs[i]);
            # else:
            #     print(train_imgs[i]);
        train_y = np.array(train_y, np.int32);
        
        #qda.fit(train_x, train_y);
        radiographs = test_data[1];
        #labels = test_data[1];

        
        x_test = [];
        y_test = [];
        x_train = [];
        y_train = [];
        cnt = 0;
        total_gt = [];
        total_pred = [];
        for index in tqdm(range(len(radiographs))):
            if radiographs[index] == '8':
                print('hi');
            if radiographs[index] == '917'  or radiographs[index] == '962' or radiographs[index] == '963' or radiographs[index] == '964':
                continue;
            if os.path.exists (os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{radiographs[index]}.jpeg')) is True:
                radiograph_image = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final',f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
            else:
                radiograph_image = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\additionalDVVD',f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);

            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            #radiograph_image = clahe.apply(radiograph_image);
            #radiograph_image = np.expand_dims(radiograph_image, axis=2);
           # radiograph_image = np.repeat(radiograph_image, 3,axis=2);

            full_body_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{radiographs[index]}.png', 
            cv2.IMREAD_GRAYSCALE);
            kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
            full_body_mask = cv2.erode(full_body_mask, kernel, iterations=10);
            full_body_mask = cv2.resize(full_body_mask, (radiograph_image.shape[1], radiograph_image.shape[0]));

            thorax_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{radiographs[index]}.png', cv2.IMREAD_GRAYSCALE);
            
            kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
            thorax_mask = cv2.erode(thorax_mask, kernel, iterations=10);
            thorax_mask = cv2.resize(thorax_mask, (radiograph_image.shape[1], radiograph_image.shape[0]));

            radiograph_image = ((np.where(thorax_mask>0, 1, 0) * radiograph_image)).astype("uint8");

            spine_meta = pickle.load(open(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{radiographs[index]}.meta', 'rb'));
            spine_mask_name = spine_meta['Spine'][-1];
            spine_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\{spine_mask_name}', cv2.IMREAD_GRAYSCALE);
            spine_mask = np.where(spine_mask>0, 255, 0);

            ribs_mask_name = spine_meta['Ribs'][-1];
            ribs_mask = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\Spine and Ribs\\labels\\{ribs_mask_name}', cv2.IMREAD_GRAYSCALE);
            ribs_mask = np.where(ribs_mask>0, 255, 0);

            spine_mask = smooth_boundaries(spine_mask,10);
            spine_mask = smooth_boundaries(spine_mask,25);
            spine_mask = draw_missing_spine(spine_mask);
            spine_mask = scale_width(spine_mask,1.5);

            residual = np.maximum(np.int32(thorax_mask) - np.int32(spine_mask), np.zeros_like(spine_mask)).astype("uint8");
            

            radiograph_image = (np.int32(radiograph_image) * np.where(spine_mask>1, 0, 1)).astype("uint8");
            ret, residual = retarget_img([radiograph_image, spine_mask, thorax_mask], residual);# is it necessary to do residual?
            radiograph_image = ret[0];
            spine_mask = ret[1];
            thorax_mask = ret[2];

            #cv2.imshow('rad', radiograph_image);
            #cv2.waitKey();

            radiograph_image_t = np.expand_dims(radiograph_image, axis=2);
            radiograph_image_t = np.repeat(radiograph_image_t, 3,axis=2);
            
            transformed = config.valid_transforms_seg(image = radiograph_image_t, mask = np.ones_like(radiograph_image));
            radiograph_image_t = transformed["image"];

            pred = self.model(radiograph_image_t.unsqueeze(dim=0).to(config.DEVICE));
            pred = torch.sigmoid(pred) > 0.5;
            pred_np = pred.permute(0,2,3,1).detach().cpu().numpy()[0];
            pred_np = np.uint8(pred_np).squeeze(axis = 2);
            pred_np = pred_np*255;
            pred_np_proc = self.post_process(pred_np);



            radiograph_image = cv2.resize(radiograph_image, (1024,1024));
            spine_mask = cv2.resize(spine_mask.astype("uint8"), (1024, 1024))
            pred_np_proc = cv2.resize(pred_np_proc, (1024,1024));
            ribs_mask = cv2.resize(ribs_mask.astype("uint8"), (1024, 1024))
            full_body_mask = cv2.resize(full_body_mask.astype("uint8"), (1024, 1024))
            thorax_mask = cv2.resize(thorax_mask, (1024, 1024));

            b = cv2.addWeighted(radiograph_image, 0.5, pred_np_proc, 0.5, 0.0);

            cv2.imshow('rad', b);
            cv2.waitKey();

            before = np.sum(np.where(pred_np_proc> 0, 1, 0));
            pred_np_proc = (pred_np_proc * np.where(spine_mask>0, 0, 1)).astype("uint8");
            after = np.sum(np.where(pred_np_proc> 0, 1, 0));
            
        
            s = after/(before+1e-6);
            rat = after / np.sum(np.where(thorax_mask>0, 1, 0));
            #pred_np_proc = (np.int32(pred_np_proc) * np.where(spine_mask>0, 0, 1)).astype("uint8")
            


            gt = lbl_lst[img_lst.index(radiographs[index])];

            # dd = total_test_sternum1[radiographs[index]][0]

            # diff = np.array(dd) - np.array([s, rat]);
            # print(diff);
            

            test_x.append([s, rat]);
            test_y.append(gt);

            # b = cv2.addWeighted(cv2.resize(radiograph_image,(1024,1024)), 0.5, pred_np_proc, 0.5, 0.0);
            # cv2.imwrite(f'{fold_cnt}\\test\\{radiographs[index]}__b.png',b);
            # cv2.imwrite(f'{fold_cnt}\\test\\{radiographs[index]}__pred.png',pred_np_proc);
            # cv2.imwrite(f'{fold_cnt}\\test\\{radiographs[index]}__pred_proc.png',pred_np_proc);
            # #y_test.append(gt_lbl[b].detach().cpu().numpy());
            # cnt += 1;
        return train_x, train_y, test_x, test_y;
    
    def final_results(self):
        tranges = [10,20,30,35,36,37,38,39,40,50,100,200,500,800,1000,1500,2000,2500,3000,4000,5000,10000000];
        best_t = 0;
        best_acc = 0;
        total_pred = [];
        total_gt = [];
        for t in tranges:
            total = [];
            for i in range(5):
                d = pickle.load(open(f'{i}\\test_data.dmp', 'rb'));
                x,y = d[0], d[1];
                correct = 0;
                for i in range(len(x)):
                    if x[i] > t:
                        lbl = 1;
                    else:
                        lbl = 0;
                    
                    total_pred.append(lbl);
                    total_gt.append(y[i]);
                    
                    if lbl == y[i]:
                        correct += 1;
                pr,rec,f1,_ = precision_recall_fscore_support(total_gt, total_pred, average='binary');
                total.append(f1);
                #total_acc.append(correct/len(x));
            avg_f1 = np.mean(total);
            if avg_f1 > best_acc:
                best_acc = avg_f1;
                best_t = t;
            

        print(f'best acc: {best_acc}\tbest_t: {best_t}');


    def post_process(self, pred):
        contours = cv2.findContours(pred, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
        ret = np.zeros_like(pred);
        all_area = [];
        for c in contours:
            all_area.append(cv2.contourArea(c));
        if np.sum(all_area) == 0:
            return ret;
        avg_area = np.mean(all_area);
        for c in contours:
            bbox = cv2.boundingRect(c);
            if cv2.contourArea(c) > avg_area*0.25 and bbox[2]/bbox[3] <1.0:
                ret = cv2.drawContours(ret, [c], 0, (255,255,255), -1);
        return ret;

        #thorax = os.path.join()