import pickle
from re import L
from sklearn.utils import shuffle
from numpy.core.fromnumeric import mean
from numpy.lib.npyio import load
from torch.nn.modules.loss import L1Loss
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
import config
from network_dataset import SternumDataset
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


class NetworkTrainer():

    def __init__(self):
        self.__initialize();
        pass

    #This function should be called once the program starts
    def __initialize(self,):

        self.model = Unet(1).to(config.DEVICE);
        self.init_weights = deepcopy(self.model.state_dict());
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.precision_estimator = Precision(num_classes=1, multiclass=False).to(config.DEVICE);
        self.recall_estimator = Recall(num_classes=1, multiclass=False).to(config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=1, multiclass=False).to(config.DEVICE);
        self.f1_esimator = F1Score(num_classes=1, multiclass=False).to(config.DEVICE);

        pass

    def __loss_func(self, output, gt):
        f_loss = focal_loss(output, gt,  arange_logits=True);
        t_loss = tversky_loss(output, gt, sigmoid=True, arange_logits=True)
        return  t_loss + f_loss;
        


    def __train_one_epoch(self, epoch, loader, model, optimizer):
        epoch_loss = [];
        step = 0;
        update_step = 1;
        pbar = enumerate(loader);
        print(('\n' + '%10s'*2) %('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        for i, (radiograph, mask, gt_lbl) in pbar:
            radiograph, mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE)
            model.zero_grad(set_to_none = True);
            # radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);
            # radiograph_np = radiograph.permute(0,2,3,1).cpu().detach().numpy();
            # radiograph_np = radiograph_np[0][:,:,1];
            # radiograph_np *= 0.229;
            # radiograph_np += 0.485;
            # radiograph_np *= 255;
            
            #cv2.imshow('radiograph', radiograph_np.astype("uint8"));
            #cv2.waitKey();
            # mask_np = mask.cpu().detach().numpy();
            # mask_np = mask_np[0];
            
            # radiograph_np = radiograph_np*0.5+0.5;
            # plt.figure();
            # plt.imshow(radiograph_np[0]);
            # plt.waitforbuttonpress();

            # cv2.imshow('mask', mask_np.astype("uint8")*255);
            # cv2.waitKey();

            # plt.figure();
            # plt.imshow(mask[0]*255);
            # plt.waitforbuttonpress();
                

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
        total_correct_classification = [];

        pbar = enumerate(loader);
        print(('\n' + '%10s'*7) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc', 'Class_Acc'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        with torch.no_grad():
            for i ,(radiograph, mask, gt_lbl) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = model(radiograph);
                loss = self.__loss_func(pred, mask);

                epoch_loss.append(loss.item());
                
                pred = (torch.sigmoid(pred)) > 0.5;

                positives = torch.sum(pred == 1,[1,2,3]);
                for b in range((pred.shape[0])):
                    if positives[b] > 1000:
                        lbl = 1;
                    else:
                        lbl = 0;
                    
                    if lbl == gt_lbl[b]:
                        total_correct_classification.append(1);
                    else:
                        total_correct_classification.append(0);

                prec = self.precision_estimator(pred.flatten(), mask.flatten().long());
                rec = self.recall_estimator(pred.flatten(), mask.flatten().long());
                acc = self.accuracy_esimator(pred.flatten(), mask.flatten().long());
                f1 = self.f1_esimator(pred.flatten(), mask.flatten().long());
                
                
                total_prec.append(prec.item());
                total_rec.append(rec.item());
                total_f1.append(f1.item());
                total_acc.append(acc.item());

                pbar.set_description(('%10s' + '%10.4g'*6) % (epoch, np.mean(epoch_loss),
                np.mean(total_prec), np.mean(total_rec), np.mean(total_f1), np.mean(total_acc), np.mean(total_correct_classification)))

        return np.mean(epoch_loss), np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1), np.mean(total_correct_classification);


    def train(self, fold_cnt, train_data, test_data):

        self.model.load_state_dict(self.init_weights);
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5);

        stopping_strategy = CombinedTrainValid(1.0,5);

        train_dataset = SternumDataset(train_data[0], train_data[1], train_data[2], config.train_transforms);
        valid_dataset = SternumDataset(test_data[0], test_data[1], test_data[2], config.valid_transforms);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

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
            self.__train_one_epoch(e, train_loader,self.model, optimizer);

            self.model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1, train_class_acc = self.__eval_one_epoch(e, train_loader, self.model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_class_acc = self.__eval_one_epoch(e, valid_loader, self.model);

            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}\tclass_acc: {train_class_acc}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}\tclass_acc: {valid_class_acc}");


            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(self.model.state_dict());
                best_acc = valid_acc;
                best_prec = valid_precision;
                best_f1 = valid_f1;
                best_recall = valid_recall;
                best_class_acc = valid_class_acc;

            if stopping_strategy(valid_loss, train_loss) is False:
                break;
            e += 1;
        f = open(f'res{fold_cnt}.txt', 'w');
        f.write(f"Precision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}\tclass_acc: {best_class_acc}");
        f.close();
        pickle.dump(best_model, open(f'ckpt{fold_cnt}.pt', 'wb'));
        self.eval(fold_cnt, f'ckpt{fold_cnt}.pt', test_data);
    

    def store_results(self, fold_cnt, test_data, train_data):

        if os.path.exists(f'{fold_cnt}\\test') is False:
            os.makedirs(f'{fold_cnt}\\test');
        

        self.model.load_state_dict(pickle.load(open(f'ckpt{fold_cnt}.pt','rb')));

        valid_dataset = SternumDataset(test_data[0], test_data[1], test_data[2], config.valid_transforms, True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

        #while(True):
        pbar = enumerate(valid_loader);
        print(('\n' + '%10s'*7) %('Epoch', 'Loss', 'Prec', 'Rec', 'F1', 'Acc', 'Class_Acc'));
        pbar = tqdm(pbar, total= len(valid_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        
        x_test = [];
        y_test = [];
        x_train = [];
        y_train = [];
        cnt = 0;
        with torch.no_grad():
            for i ,(radiograph, mask, gt_lbl, file_name) in pbar:
                radiograph,mask = radiograph.to(config.DEVICE), mask.to(config.DEVICE);

                pred = self.model(radiograph);
                pred = torch.sigmoid(pred) > 0.5;
                pred_np = pred.permute(0,2,3,1).detach().cpu().numpy();
                pred_np = np.uint8(pred_np).squeeze(axis = 3);

                for i in range(pred_np.shape[0]):
                    ret = self.post_process(pred_np[i], file_name[i]);
                    pred_np[i] = ret;

                positives = np.sum(pred_np, (1,2));

                x_test.extend(positives);
                

                radiograph_np = radiograph.permute(0,2,3,1).detach().cpu().numpy();
                radiograph_np = radiograph_np *  [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406];
                radiograph_np = np.uint8(radiograph_np*255);
                for b in range((pred.shape[0])):
                    cv2.imwrite(f'{fold_cnt}\\test\\{cnt}_{gt_lbl[b]}.png', radiograph_np[b]);
                    cv2.imwrite(f'{fold_cnt}\\test\\{cnt}_{gt_lbl[b]}_seg.png', pred_np[b]*255);
                    y_test.append(gt_lbl[b].detach().cpu().numpy());
                    cnt += 1;


        

        pickle.dump([x_test,y_test], open(f'{fold_cnt}\\test_data.dmp','wb'));
        #pickle.dump([x_train,y_train], open(f'{fold_cnt}\\train_data.dmp','wb'));


        
        # plt.scatter(all_data, all_lbl);
        # plt.show();
    
   
            

        
        # plt.scatter(all_data, all_lbl);
        # plt.show();
    
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


    def post_process(self, pred, file_name):

        #remove outside of thorax predictions
        thorax_img = cv2.imread(os.path.join('C:\\PhD\\Thesis\\Tests\\Segmentation Results\\thorax', f'{file_name}_thorax.png'), cv2.IMREAD_GRAYSCALE);
        thorax_img = cv2.resize(thorax_img,(1024,1024));
        thorax_img = np.where(thorax_img > 1, 1, 0);
        thorax_sternum = np.logical_and(pred.squeeze(), thorax_img).astype(np.uint8);


        w,h = thorax_sternum.shape;
        ret = np.zeros_like(thorax_sternum);
        contours = cv2.findContours(thorax_sternum, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
        for c in contours:
            area = cv2.contourArea(c) / (w*h);

        return thorax_sternum;

        #thorax = os.path.join()
