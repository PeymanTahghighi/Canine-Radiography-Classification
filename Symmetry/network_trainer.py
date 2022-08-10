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
from utility import divide_image_symmetry_line, get_symmetry_line, remove_blobs, remove_blobs_spine
from thorax import segment_thorax


class NetworkTrainer():

    def __init__(self):
        self.__initialize();
        pass

    #This function should be called once the program starts
    def __initialize(self,):

        self.model = Unet(3).to(config.DEVICE);
        self.init_weights = deepcopy(self.model.state_dict());
        self.scaler = torch.cuda.amp.grad_scaler.GradScaler();
        self.precision_estimator = Precision(num_classes=3).to(config.DEVICE);
        self.recall_estimator = Recall(num_classes=3).to(config.DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=3).to(config.DEVICE);
        self.f1_esimator = F1Score(num_classes=3).to(config.DEVICE);

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


    def train(self, fold_cnt, train_data, test_data):

        self.model.load_state_dict(self.init_weights);
        optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5);

        stopping_strategy = CombinedTrainValid(0.7,2);

        train_dataset = SternumDataset(train_data[0], train_data[1], config.train_transforms);
        valid_dataset = SternumDataset(test_data[0], test_data[1], config.valid_transforms);

        train_loader = DataLoader(train_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True);

        valid_loader = DataLoader(valid_dataset, 
        batch_size= config.BATCH_SIZE, shuffle=False);

        best = 100;
        e = 1;
        best_model = None;
        best_prec = 0;
        best_recall = 0;
        best_f1 = 0;
        best_acc = 0;

        while(True):
            self.model.train();
            self.__train_one_epoch(e, train_loader,self.model, optimizer);

            self.model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.__eval_one_epoch(e, train_loader, self.model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(e, valid_loader, self.model);

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
        pickle.dump(best_model, open(f'ckpt{fold_cnt}.pt', 'wb'));

        #load model with best weights to save outputs
        self.model.load_state_dict(best_model);
        self.save_samples(fold_cnt, test_data[0]);
    

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
    
    def save_samples(self, fold_cnt, test_img):
        if os.path.exists(f'{fold_cnt}'):
            for f in os.listdir(f'{fold_cnt}'):
                os.remove(f'{fold_cnt}\\{f}');
        else:
            os.mkdir(f'{fold_cnt}');
        
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





