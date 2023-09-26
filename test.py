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
from deep_learning.model_trainer import NetworkTrainer
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
import matplotlib.pyplot as plt

#---------------------------------------------------------
def load_folds():
    fold_lst = glob('cache\\*.fold');
    folds = [];
    for f in fold_lst:
        folds.append(pickle.load(open(f, 'rb')));

    return folds;
#---------------------------------------------------------

def sliding_window_inferrer(model, batch, mask, step = 200):

    b,_,h,w = batch.shape;
    ribs_total_output = torch.zeros((h,w));
    spine_total_output = torch.zeros((h,w));
    patch_pred_counter = torch.zeros((h,w));
    debug_output = False;
    
    start_h = 0;
    start_w = 0;
    while(True):
        end_h = min(start_h+config.CROP_SIZE, h);
        end_w = min(start_w+config.CROP_SIZE, w);
        cropped_batch = batch[:,:,start_h:end_h, start_w: end_w];
        if end_h - start_h < config.CROP_SIZE or end_w - start_w < config.CROP_SIZE:
            padded_batch = torch.zeros((1, 3, config.CROP_SIZE, config.CROP_SIZE));
            padded_batch[:, :, :end_h - start_h, :end_w - start_w] = cropped_batch;
            cropped_batch = padded_batch.to(config.DEVICE);
        output = model(cropped_batch);
        output = torch.argmax(torch.softmax(output, dim = 1), dim = 1);
        ribs_output = (output == 1).squeeze().cpu();
        spine_output = (output == 2).squeeze().cpu();
        if debug_output:
            cropped_batch_np = cropped_batch.permute(0,2,3,1).detach().cpu().numpy()[0];
            ribs_output_np = ribs_output.permute(1,2,0).detach().cpu().numpy();
            spine_output_np = spine_output.permute(1,2,0).detach().cpu().numpy();
            fig, ax = plt.subplots(1,3);
            ax[0].imshow(cropped_batch_np, cmap='gray');
            ax[1].imshow(ribs_output_np, cmap='gray');
            ax[2].imshow(spine_output_np, cmap='gray');
            plt.show();

        ribs_total_output[start_h:end_h, start_w: end_w] += ribs_output[:end_h-start_h, :end_w - start_w];
        spine_total_output[start_h:end_h, start_w: end_w] += spine_output[:end_h-start_h, :end_w - start_w];
        patch_pred_counter[start_h:end_h, start_w: end_w]+=1;

        if debug_output:
            ribs_total_output_np = ribs_total_output.detach().cpu().numpy();
            spine_total_output_np = spine_total_output.detach().cpu().numpy();
            fig, ax = plt.subplots(1,2);
            ax[0].imshow(spine_total_output_np, cmap='gray');
            ax[1].imshow(ribs_total_output_np, cmap='gray');
            plt.show();
        
        if h > start_h+config.CROP_SIZE:
            start_h+=step;
        elif h <= start_h+config.CROP_SIZE and w <= start_w+config.CROP_SIZE:
            break;
        else:
            start_w+=step;
            start_h = 0;
    ribs_total_output = ribs_total_output>(patch_pred_counter/2);
    spine_total_output = spine_total_output>(patch_pred_counter/2);
    pred = torch.zeros_like(spine_total_output, dtype=torch.int32);
    a = ribs_total_output==1
    pred[ribs_total_output==1] = 1;
    pred[spine_total_output==1] = 2;

    prec = precision_estimator(pred.to(config.DEVICE).flatten(), mask.flatten().long());
    rec = recall_estimator(pred.to(config.DEVICE).flatten(), mask.flatten().long());
    f1 = f1_esimator(pred.to(config.DEVICE).flatten(), mask.flatten().long());
    
    if debug_output:
        ribs_total_output = ribs_total_output.cpu().numpy();
        spine_total_output = spine_total_output.cpu().numpy();
        fig, ax = plt.subplots(1,2);
        ax[0].imshow(ribs_total_output, cmap='gray');
        ax[1].imshow(spine_total_output, cmap='gray');
        plt.show();

    return prec, rec, f1;

sd1 = pickle.load(open('C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\0\\spine and ribs.pt', 'rb'));
sd2 = pickle.load(open('C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\spine and ribs.pt', 'rb'));
for k in sd1.keys():
    diff = sd1[k] - sd2[k];
    if torch.sum(diff) !=0:
        print(k);

folds = load_folds();


total_cranial = [];
total_caudal = [];
total_symmetry = [];
total_sternum = [];
total_quality = [];
total_tips= [];


start_fold = 0;


spine_and_ribs_segmentation_model = Unet(3);
ckpt = torch.load('C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\Tune_Epochs\\results\\0\\spine and ribs_best.ckpt')['model'];
spine_and_ribs_segmentation_model.load_state_dict(ckpt);
spine_and_ribs_segmentation_model.eval();
spine_and_ribs_segmentation_model.to(config.DEVICE);

tune_results = [];

cur_exp_results = [];
for idx in range(start_fold,len(folds)):
    train_imgs,train_mask,train_lbl, train_grain_lbl, _, \
    _, \
    _,\
    _, test_imgs, test_mask, test_lbl, test_grain_lbl = folds[idx][0], folds[idx][1], folds[idx][2], folds[idx][3], folds[idx][4], folds[idx][5], folds[idx][6], folds[idx][7], folds[idx][8], folds[idx][9], folds[idx][10], folds[idx][11];

    print(f'\n================= Starting fold {idx} =================\n');

    num_classes = 3;
    dataset = CanineDataset(test_imgs, test_mask[:,0], train=False, transforms=config.test_transforms);
    loader = DataLoader(dataset, 1);
    precision_estimator = Precision('binary' if num_classes ==1 else 'multiclass', num_classes=num_classes, average='macro').to(config.DEVICE);
    recall_estimator = Recall('binary' if num_classes ==1 else 'multiclass', num_classes=num_classes, average='macro').to(config.DEVICE);
    f1_esimator = F1Score('binary' if num_classes ==1 else 'multiclass', num_classes=num_classes, average='macro').to(config.DEVICE);
    
    total_prec = [];
    total_rec = [];
    total_f1 = [];
    #(2-1)
    print('------------- Inferring spine and ribs model ---------------\n');
    pbar = enumerate(loader);
    print(('\n' + '%10s'*3) %('Prec', 'Rec', 'F1'));
    pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    for img, (img, msk) in pbar:
        img, msk = img.to(config.DEVICE), msk.to(config.DEVICE);
        with torch.no_grad():
            prec, rec, f1 = sliding_window_inferrer(spine_and_ribs_segmentation_model, img, msk);
            total_prec.append(prec.item());
            total_rec.append(rec.item());
            total_f1.append(f1.item());

            pbar.set_description(('%10.4g'*3) % (
            np.mean(total_prec), np.mean(total_rec), np.mean(total_f1)))