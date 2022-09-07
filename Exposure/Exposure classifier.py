
from copy import deepcopy
from re import S
from tabnanny import verbose
import numpy as np
import cv2
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pytorchvideo
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits, cross_entropy
from torchmetrics import *
from stopping_strategy import CombinedTrainValid
import pickle
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter


IMAGE_SIZE = 1536;
DEVICE = 'cuda' if torch.cuda.is_available() else ' cpu';

def focal_loss(logits,
                true,
                alpha = 0.8,
                gamma = 2.0,
                arange_logits = False,
                mutual_exclusion = False):

    if mutual_exclusion is False:
        if arange_logits is True:
            logits = logits.permute(0,2,3,1);
        
        bce = binary_cross_entropy_with_logits(logits.squeeze(dim=1), true.float(), reduction='none');
        bce_exp = torch.exp(-bce);
        f_loss = torch.mean(alpha * (1-bce_exp)**gamma*bce);
        return f_loss;
    else:
        #logits = logits.permute(0,2,3,1).reshape(-1,3);
        true = true.view(-1);
        ce_loss = cross_entropy(logits, true.long(), reduction='none');
        p = torch.softmax(logits, axis = 1);
        #true_one_hot = one_hot(true.long(), Config.NUM_CLASSES);
        p = torch.take_along_dim(p, true.long().unsqueeze(dim = 1), dim = 1).squeeze();
        #p = torch.index_select(p, dim = 3, index = true);
        #assuming true is a one hot vector
        #ce_loss = -torch.log(p * true_one_hot + 1e-6);
        focal_mul = (1-p)**gamma;
        f_loss = focal_mul * ce_loss;
        return torch.mean(f_loss);

train_transforms = A.Compose(
[
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    # A.PadIfNeeded(min_height=512, min_width=512),
    # A.CenterCrop(p=0.5, height = 512, width = 512),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)

valid_transforms = A.Compose(
    [
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)

def retarget_img(img):
    h,w = img.shape;
    img_row = np.where(img>0);
    first_row = img_row[0][0];
    last_row = img_row[0][-1];

    img_row = np.where(np.transpose(img)>0);

    first_col = img_row[0][0];
    last_col = img_row[0][-1];
    new_img = img[first_row:last_row, first_col:last_col];
    return new_img;

class ExposureDataset(Dataset):
    def __init__(self, imgs, lbls, transforms) -> None:
        super().__init__()
        self.__imgs = imgs;
        self.__lbls = lbls;
        self.__transforms = transforms;

    def __len__(self):
        return len(self.__imgs);
    
    def __getitem__(self, index):
        img = cv2.imread(self.__imgs[index][0], cv2.IMREAD_GRAYSCALE);
        img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE));
        thorax = cv2.imread(self.__imgs[index][1], cv2.IMREAD_GRAYSCALE);
        thorax = cv2.resize(thorax, (IMAGE_SIZE,IMAGE_SIZE));
        thorax = np.where(thorax > 0, 1, 0);
        img = (img*thorax).astype("uint8");
        img = retarget_img(img);
        img = np.expand_dims(img, axis= 2);
        img = np.repeat(img, 3, axis = 2);

        img_transformed = self.__transforms(image = img);
        img = img_transformed['image'];

        lbl = self.__lbls[index];
        
        return img, lbl

def train_step(epoch, model, loader, optimizer):
    total_loss = [];
    pbar = enumerate(loader);
    print(('\n'+'%10s'*2)%('Epoch', 'Loss'));
    l = len(loader);
    pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
    accumulation_step = 1;
    accumulated_loss = [];
    for i, (radiograph, lbl) in pbar:
        radiograph, lbl = radiograph.to(DEVICE), lbl.to(DEVICE);
        output = model(radiograph);
        loss = focal_loss(output, lbl, mutual_exclusion=True);
        accumulated_loss.append(loss.item());

        loss.backward();

        if (i+1)% accumulation_step == 0:
            optimizer.step();
            total_loss.append(np.mean(accumulated_loss));
            model.zero_grad(set_to_none = True);
            accumulated_loss = [];


        pbar.set_description(('%10s' + '%10.4g')%(epoch, np.mean(total_loss)));

def valid_step(epoch, model, loader, estimators):
    total_loss = [];
    total_prec = [];
    total_rec = [];
    total_f1 = [];
    total_acc = [];
    with torch.no_grad():
        pbar = enumerate(loader);
        print(('\n'+'%10s'*2)%('Epoch', 'Loss'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
        for i, (radiograph, lbl) in pbar:
            radiograph, lbl = radiograph.to(DEVICE), lbl.to(DEVICE);
            output = model(radiograph);
            loss = focal_loss(output, lbl, mutual_exclusion=True);
            total_loss.append(loss.item());
            output = (torch.softmax(output, dim = 1));
            output = torch.argmax(output, dim = 1);

            prec = estimators[0](output.flatten(), lbl.flatten().long());
            rec = estimators[1](output.flatten(), lbl.flatten().long());
            acc = estimators[2](output.flatten(), lbl.flatten().long());
            f1 = estimators[3](output.flatten(), lbl.flatten().long());
            
            
            total_prec.append(prec.item());
            total_rec.append(rec.item());
            total_f1.append(f1.item());
            total_acc.append(acc.item());

            pbar.set_description(('%10s' + '%10.4g'*5) % (epoch, np.mean(total_loss),
            np.mean(total_prec), np.mean(total_rec), np.mean(total_f1), np.mean(total_acc)))
    
    return np.mean(total_loss), np.mean(total_acc), np.mean(total_prec), np.mean(total_rec), np.mean(total_f1);


def get_contrast(img):
    kernel = np.ones((5,5), np.uint8);

    min = cv2.erode(img, kernel, iterations=1);
    max = cv2.dilate(img, kernel, iterations=1);

    div_zero_avoid = np.ones(shape=max.shape, )

    contrast = (max - min) / (max + min + div_zero_avoid);
    contrast = np.mean(contrast);

    return contrast;

def get_std(img):
    return img.std();

if __name__ == "__main__":
    
    labeled_imgs = glob('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Exposure\\*.meta');

    lbl_dict = dict();
    total_lbl = [];
    total_imgs = [];
    for img_path in labeled_imgs:
        meta = pickle.load(open(img_path,'rb'));
        lbl = meta['misc'][0];
        if lbl not in lbl_dict:
            lbl_dict[lbl] = 1;
        else:
            lbl_dict[lbl] += 1;
        
        total_lbl.append(lbl);
        

        file_name = os.path.basename(img_path);
        file_name = file_name[:file_name.rfind('.')];
        total_imgs.append([os.path.join('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f'{file_name}.jpeg'),
        os.path.join(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\Segmentation Results\\thorax', f'{file_name}_thorax.png')]);
        
        # img = cv2.imread(os.path.join('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f'{file_name}.jpeg'), cv2.IMREAD_GRAYSCALE);
        # mask = cv2.threshold(img, thresh=40,maxval=255, type= cv2.THRESH_BINARY)[1];
        # #cv2.imshow('m', mask);
        # #cv2.waitKey();
        # hist = cv2.calcHist([img], [0], mask, [256], [0,255]);
        # hist = hist / hist.sum();
        # plt.plot(hist);
        # plt.savefig(f'res\\{file_name}_{lbl}.png');
        # plt.clf();
        # cv2.imwrite(f'res\\{file_name}.png', img);
        # total_features.append(hist);

    
    #print(lbl_dict);
    

    le = LabelEncoder();
    total_lbl = le.fit_transform(total_lbl);
    
    kfold = StratifiedKFold(n_splits=5);

    total_imgs = np.array(total_imgs);
    total_lbl = np.array(total_lbl);

    # param_args = {
    #     'svc__C':[0.001,0.01,0.1,1,10,100],
    #     'svc__kernel': ['rbf', 'linear']
    # };

    # pipe = Pipeline([('scalar',StandardScaler()), ('svc',SVC(class_weight='balanced'))]);
    # gs = GridSearchCV(pipe, param_args, scoring='f1_macro', n_jobs=-1, cv = 5);
    # gs = gs.fit(total_features.squeeze(), total_lbl);
    # print(gs.best_score_);

    settings = [{''}];

    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True);
    model.classifier.fc =  nn.Linear(1792, 3, bias=True);
    model = model.to(DEVICE);
    init_weights = deepcopy(model.state_dict());
    optimizer = optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5);
    sched = ExponentialLR(optimizer,0.99, verbose=True);
    writer = SummaryWriter('exp');

    precision_estimator = Precision(num_classes=3, multiclass=True, average='macro').to(DEVICE);
    recall_estimator = Recall(num_classes=3, multiclass=True , average='macro').to(DEVICE);
    accuracy_esimator = Accuracy(num_classes=3, multiclass=True, average='macro').to(DEVICE);
    f1_esimator = F1Score(num_classes=3, multiclass=True, average='macro').to(DEVICE);

    stopping_strategy = CombinedTrainValid(1.0,5);

    fold_cnt = 0;
    total_f1 = list();

    for train_id, valid_id in kfold.split(total_imgs, total_lbl):
        model.load_state_dict(init_weights);
        print(f'Starting fold {fold_cnt}...')
        train_X, train_y = total_imgs[train_id], total_lbl[train_id];    
        valid_X, valid_y = total_imgs[valid_id], total_lbl[valid_id];

        train_dataset = ExposureDataset(train_X, train_y, train_transforms);
        valid_dataset = ExposureDataset(valid_X, valid_y, valid_transforms);

        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2);
        valid_loader = DataLoader(valid_dataset, batch_size=2, shuffle=True, num_workers=2);

        e = 1;
        best = 100;
        best_prec = 0;
        best_recall = 0;
        best_f1 = 0;
        best_acc = 0;
        while(True):
            model.train();
            train_step(e, model, train_loader, optimizer);
            model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = valid_step(e, model, train_loader, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);
            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = valid_step(e, model, valid_loader, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);
    
            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

            writer.add_scalar(f'Loss{fold_cnt}/train', train_loss, e);
            writer.add_scalar(f'Loss{fold_cnt}/valid', valid_loss, e);

            if(valid_loss < best):
                print("New best model found!");
                best = valid_loss;
                best_model = deepcopy(model.state_dict());
                best_prec = valid_precision;
                best_recall = valid_recall;
                best_f1 = valid_f1;
                best_acc = valid_acc;

            if stopping_strategy(valid_loss, train_loss) is False:
                break;
            sched.step(e);

            e += 1;

        fold_cnt += 1;
        f = open(f'res_{fold_cnt}.txt', 'w');
        f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        total_f1.append(best_f1);
        f.close();
        pickle.dump(best_model, open(f'{fold_cnt}.dmp', 'wb'));
    
    print(f'avg: {np.mean(total_f1)}');
    
