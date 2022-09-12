
from copy import deepcopy
from re import S
from statistics import mode
from tabnanny import verbose
import numpy as np
import cv2
import pandas as pd
from glob import glob
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import RobustScaler, StandardScaler
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
from pytorch_grad_cam import EigenGradCAM, FullGrad, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image

IMAGE_SIZE = 512;
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

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

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

def apply_mask(img, mask):
    # simplified_mask = np.zeros_like(mask);
    # contours = cv2.findContours(mask, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0];
    # for c in contours:
    #     x,y,w,h = cv2.boundingRect(c);
    #     simplified_mask = cv2.rectangle(simplified_mask, (x,y), (x+w, y+h), (255,255,255),-1);

    mask = np.where(mask>0, 1, 0);
    img = img * mask;
    

    h,w = img.shape;
    img_row = np.where(img>0);
    first_row = img_row[0][0];
    last_row = img_row[0][-1];

    img_row = np.where(np.transpose(img)>0);

    first_col = img_row[0][0];
    last_col = img_row[0][-1];
    new_img = img[first_row:last_row, first_col:last_col];
    mask = mask[first_row:last_row, first_col:last_col];
    #cv2.imshow('img', new_img.astype("uint8"));
    #cv2.waitKey();
    return new_img,mask;

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
    total_features = [];
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
        os.path.join(f'D:\\PhD\\Thesis\\Segmentation Results\\thorax', f'{file_name}_thorax.png')]);
        
        img = cv2.imread(os.path.join('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f'{file_name}.jpeg'), cv2.IMREAD_GRAYSCALE);
        mask = cv2.imread(os.path.join(f'D:\\PhD\Thesis\\Segmentation Results\\thorax', f'{file_name}_thorax.png'), cv2.IMREAD_GRAYSCALE);
        # heart_mask = cv2.imread(os.path.join(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\heart\\labels', f'{file_name}_0.png'), cv2.IMREAD_GRAYSCALE);
        mask = cv2.resize(mask, (1024,1024));
        img = cv2.resize(img, (1024,1024));
        # heart_mask = cv2.resize(heart_mask, (img.shape[1],img.shape[0]));
        # heart_mask = np.where(heart_mask >0, 1, 0);
        # heart = img*heart_mask;
        # heart_flatten = heart.flatten();
        # heart_thresh = heart_flatten == 0;
        # heart_flatten = np.delete(heart_flatten, heart_thresh); 
        # std = np.std(heart_flatten);
        # snr = signaltonoise(heart_flatten);
        # print(f'std: {std}\tsnr: {snr}\tlabel: {lbl}');
        # cv2.imshow(f'std: {std}\tsnr: {snr}\tlabel: {lbl}', heart.astype("uint8"));
        # cv2.waitKey();

        #b = cv2.addWeighted(img.astype("uint8"), 0.5, heart_mask.astype("uint8")*255, 0.5, 0.0);
        #cv2.imshow('b', b);
        #cv2.waitKey();
        img,mask = apply_mask(img, mask);
        # img_flatten = img.flatten();
        # mask_flatten = mask.flatten();
        # mask_flatten = np.where(mask_flatten<1)[0];
        # img_flatten = np.delete(img_flatten, mask_flatten);
        # std = np.std(img_flatten);
        # snr = signaltonoise(img_flatten);
        # #print(f'std: {std}\tsnr: {snr}\tlabel: {lbl}');
        # #cv2.imshow(f'std: {std}\tsnr: {snr}\tlabel: {lbl}', img.astype("uint8"));
        # #cv2.waitKey();
        
        img = cv2.resize(img.astype("uint8"), (1024,1024));
        mask = cv2.resize(mask.astype("uint8"), (1024,1024));
        # b = cv2.addWeighted(img, 0.5, mask.astype("uint8")*255, 0.5, 0.0);
        # cv2.imshow('img', b);
        # cv2.waitKey();
        # #mask = cv2.threshold(img, thresh=40,maxval=255, type= cv2.THRESH_BINARY)[1];
        # #cv2.imshow('m', mask);
        # #cv2.waitKey();
        hist = cv2.calcHist([img.astype("uint8")], [0], mask.astype("uint8"), [256], [0,255]);
        hist = hist / hist.sum();
        # plt.plot(hist);
        # plt.savefig(f'res\\{file_name}_{lbl}.png');
        # plt.clf();
        # cv2.imwrite(f'res\\{file_name}.png', img);
        total_features.append(hist);

    
    #print(lbl_dict);
    

    le = LabelEncoder();
    total_lbl = le.fit_transform(total_lbl);
    total_lbl = np.array(total_lbl);
    # total_lbl[total_lbl == 'Underexposed'] = 0;
    # total_lbl[total_lbl == 'Normal'] = 0;
    # total_lbl[total_lbl == 'Overexposed'] = 1;
    # total_features = np.array(total_features)
    # total_lbl = np.array(total_lbl, np.int32)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42);

    # total_imgs = np.array(total_imgs);
    # total_features = np.array(total_features);
    # total_prec = [];
    # total_rec = [];
    # total_f1 = [];
    # total_cm = [];

    # for train_id, valid_id in kfold.split(total_features, total_lbl):
    #     train_x, train_y, test_x, test_y = total_features[train_id], total_lbl[train_id], total_features[valid_id], total_lbl[valid_id];
    #     model = Pipeline([('scalar',RobustScaler()), ('svc',SVC(class_weight='balanced', C=1000, gamma=0.001, kernel = 'rbf'))]);
    #     model = model.fit(train_x, train_y);

    #     all_predictions = [];
    #     for i in range(len(test_x)):
    #         lbl = model.predict(test_x[i].reshape(1,-1));
    #         all_predictions.append(lbl[0]);

    #         if lbl[0] != test_y[i]:
    #             print(f'currect lbl: {test_y[i]} prediciton: {lbl}\t{total_imgs[i]}');
        
    #     prec, rec, f1, _ = precision_recall_fscore_support(test_y, all_predictions,average='macro');
    #     cm = confusion_matrix(test_y, all_predictions);
    #     total_f1.append(f1);
    #     total_prec.append(rec);
    #     total_rec.append(prec);
    #     total_cm.append(cm);
    # print(np.mean(total_f1));
    # total_cm = np.array(total_cm);
    # total_cm = np.sum(total_cm, axis = 0);
    # disp = ConfusionMatrixDisplay(total_cm, display_labels=['Underexposed', 'Normal', 'Overexposed']);
    # disp.plot();
    # plt.show();
        



    # param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0];

    # param_grid = [
    #     {'svc__C' : param_range,
    #     'svc__kernel' : ['linear']},
    #     {
    #         'svc__C': param_range,
    #         'svc__gamma' : param_range,
    #         'svc__kernel' : ['rbf']
    #     }
    # ];


    # pipe = Pipeline([('scalar',RobustScaler()), ('svc',SVC(class_weight='balanced'))]);
    # gs = GridSearchCV(pipe, param_grid, scoring='f1', n_jobs=-1, cv = 10);
    # gs = gs.fit(total_features.squeeze(), total_lbl);
    # print(gs.best_params_);

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

    total_imgs = np.array(total_imgs);

    fold_cnt = 1;
    total_f1 = list();

    for train_id, valid_id in kfold.split(total_imgs, total_lbl):
        model.load_state_dict(pickle.load(open(f'{fold_cnt+1}.dmp', 'rb')));
        #model.load_state_dict(init_weights);
        print(f'Starting fold {fold_cnt}...')
        train_X, train_y = total_imgs[train_id], total_lbl[train_id];    
        valid_X, valid_y = total_imgs[valid_id], total_lbl[valid_id];

        train_dataset = ExposureDataset(train_X, train_y, train_transforms);
        valid_dataset = ExposureDataset(valid_X, valid_y, valid_transforms);

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1);
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1);

        for i, (img, lbl) in tqdm(enumerate(valid_loader)):
            img = img.to(DEVICE);
            f = model.features[-1];
            target_layers = [f];
            output = model(img);
            output = torch.argmax(torch.softmax(output, dim = 1), dim=1);
            output = output.detach().cpu().numpy();
            lbl = le.inverse_transform(lbl);
            output = le.inverse_transform(output);

            cam = FullGrad(model=model, target_layers=target_layers, use_cuda=True)

            targets = None;

            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=img, targets=targets)
            rgb_img = img.permute(0,2,3,1).detach().cpu().numpy()[0];
            rgb_img = rgb_img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406];
            rgb_img = (rgb_img*255).astype("uint8");
            rgb_img = rgb_img/255;

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            cam = np.uint8(grayscale_cam*255);
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            cam = cv2.merge([cam, cam, cam])
            images = np.hstack((np.uint8(255*rgb_img), cam , visualization))
            img = Image.fromarray(images)
            img.save(f'cam\\{fold_cnt}\\{i}_{lbl[0]}-{output}_f.png');
            
            #cv2.imwrite(f'res\\{fold_cnt}\\{i}_{lbl[0]}.png', visualization);
        
        fold_cnt += 1;


        # e = 1;
        # best = 100;
        # best_prec = 0;
        # best_recall = 0;
        # best_f1 = 0;
        # best_acc = 0;
        # while(e <= 10):
        #     model.train();
        #     train_step(e, model, train_loader, optimizer);
        #     model.eval();
        #     train_loss, train_acc, train_precision, train_recall, train_f1 = valid_step(e, model, train_loader, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);
        #     valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = valid_step(e, model, valid_loader, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);
    
        #     print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
        #     print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

        #     writer.add_scalar(f'Loss{fold_cnt}/train', train_loss, e);
        #     writer.add_scalar(f'Loss{fold_cnt}/valid', valid_loss, e);

        #     if(valid_loss < best):
        #         print("New best model found!");
        #         best = valid_loss;
        #         best_model = deepcopy(model.state_dict());
        #         best_prec = valid_precision;
        #         best_recall = valid_recall;
        #         best_f1 = valid_f1;
        #         best_acc = valid_acc;

        #     if stopping_strategy(valid_loss, train_loss) is False:
        #         break;
        #     sched.step(e);

        #     e += 1;

        # fold_cnt += 1;
        # f = open(f'res_{fold_cnt}.txt', 'w');
        # f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        # total_f1.append(best_f1);
        # f.close();
        # pickle.dump(best_model, open(f'{fold_cnt}.dmp', 'wb'));
    
    print(f'avg: {np.mean(total_f1)}');
    
