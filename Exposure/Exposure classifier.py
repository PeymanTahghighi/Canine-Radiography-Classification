
from copy import deepcopy
import math
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
from pytorch_grad_cam import EigenGradCAM, FullGrad, EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from scipy.signal import convolve2d
import skimage.exposure as skie
import cython
from torch.optim.lr_scheduler import ReduceLROnPlateau

IMAGE_SIZE = 1024;
DEVICE = 'cuda' if torch.cuda.is_available() else ' cpu';
BATCH_SIZE = 4;

def retarget_img(imgs, mask):

    h,w = mask.shape;
    mask_row = np.where(mask>0);
    first_row = mask_row[0][0];
    last_row = mask_row[0][-1];

    mask_row = np.where(np.transpose(mask)>0);

    first_col = mask_row[0][0];
    last_col = mask_row[0][-1];
    new_mask = mask[first_row:last_row, first_col:last_col];

    ret_img = [];
    for img in imgs:
        new_img = img[first_row:last_row, first_col:last_col];
        ret_img.append(new_img);

    return ret_img, new_mask;

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

overexposure_transforms = A.Compose([
    A.OneOf([
        A.Sequential([
            A.RandomGamma((200.0,500.0), p=1.0),
            A.GaussNoise((50,200), mean = 10, p=0.75)
            ]),      
    ], p = 1.0),
])

underexposure_transforms = A.Compose([
    #A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.Sequential([
            A.RandomGamma((20.0,50.0), p=1.0),
            A.GaussNoise((50,200), mean = 10, p=0.75)
        ]),
        
    ], p = 1.0),
])

train_transforms = A.Compose(
[
    A.HorizontalFlip(p=0.5),
   # A.CLAHE(2,(8,8), always_apply=True, p=1.0),
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
]
)

valid_transforms = A.Compose(
    [
   # A.CLAHE(2,(8,8), always_apply=True, p=1.0),
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
    def __init__(self, imgs, lbls, transforms, train=True) -> None:
        super().__init__()
        self.__imgs = imgs;
        self.__lbls = list(map(int,lbls));
        self.__transforms = transforms;
        self.__train = train;
    def __len__(self):
        return len(self.__imgs);
    
    def __getitem__(self, index):
        img = cv2.imread(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final\\{self.__imgs[index]}.jpeg', cv2.IMREAD_GRAYSCALE);
        lbl = self.__lbls[index];
        
        img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE));
        thorax_mask = cv2.imread(f'thorax\\{self.__imgs[index]}.png', cv2.IMREAD_GRAYSCALE);
        thorax_mask = cv2.resize(thorax_mask, (IMAGE_SIZE,IMAGE_SIZE));
        thorax_mask = np.where(thorax_mask > 0, 1, 0);
        img = (img*thorax_mask).astype("uint8");
        img,thorax_mask = retarget_img([img],thorax_mask);
        img = img[0];
        
        #augment image and alter label if needed
        if int(lbl) == 0 and self.__train == True:
            r = np.random.randint(1,3);
            if r%2 == 0:
                r = np.random.randint(1,3);
                if r%2 == 0:
                    trans = underexposure_transforms(image = img);
                    img = trans['image'];
                    img = (img*thorax_mask).astype("uint8");
                    cv2.imshow('r', img);
                    cv2.waitKey();
                    lbl = 1;
                else:
                    trans = overexposure_transforms(image = img);
                    img = trans['image'];
                    img = (img*thorax_mask).astype("uint8");
                    cv2.imshow('r', img);
                    cv2.waitKey();
                    lbl = 2;
            
        img = np.expand_dims(img, axis= 2);
        img = np.repeat(img, 3, axis = 2);

        img_transformed = self.__transforms(image = img);
        img = img_transformed['image'];

        
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
        output = model(radiograph.float());
        loss = focal_loss(output, lbl, mutual_exclusion=True);
        accumulated_loss.append(loss.item());

        loss.backward();

        if (i+1)% accumulation_step == 0:
            optimizer.step();
            total_loss.append(np.mean(accumulated_loss));
            model.zero_grad(set_to_none = True);
            accumulated_loss = [];


        pbar.set_description(('%10s' + '%10.4g')%(epoch, np.mean(total_loss)));
    return np.mean(total_loss);

def valid_step(epoch, model, loader, estimators):
    total_loss = [];
    total_prec = [];
    total_rec = [];
    total_f1 = [];
    total_acc = [];
    total_gt = [];
    total_pred = [];
    with torch.no_grad():
        pbar = enumerate(loader);
        print(('\n'+'%10s'*6)%('Epoch', 'Loss', 'Prec', 'Recall', 'F1', 'Acc'));
        pbar = tqdm(pbar, total= len(loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}');
        for i, (radiograph, lbl) in pbar:
            radiograph, lbl = radiograph.to(DEVICE), lbl.to(DEVICE);
            output = model(radiograph);
            loss = focal_loss(output, lbl, mutual_exclusion=True);
            total_loss.append(loss.item());
            output = (torch.softmax(output, dim = 1));
            output = torch.argmax(output, dim = 1);
            total_pred.append(output.detach().cpu().numpy()[0]);
            total_gt.append(lbl.detach().cpu().numpy()[0]);

            # prec = estimators[0](output.flatten(), lbl.flatten().long());
            # rec = estimators[1](output.flatten(), lbl.flatten().long());
            # acc = estimators[2](output.flatten(), lbl.flatten().long());
            # f1 = estimators[3](output.flatten(), lbl.flatten().long());
            
            
            # total_prec.append(prec.item());
            # total_rec.append(rec.item());
            # total_f1.append(f1.item());
            # total_acc.append(acc.item());
    
    prec, rec, f1, _ = precision_recall_fscore_support(total_pred, total_gt, average='macro');
    #cm = confusion_matrix(total_gt, total_pred);
    return np.mean(total_loss), 0, prec, rec, f1;

def preload_dataset():

    df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx')
    img_list = list(df['Image']);
    img_list = list(map(str,img_list));
    exp = list(df['Exposure']);
    exp = list(map(int, exp));
    exp = np.array(exp);
    img_list = np.array(img_list);
    return store_folds(img_list, exp);
        
        

    # for img_path in labeled_imgs:
    #     meta = pickle.load(open(img_path,'rb'));
    #     lbl = meta['misc'][0];
    #     if lbl not in lbl_dict:
    #         lbl_dict[lbl] = 1;
    #     else:
    #         lbl_dict[lbl] += 1;
        
    #     total_lbl.append(lbl);
        

    #     file_name = os.path.basename(img_path);
    #     file_name = file_name[:file_name.rfind('.')];
    #     total_imgs.append([os.path.join('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f'{file_name}.jpeg'),
    #     os.path.join(f'D:\\PhD\\Thesis\\Segmentation Results\\thorax', f'{file_name}_thorax.png')]);
        
    #     img = cv2.imread(os.path.join('C:\\Users\Admin\\OneDrive - University of Guelph\\Miscellaneous\\DVVD-Final', f'{file_name}.jpeg'), cv2.IMREAD_GRAYSCALE);
        
        
        #kernel = kernel / 25;
        
    #     mask_meta = pickle.load(open(os.path.join(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels', f'{file_name}.meta'), 'rb'));
    #     p = mask_meta['Spine'][2];
    #     mask = cv2.imread(os.path.join('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels', f'{p}'), cv2.IMREAD_GRAYSCALE);
    #     mask = np.where(mask >0, 255, 0).astype('uint8');
    #     mask_thorax = cv2.imread(os.path.join('D:\\PhD\\Thesis\\Unsupervised-Canine-Radiography-Classification\\results\\train_data', f'{file_name}.png'), cv2.IMREAD_GRAYSCALE);
    #     mask_thorax = cv2.resize(mask_thorax, (img.shape[1], img.shape[0]));
    #     mask_thorax = np.where(mask_thorax > 0, 1, 0);
    #     mask = smooth_boundaries(mask,10);
    #     mask = smooth_boundaries(mask,25);

        
    #     #mask = smooth_boundaries(mask,50);
    #    # mask = scale_width(mask, 2);

    #     # heart_mask = cv2.imread(os.path.join(f'C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\heart\\labels', f'{file_name}_0.png'), cv2.IMREAD_GRAYSCALE);
    #     # cv2.imshow('orig', img);
    #     # cv2.imshow('unsharp', res);
    #     # cv2.waitKey();
    #     # heart_mask = cv2.resize(heart_mask, (img.shape[1],img.shape[0]));
    #     # heart_mask = np.where(heart_mask >0, 1, 0);
    #     # heart = img*heart_mask;
    #     # heart_flatten = heart.flatten();
    #     # heart_thresh = heart_flatten == 0;
    #     # heart_flatten = np.delete(heart_flatten, heart_thresh); 
    #     # std = np.std(heart_flatten);
    #     # snr = signaltonoise(heart_flatten);
    #     # print(f'std: {std}\tsnr: {snr}\tlabel: {lbl}');
    #     # cv2.imshow(f'std: {std}\tsnr: {snr}\tlabel: {lbl}', heart.astype("uint8"));
    #     # cv2.waitKey();

    #     #b = cv2.addWeighted(img.astype("uint8"), 0.5, heart_mask.astype("uint8")*255, 0.5, 0.0);
    #     #cv2.imshow('b', b);
    #     #cv2.waitKey();
    #     img = img*mask_thorax;
    #     img = np.uint8(img);

    #     img_filt = cv2.filter2D(img, -1, kernel);
    #     diff = cv2.subtract(img_filt, img);
    #     #cv2.imshow('orig', img);
    #     #cv2.imshow('filt', img_filt);
    #     s = np.sum(diff);
    #     #print(s);
    #     #cv2.imshow('diff', diff);
    #     #cv2.waitKey();
    #     #cv2.imshow('img', img);
    #     #cv2.waitKey();
    #     noise_variance = estimate_noise(img);
    #     mask = np.where(mask>0, 1, 0);
    #     img = img * mask;

    #     #img = cv2.equalizeHist(img.astype("uint8"));

    #     # res = deepcopy(img);
    #     # for k in range(1):
    #     #     gauss = cv2.GaussianBlur(res, (3,3), 1);
    #     #     res = cv2.addWeighted(res, 1.5,gauss, -0.5, 0 );
        
    #     # img = res;
    #     #img,mask = apply_mask(img, mask);

    #     #cv2.imshow('img', img.astype("uint8"));
    #     #cv2.waitKey();
    #     # img_flatten = img.flatten();
    #     # mask_flatten = mask.flatten();
    #     # mask_flatten = np.where(mask_flatten<1)[0];
    #     # img_flatten = np.delete(img_flatten, mask_flatten);
    #     # std = np.std(img_flatten);
    #     # snr = signaltonoise(img_flatten);
    #     # #print(f'std: {std}\tsnr: {snr}\tlabel: {lbl}');
    #     # #cv2.imshow(f'std: {std}\tsnr: {snr}\tlabel: {lbl}', img.astype("uint8"));
    #     # #cv2.waitKey();
        
    #     #img = cv2.resize(img.astype("uint8"), (1024,1024));
    #     #mask = cv2.resize(mask.astype("uint8"), (1024,1024));
    #     mask_hist = np.where(img > 0, 1, 0);
    #     img_flatten = img.flatten();
    #     img_flatten = np.delete(img_flatten, mask_hist);
    #     std = np.std(img_flatten);
    #     snr = signaltonoise(img_flatten);
    #     print(f'{s}\t lbl: {lbl}\t{file_name}');
    #     img = cv2.resize(img.astype("uint8"), (1024,1024));
    #     #cv2.imshow('img', img.astype("uint8"));
    #     #cv2.waitKey();
    #     # b = cv2.addWeighted(img, 0.5, mask.astype("uint8")*255, 0.5, 0.0);
    #     # cv2.imshow('img', b);
    #     # cv2.waitKey();
    #     # #mask = cv2.threshold(img, thresh=40,maxval=255, type= cv2.THRESH_BINARY)[1];
    #     # #cv2.imshow('m', mask);
    #     # #cv2.waitKey();
    #     # hist = cv2.calcHist([img.astype("uint8")], [0], mask_hist.astype("uint8"), [256], [0,255]);
    #     # hist = hist / hist.sum();
    #     # plt.plot(hist);
    #     # plt.savefig(f'res\\{file_name}_{lbl}.png');
    #     # plt.clf();
        #cv2.imwrite(f'res\\{file_name}_{lbl}.png', img);
        # cv2.imwrite(f'res\\{file_name}_sharp.png', res);
        #total_features.append([std,snr]);

def store_folds(img_list, lbl_list):
    s = StratifiedKFold(5);
    fold_cnt = 0;
    for train_id, test_id in s.split(img_list, lbl_list):
        train_img, train_lbl, test_img, test_lbl = img_list[train_id], lbl_list[train_id], img_list[test_id], lbl_list[test_id];
        pickle.dump([train_img, train_lbl, test_img, test_lbl], open(f'fold_{fold_cnt}.dmp','wb'));
    fold_cnt+=1;

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

def estimate_noise(I):

  H, W = I.shape

  M = [[1, -2, 1],
       [-2, 4, -2],
       [1, -2, 1]]

  sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
  sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

  return sigma

def rescale_range(img, hist, y1,y2):
    ret = deepcopy(img);
    x1 = 0;
    x2 = 255;

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 0:
                ret[i][j] = ((y2-y1)/(x2-x1))*(img[i][j]) + y1;
def gamma_transform(img, gamma):
    ret = deepcopy(img);
    #ret[:,:] = int(((ret[:,:] / 255)**gamma)*255)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 0:
                inp = img[i][j];
                inp /= 255;
                o = inp**gamma;
                o*=255;
                ret[i][j] = int(o);
    
    return ret;

def validate_model(valid_loader):
    for i, (img, lbl) in tqdm(enumerate(valid_loader)):
        img = img.to(DEVICE);
        f = model.features[-1];
        target_layers = [f];
        output = model(img);
        output = torch.argmax(torch.softmax(output, dim = 1), dim=1);
        output = output.detach().cpu().numpy();

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
EPSILON = 1e-6;

def cross_entropy(p,q):
    return np.sum(-p*np.log(q));

def JSD(p,q):
    p = p + EPSILON;
    q = q + EPSILON;
    avg = (p+q)/2;
    jsd = (cross_entropy(p,avg) - cross_entropy(p,p))/2 + (cross_entropy(q,avg) - cross_entropy(q,q))/2;
    #clamp
    if jsd > 1.0:
        jsd = 1.0;
    elif jsd < 0.0:
        jsd = 0.0;
    
    return jsd;

def query(data, q):
    jsd_dict = dict();
    target_hist = data[q][0];
    for h in data.keys():
        if h != q:
            jsd_dict[h] = JSD(target_hist, data[h][0]);
    
    jsd_dict = dict(sorted(jsd_dict.items(), key=lambda item: item[1]))
    jsd_dict_keys = list(jsd_dict.keys());
    for i in range(5):
        print(f'{jsd_dict_keys[i]} : {data[jsd_dict_keys[i]][1]}');
    print(jsd_dict);
    



if __name__ == "__main__":

    # img_path_list = glob('C:\\Users\\Admin\\OneDrive - University of Guelph\Miscellaneous\\DVVD-Final\\*.jpeg');
    # df = pd.read_excel('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\dvvd_list_final.xlsx');
    # exp_list = list(df['Exposure']);
    # img_list = list(map(str, list(df['Image'])));

    # all_data = dict();
    # for img_path in img_path_list:
    #     file_name = os.path.basename(img_path);
    #     file_name = file_name[:file_name.rfind('.')];
    #     if file_name in img_list:
    #         img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
    #         thorax_mask = cv2.imread(f'thorax\\{file_name}.png', cv2.IMREAD_GRAYSCALE);
    #         img = cv2.resize(img, (1024,1024));
    #         hist = cv2.calcHist([img],[0], thorax_mask, [256], [0,256]);
    #         hist = hist / np.sum(hist);
    #         all_data[file_name] = [hist, exp_list[img_list.index(file_name)]];
    
    # pickle.dump(all_data, open('hist.dmp', 'wb'));

    #all_data = pickle.load(open('hist.dmp', 'rb'));
    #query(all_data, '668');


    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True);
    model.classifier.fc =  nn.Linear(1792, 3, bias=True);
    model = model.to(DEVICE);
    init_weights = deepcopy(model.state_dict());
    optimizer = optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5);
    sched = ReduceLROnPlateau(optimizer,threshold=1e-3, factor = 0.5, verbose=True);
    writer = SummaryWriter('exp');

    precision_estimator = Precision(num_classes=3, multiclass=True, average='macro').to(DEVICE);
    recall_estimator = Recall(num_classes=3, multiclass=True , average='macro').to(DEVICE);
    accuracy_esimator = Accuracy(num_classes=3, multiclass=True, average='macro').to(DEVICE);
    f1_esimator = F1Score(num_classes=3, multiclass=True, average='macro').to(DEVICE);

    stopping_strategy = CombinedTrainValid(1.0,2);

    fold_lst = glob('C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\cache\\*.fold');
    folds = [];
    for f in fold_lst:
        folds.append(pickle.load(open(f, 'rb')));
    fold_cnt = 0;
    total_f1 = [];
    for idx in range(5):
        train_imgs,train_mask,train_lbl, train_grain_lbl, cranial_features, \
            caudal_features, \
            tips_features,\
            sternum_features, test_imgs, test_mask, test_lbl, test_grain_lbl = folds[idx][0], \
            folds[idx][1], \
            folds[idx][2],\
            folds[idx][3], \
            folds[idx][4], \
            folds[idx][5], \
            folds[idx][6], \
            folds[idx][7], \
            folds[idx][8], \
            folds[idx][9], \
            folds[idx][10], \
            folds[idx][11];
        
        

        overexposure_transforms = A.Compose([
            A.OneOf([
                A.Sequential([
                    A.RandomGamma((200.0,500.0), p=1.0),
                    A.GaussNoise((50,200), mean = 10, p=0.75)
                    ]),      
            ], p = 1.0),
        ])

        underexposure_transforms = A.Compose([
            A.OneOf([
                A.Sequential([
                    A.RandomGamma((20.0,50.0), p=1.0),
                    A.GaussNoise((50,200), mean = 10, p=0.75)
                ]),
                
            ], p = 1.0),
        ])
            
        model.load_state_dict(init_weights);
        print(f'=========Starting fold {fold_cnt}=========')

        train_dataset = ExposureDataset(train_imgs, train_grain_lbl[:,-1], train_transforms);
        valid_dataset = ExposureDataset(test_imgs, test_grain_lbl[:, -1], valid_transforms, train=False);

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1);
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1);

        e = 1;
        best = 100;
        best_prec = 0;
        best_recall = 0;
        best_f1 = 0;
        best_acc = 0;
        while(True):
            model.train();
            train_loss = train_step(e, model, train_loader, optimizer);
            model.eval();
            #train_loss, train_acc, train_precision, train_recall, train_f1 = valid_step(e, model, train_loader, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);
            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = valid_step(e, model, valid_loader, [precision_estimator, recall_estimator, accuracy_esimator, f1_esimator]);

            print(('\n'+'%10s'*7)%('Epoch', 'Train', 'Valid', 'Precision', 'Recall', 'Accuracy', 'F1'));
            print(('\n'+'%10.4g'*7)%(e, train_loss, valid_loss, valid_precision, valid_recall, valid_acc, valid_f1));

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
                f = open(f'res_{fold_cnt}.txt', 'w');
                f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
                f.close();
                pickle.dump(best_model, open(f'{fold_cnt}.dmp', 'wb'));

            # if stopping_strategy(valid_loss, train_loss) is False:
            #     break;
            sched.step(e);

            e += 1;

        f = open(f'res_{fold_cnt}.txt', 'w');
        f.write(f"Valid \tPrecision: {best_prec}\tRecall: {best_recall}\tAccuracy: {best_acc}\tF1: {best_f1}");
        total_f1.append(best_f1);
        f.close();
        pickle.dump(best_model, open(f'{fold_cnt}.dmp', 'wb'));
        fold_cnt += 1;
    
    print(f'avg: {np.mean(total_f1)}');
    
