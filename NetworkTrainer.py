import torch
import torch.nn as nn
import torch.optim as optim
import Config
from NetworkDataset import NetworkDataset
from torch.utils.data import DataLoader, SubsetRandomSampler, sampler
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import os
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F
import Config
import logging
from Network import *
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from StoppingStrategy import CombinedTrainValid, GeneralizationLoss, StoppingStrategy
from utils import *
from sklearn.model_selection import train_test_split
from torchmetrics import *
from torch.utils.data import SubsetRandomSampler
from Strategy import *

class NetworkTrainer():

    def __init__(self):
        super().__init__();
        self.__initialize();
        pass

    #This function should be called once the program starts
    def __initialize(self,):
        self.model = Resnet(num_classes=3).to(Config.DEVICE);
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2);

        self.precision_estimator = Precision(num_classes=3, average='macro').to(DEVICE);
        self.recall_estimator = Recall(num_classes=3, average='macro').to(DEVICE);
        self.accuracy_esimator = Accuracy(num_classes=3,average='macro').to(DEVICE);
        self.f1_esimator = F1(num_classes=3, average='macro').to(DEVICE);

        self.scaler = torch.cuda.amp.grad_scaler.GradScaler()

        self.l1_loss = nn.L1Loss().to(Config.DEVICE);
        
        #Initialize transforms for training and validation
        self.train_transforms = A.Compose(
            [
                A.Resize(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ToTensorV2(),
            ],
            additional_targets={'mask': 'mask'}
        )

        self.valid_transforms = A.Compose(
                [
                #A.PadIfNeeded(min_height = 512, min_width = 512),
                #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = True),
                A.Resize(Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ToTensorV2()
                ]
        )

        self.__dataset_loader = DatasetLoader("C:\\PhD\\Miscellaneous\\vet2");
        
        pass

    #This function should be called everytime we want to train a new model
    def initialize_new_train(self):
        
        self.writer = SummaryWriter('experiments');

        img_list, lbl_list, weights = self.__dataset_loader.load();
        lbl_list = np.array(lbl_list)[:,1];

        X_train, X_valid, y_train, y_valid = train_test_split(img_list, lbl_list, test_size=0.2, random_state=40);

        self.train_dataset = NetworkDataset(X_train, y_train, self.train_transforms, train = True);
        self.valid_dataset = NetworkDataset(X_valid, y_valid, self.valid_transforms, train = False);
        #self.test_dataset = NetworkDataset(xray_list_test, mask_list_test, self.valid_transforms, train = False);

        self.train_loader = DataLoader(self.train_dataset, 
        batch_size= Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True);

        self.valid_loader = DataLoader(self.valid_dataset, 
        batch_size = Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True);

        # self.test_loader = DataLoader(self.test_dataset, 
        # batch_size= Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True);

        self.added_indices = [];
        self.ce = nn.CrossEntropyLoss(torch.tensor(weights)).to(DEVICE);

        self.stopping_strategy = CombinedTrainValid(2,5);


    def __train_one_epoch(self, loader, model, optimizer):

        epoch_loss = 0;
        step = 0;
        update_step = 1;
        with tqdm(loader, unit="batch") as batch_data:
            for radiograph, mask, _ in batch_data:
                radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE)

                with torch.cuda.amp.autocast_mode.autocast():
                    pred = model(radiograph);
                    loss_ce = self.ce(pred,mask.long());

                    loss = loss_ce;

                self.scaler.scale(loss).backward();
                epoch_loss += loss.item();
                step += 1;

                if step % update_step == 0:
                    self.scaler.step(optimizer);
                    self.scaler.update();
                    optimizer.zero_grad();
                    

    def __eval_one_epoch(self, loader, model):
        epoch_loss = 0;
        total_pred_lbl = None;
        total_mask = None;
        first = True;
        count = 0;
        
        with torch.no_grad():
            with tqdm(loader, unit="batch") as epoch_data:
                for radiograph, mask, _ in epoch_data:
                    radiograph,mask = radiograph.to(Config.DEVICE), mask.to(Config.DEVICE);

                    pred = model(radiograph);
                    loss = self.ce(pred,mask.long());

                    epoch_loss += loss.item();
                    
                    if first is True:
                        total_pred = pred;
                        total_mask = mask;
                        first = False;
                    else:
                        total_pred = torch.cat([total_pred, pred], dim=0);
                        total_mask = torch.cat([total_mask, mask], dim=0);

                    count += 1;
        total_pred_lbl = torch.argmax(torch.softmax(total_pred, dim=1), dim=1);
        total_mask = total_mask;
        prec = self.precision_estimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        rec = self.recall_estimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        acc = self.accuracy_esimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        f1 = self.f1_esimator(total_pred_lbl.flatten(), total_mask.flatten().long());
        return epoch_loss / count, acc, prec, rec, f1;

    def start_train_slot(self):
        logging.info("Start training...");

        self.initialize_new_train();
        best_valid_loss = 9999;
        e = 1;
        while(True):
            self.model.train();
            self.__train_one_epoch(self.train_loader,self.model, self.optimizer);

            self.model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.__eval_one_epoch(self.train_loader, self.model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(self.valid_loader, self.model);

            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

            self.writer.add_scalar('training/loss', float(train_loss),e);
            self.writer.add_scalar('training/precision', float(train_precision),e);
            self.writer.add_scalar('training/recall', float(train_recall),e);
            self.writer.add_scalar('training/accuracy', float(train_acc),e);
            self.writer.add_scalar('training/f1', float(train_f1),e);

            self.writer.add_scalar('validation/loss', float(valid_loss),e);
            self.writer.add_scalar('validation/precision', float(valid_precision),e);
            self.writer.add_scalar('validation/recall', float(valid_recall),e);
            self.writer.add_scalar('validation/accuracy', float(valid_acc),e);
            self.writer.add_scalar('validation/f1', float(valid_f1),e);

            if(valid_loss < best_valid_loss):
                print("New best model found...");
                #save_checkpoint(model, e);
                best_valid_loss = valid_loss;
                best_model = deepcopy(self.model.state_dict());
               # save_samples(self.model, self.valid_loader, e, 'evaluation');

            if self.stopping_strategy(valid_loss, train_loss) is False:
                break;
            e += 1;

            # num_experiments +=1;
            # print(f"\n-----------\nFinal evaluation iteration {num_experiments}");
            # self.model.load_state_dict(best_model);
            # self.model.eval();
            # test_loss, test_acc, test_precision, test_recall, test_f1 = self.__eval_one_epoch(self.test_loader, self.model);
            # if best_f1 < test_f1:
            #     best_f1 = test_f1;

            # history.append([test_loss, test_acc.cpu(), test_precision.cpu(), test_recall.cpu(), test_f1.cpu()]);

        
            # print(f"Test {num_experiments}\tLoss: {test_loss}\tPrecision: {test_precision}\tRecall: {test_recall}\tAccuracy: {test_acc}\tF1: {test_f1}");
            # print(f"\n-----------\n");

            # train_loader, valid_loader = self.get_data(5, self.model);
            # self.stopping_strategy.reset();
        
        # plot_titles = ["Loss", "Acc", "Precision", "Recall", "F1"];
        # history = np.array(history).T;
        # for i in range(1,6):
        #     plt.subplot(1,5,i);
        #     plt.plot(np.arange(num_experiments), history[i-1]);
        #     plt.title(plot_titles[i-1]);
        # plt.savefig("fig.png");
        # plt.show();
    
    def start_train_slot_whole_data(self):
        logging.info("Start training...");

        self.initialize_new_train();

        best = 100;
        e = 0;
        best_model = None;

        while(True):
            self.model.train();
            self.__train_one_epoch(self.train_loader,self.model, self.optimizer);

            self.model.eval();
            train_loss, train_acc, train_precision, train_recall, train_f1 = self.__eval_one_epoch(self.train_loader, self.model);

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.__eval_one_epoch(self.valid_loader, self.model);

            print(f"Epoch {e}\tLoss: {train_loss}\tPrecision: {train_precision}\tRecall: {train_recall}\tAccuracy: {train_acc}\tF1: {train_f1}");
            print(f"Valid \tLoss: {valid_loss}\tPrecision: {valid_precision}\tRecall: {valid_recall}\tAccuracy: {valid_acc}\tF1: {valid_f1}");

            self.writer.add_scalar('training/loss', float(train_loss),e);
            self.writer.add_scalar('training/precision', float(train_precision),e);
            self.writer.add_scalar('training/recall', float(train_recall),e);
            self.writer.add_scalar('training/accuracy', float(train_acc),e);
            self.writer.add_scalar('training/f1', float(train_f1),e);

            self.writer.add_scalar('validation/loss', float(valid_loss),e);
            self.writer.add_scalar('validation/precision', float(valid_precision),e);
            self.writer.add_scalar('validation/recall', float(valid_recall),e);
            self.writer.add_scalar('validation/accuracy', float(valid_acc),e);
            self.writer.add_scalar('validation/f1', float(valid_f1),e);

            if(valid_loss < best):
                print("New best model found!");
                save_checkpoint(self.model, e);
                best = valid_loss;
                best_model = deepcopy(self.model.state_dict());
                save_samples(self.model, self.valid_loader, e, 'evaluation');

            if self.stopping_strategy(valid_loss) is False:
                break;

        self.model.load_state_dict(best_model);
        self.model.eval();
        print(f"\n-----------\nFinal evaluation iteration ");
        test_loss, test_acc, test_precision, test_recall, test_f1 = self.__eval_one_epoch(self.test_loader, self.model);

        print(f"Test {e}\tLoss: {test_loss}\tPrecision: {test_precision}\tRecall: {test_recall}\tAccuracy: {test_acc}\tF1: {test_f1}");
        print(f"\n-----------\n");
        
    
    def terminate_slot(self):
        self.trainer.terminate();
        self.evaluator.terminate();

    def load_model(self, name):
        load_checkpoint(name, self.model,);

    '''
        Predict of unlabeled data and update the second entry in  dictionary to 1.
    '''
    def predict(self, lbl, dc):
        #Because predicting is totally based on the initialization of the model,
        # if we haven't loaded a model yet or the loading wasn't successfull
        # we should not do anything and return immediately.
        if self.model_load_status:
            self.gen.eval();
            with torch.no_grad():
                radiograph_image = cv2.imread(os.path.sep.join(['images', lbl]),cv2.IMREAD_GRAYSCALE);
                clahe = cv2.createCLAHE(5,(9,9));
                radiograph_image = clahe.apply(radiograph_image);
                
                w,h = radiograph_image.shape;
                transformed = self.valid_transforms(image = radiograph_image);
                radiograph_image = transformed["image"];
                radiograph_image = torch.unsqueeze(radiograph_image,0);
                radiograph_image = radiograph_image.to(Config.DEVICE);
                p = self.gen(radiograph_image);
                mask_list = [];
                if Config.NUM_CLASSES > 2:
                    num_classes = p.size()[1];
                    p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                    p = np.argmax(p, axis=2);
                    
                    #Convert each class to a predefined color
                    for i in range(1,num_classes):
                        mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                        tmp = (p==i);
                        mask[(tmp)] = Config.PREDEFINED_COLORS[i-1];
                        mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_NEAREST);
                        mask_list.append(mask);
                else:
                    p = (p > 0.5).long();
                    p = p.permute(0,2,3,1).cpu().detach().numpy()[0];
                    mask = np.zeros(shape=(Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3),dtype=np.uint8);
                    tmp = (p==1).squeeze();
                    mask[(tmp)] = Config.PREDEFINED_COLORS[0];
                    mask = cv2.resize(mask,(h,w), interpolation=cv2.INTER_NEAREST);
                    mask_list.append(mask);

                # p = p[0]*255;
                # p = np.array(p, dtype=np.uint8);
                

                # kernel = np.ones((9,9), dtype=np.uint8);
                # opening = cv2.morphologyEx(p, cv2.MORPH_OPEN, kernel);
                # close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel);

                #save image to predicition directory
                #cv2.imwrite(os.path.sep.join(['predictions',os.path.basename(lbl)]), mask);
