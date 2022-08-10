#===========================================================
#===========================================================
import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from zmq import device
from network import Unet
#===========================================================
#===========================================================

class SternumClassifier():
    def __init__(self, network_path, device):

        #load network weights
        if os.path.exists(network_path):
            checkpoint = torch.load(network_path, map_location=device);
        else:
            print("ERROR: Path to network does not exists...");
            return;

        #load and initialize network
        self.net = Unet(num_classes=1).to(device);
        self.net.load_state_dict(checkpoint["state_dict"]);

        self.device = device;

    
    def __segment(self, img_path):
        '''
        Given image path, this function runs a network that segments
        sternum from the given image and if the number of pixels marked
        is above a threshold, returns true.
        '''

        transform = A.Compose(
            [
                A.Resize(1024, 1024),
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        )    

        #load image
        radiograph_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
        if radiograph_image is None:
            print("ERROR: could not load image...");
            return;
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8));
        radiograph_image = clahe.apply(radiograph_image);
        radiograph_image = np.expand_dims(radiograph_image, axis=2);
        radiograph_image = np.repeat(radiograph_image, 3,axis=2);

        transformed = transform(image = radiograph_image);
        radiograph_image = transformed["image"];

        radiograph_image = radiograph_image.unsqueeze(0).to(self.device);

        output = self.net(radiograph_image);
        output = torch.sigmoid(output) > 0.5;
        return output.detach().cpu().numpy();

    def classify(self, img_path):
        segment = self.__segment(img_path = img_path);

        positives = (segment == 1).sum();

        if positives > 1000:
            return True;
        return False;