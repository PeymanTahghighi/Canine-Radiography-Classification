
from pandas.io import pickle
import numpy as np
import os
from torch.utils.data import Dataset
import cv2
from sklearn.utils import shuffle
from glob import glob
import pickle
import matplotlib.pyplot as plt
import config
from utility import draw_missing_spine, retarget_img, smooth_boundaries
from tqdm import tqdm

class CanineDataset(Dataset):
    def __init__(self, radiographs, masks,train = True,  multilabel=False):
        self.__radiographs = radiographs;
        self.__masks = masks;
        self.__train = train;
        if multilabel:
            
            num_masks = masks.shape[1];
            temp_masks = [];
            for m in tqdm(masks):
                sr = pickle.load(open(m[0].replace('\\', '/'), 'rb'));
                d = pickle.load(open(m[1], 'rb'));
                h = pickle.load(open(m[2], 'rb'));
                fb = pickle.load(open(m[3], 'rb'));

                d = cv2.resize(d.astype("uint8")*255, (h.shape[1], h.shape[0])) >0;

                mask = np.zeros((sr.shape[0], sr.shape[1], num_masks+1), dtype="uint8");
                mask[:,:,0] = np.where(sr == 1, 1,0).squeeze();
                mask[:,:,1] = np.where(sr == 2, 1,0).squeeze();
                mask[:,:,2] = d;
                mask[:,:,3] = h;
                mask[:,:,4] = fb;

                if config.DEBUG_TRAIN_DATA:
                    fig,ax = plt.subplots(1,num_masks+1);
                    ax[0].imshow(mask[:,:,0]);
                    ax[1].imshow(mask[:,:,1]);
                    ax[2].imshow(mask[:,:,2]);
                    ax[3].imshow(mask[:,:,3]);
                    ax[4].imshow(mask[:,:,4]);
                    plt.show();

                temp_masks.append(mask);

        
        temp_radiographs = [];
        for rad in tqdm(radiographs):
            radiograph = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{rad}.jpeg'),cv2.IMREAD_GRAYSCALE);
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            radiograph = clahe.apply(radiograph);
            radiograph = np.expand_dims(radiograph, axis=2);
            radiograph = np.repeat(radiograph, 3,axis=2);
            temp_radiographs.append(radiograph);


        self.__masks = temp_masks;
        self.__radiographs = temp_radiographs;

        # reload = False;
        # pos = 0;
        # neg = 0;
        # if reload is True:
        #     for index in tqdm(range(len(radiographs))):
        #         radiograph_image = cv2.imread(os.path.join(config.IMAGE_DATASET_ROOT,f'{radiographs[index]}.jpeg'),cv2.IMREAD_GRAYSCALE);
        #         if task == 'spineribs':
        #             mask = pickle.load(open(f'cache\\{radiographs[index]}_SR.msk', 'rb'));
        #         else:
        #             full_body_mask = cv2.imread(f'C:\\PhD\\Thesis\\Unsupervised Canine Radiography Classification\\results\\train_data\\{radiographs[index]}.png', 
        #             cv2.IMREAD_GRAYSCALE);
        #             kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
        #             full_body_mask = cv2.erode(full_body_mask, kernel, iterations=10);
        #             full_body_mask = cv2.resize(full_body_mask, (radiograph_image.shape[1], radiograph_image.shape[0]));
        #             radiograph_image = ((np.where(full_body_mask>0, 1, 0) * radiograph_image)).astype("uint8");
        #             mask =  pickle.load(open(masks[index], 'rb'));

                    

        #             spine_mask = smooth_boundaries(spine_mask,10);
        #             spine_mask = smooth_boundaries(spine_mask,25);
        #             spine_mask = draw_missing_spine(spine_mask);

        #             radiograph_image = (np.int32(radiograph_image) * np.where(spine_mask>1, 0, 1)).astype("uint8");
        #             #mask = (np.int32(mask) * np.where(spine_mask>1, 0, 1)).astype("uint8");
        #             sternum_contours = cv2.findContours(mask.astype("uint8")*255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];

        #             returned_sternums = np.zeros_like(mask);
        #             for idx,c in enumerate(sternum_contours):
        #                 tmp = np.zeros_like(mask);
        #                 tmp = cv2.drawContours(tmp, [c], 0, (255,255,255), -1);
        #                 area_before = np.sum(tmp)/255;
        #                 residual_tmp = ((np.where(spine_mask>0, 0, 1) * np.where(tmp>0, 1, 0))*255).astype("uint8");
        #                 area_after = np.sum(residual_tmp)/255;
        #                 rat = (area_after / area_before);
        #                 bbox = cv2.boundingRect(c);
        #                 if rat > 0.6 and bbox[2] > 5 and bbox[3] > 5:
        #                     returned_sternums = cv2.drawContours(returned_sternums, [c], 0, (255,255,255), -1);
                            


        #             ret, full_body_mask = retarget_img([radiograph_image,returned_sternums], full_body_mask);
        #             #cv2.imwrite(f'tmp\\{radiographs[index]}.png', ret[0]);
        #             #cv2.imwrite(f'tmp\\{radiographs[index]}_m.png', (ret[1]*255).astype("uint8"));
        #             radiograph_image = ret[0];
        #             mask = np.where(ret[1]>0, 1, 0);


        #             # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #             # radiograph_image = clahe.apply(radiograph_image);
        #         radiograph_image = np.expand_dims(radiograph_image, axis=2);
        #         radiograph_image = np.repeat(radiograph_image, 3,axis=2);
        #         # if self.__train is True:
        #         #     transformed = config.train_transforms(image = radiograph_image, mask = mask);
        #         # else:
        #         #     transformed = config.valid_transforms(image = radiograph_image, mask = mask);


        #         # radiograph_image = transformed["image"];
        #         # mask = transformed["mask"];

        #         self.__radiographs.append(radiograph_image);
        #         self.__masks.append(mask)
        #     if train is True:
        #         pickle.dump([self.__radiographs, self.__masks], open(f'sternum_data_train.dmp', 'wb'));
        #     else:
        #         pickle.dump([self.__radiographs, self.__masks], open(f'sternum_data_test.dmp', 'wb'));
        # else:
        #     if train is True:
        #         self.__radiographs, self.__masks  = pickle.load(open(f'sternum_data_train.dmp', 'rb'))
        #     else:
        #         self.__radiographs, self.__masks  = pickle.load(open(f'sternum_data_test.dmp', 'rb'))

            
    def __len__(self):
        return len(self.__radiographs);

    def __getitem__(self, index):
        
        radiograph = self.__radiographs[index]
        mask = self.__masks[index];
        
        
        
        if self.__train is True:
            transformed = config.train_transforms(image = radiograph, mask = mask);
        else:
            transformed = config.valid_transforms(image = radiograph, mask = mask);


        radiograph = transformed["image"];
        mask = transformed["mask"];

        return radiograph, mask;