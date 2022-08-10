from copyreg import pickle
from glob import glob
import pickle
from re import L
import cv2
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
from network_trainer import NetworkTrainer
import seaborn as sns
import matplotlib.pyplot as plt

TRAIN = False;
def test(root):
    meta_files = glob(f"{root}\\labels\\*.meta");

    image_list = [];
    mask_list = [];
    lbl_list = [];
    all_sternum_area = [];
    mi = 100;
    for idx, m in enumerate(meta_files):
        meta_data = pickle.load(open(m, 'rb'));
        if 'Sternum' in meta_data.keys():
            sternum_mask = cv2.imread(os.path.join(root, 'labels', meta_data['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
            sternum_mask = np.where(sternum_mask > 0, 255, 0).astype("uint8");
            contours = cv2.findContours(sternum_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
            w,h = sternum_mask.shape;
            for c in contours:
                area = cv2.contourArea(c);
                if area < mi:
                    mi = area;
                all_sternum_area.append(area/(w*h));
    print(f'min:{mi}')
    sns.displot(all_sternum_area, kind='kde');
    plt.show();
    

def preload_dataset(root):
    meta_files = glob(f"{root}\\labels\\*.meta");

    image_list = [];
    mask_list = [];
    lbl_list = [];

    for idx, m in enumerate(meta_files):
        meta_data = pickle.load(open(m, 'rb'));
        if 'Sternum' in meta_data.keys():
            sternum_mask = cv2.imread(os.path.join(root, 'labels', meta_data['Sternum'][2]), cv2.IMREAD_GRAYSCALE);
            sternum_mask = np.where(sternum_mask > 0, 255, 0).astype("uint8");
            mask_list.append(os.path.join(root, 'labels', meta_data['Sternum'][2]));
            #assign label based on summation of positive pixels
            sternum_mask_sum = sternum_mask.sum()/255;
            if sternum_mask_sum > 10:
                lbl_list.append(1);
            else:
                lbl_list.append(0);

            #get image from images folder
            file_name = os.path.basename(m);
            file_name = file_name[:file_name.rfind('.')];

            if os.path.exists(os.path.join(root,'images', f"{file_name}.jpeg")):
                image_list.append(os.path.join(root,'images', f"{file_name}.jpeg"));
            elif os.path.exists(os.path.join(root,'images', f"{file_name}.png")):
                image_list.append(os.path.join(root,'images', f"{file_name}.png"));
            else:
                print(f"{file_name} does not exists");
        else:
            print(m);
        
    image_list = np.array(image_list);
    mask_list = np.array(mask_list);
    lbl_list = np.array(lbl_list);

    return image_list, mask_list, lbl_list;

if __name__ == "__main__":
    # image_list, mask_list, lbl_list = preload_dataset('C:\\PhD\\Miscellaneous\\Sternum');
    # all_data = [image_list, mask_list, lbl_list];
    # pickle.dump(all_data, open('all_data.dmp', 'wb'));
    #test('C:\\PhD\\Miscellaneous\\Sternum');
    image_list, mask_list, lbl_list = pickle.load(open('all_data.dmp', 'rb'));


    pos_labels = (lbl_list == 1).sum();
    neg_labels = (lbl_list == 0).sum();

    print(f'Total pos: {pos_labels} neg: {neg_labels}');

    

    nt = NetworkTrainer();
    kfold = KFold(5);

    # curr_fold = 1;
    # for train_idx, test_idx in kfold.split(image_list, lbl_list):
    #     pickle.dump([train_idx,test_idx], open(f'{curr_fold}.dmp', 'wb'));
    #     curr_fold += 1;
    train_idxs = [pickle.load(open('1.dmp', 'rb'))[0], pickle.load(open('2.dmp', 'rb'))[0], pickle.load(open('3.dmp', 'rb'))[0], pickle.load(open('4.dmp', 'rb'))[0], pickle.load(open('5.dmp', 'rb'))[0]]
    test_idxs = [pickle.load(open('1.dmp', 'rb'))[1], pickle.load(open('2.dmp', 'rb'))[1], pickle.load(open('3.dmp', 'rb'))[1], pickle.load(open('4.dmp', 'rb'))[1], pickle.load(open('5.dmp', 'rb'))[1]];

    for curr_fold in range(0,5):
        print(f'===============Starting fold: {curr_fold}==================')
        img_train, mask_train, lbl_train = image_list[train_idxs[curr_fold]], mask_list[train_idxs[curr_fold]], lbl_list[train_idxs[curr_fold]];
        img_test, mask_test, lbl_test = image_list[test_idxs[curr_fold]], mask_list[test_idxs[curr_fold]], lbl_list[test_idxs[curr_fold]];

        if TRAIN:
            nt.train(curr_fold, [img_train, mask_train, lbl_train], [img_test, mask_test, lbl_test]);
        else:
            #nt.store_results(curr_fold, [img_test, mask_test, lbl_test], [img_train, mask_train, lbl_train]);
            pass
    nt.final_results();
            #fold_cnt+=1;
        

