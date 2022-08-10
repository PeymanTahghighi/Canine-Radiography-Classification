from copyreg import pickle
from glob import glob
import pickle
from re import L
import cv2
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
from network_trainer import NetworkTrainer
import pandas as pd

def replace_paranthes(name):
    
    idx = name.find('(');
    if idx == -1:
        return name;
    num = name[idx+1:idx+2];
    new_name = f"{name[:idx]}-{num}";
    new_name = new_name.replace(' ', '');
    return new_name;

TRAIN = True;
def preload_dataset(root):
    meta_files = glob(f"{root}\\labels\\*.meta");
    gt_data_df = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');
    gt_img_list = list(gt_data_df['Image']);
    gt_img_list = list(map(str,gt_img_list));

    image_list = [];
    mask_list = [];
    lbl_list = [];

    for idx, m in enumerate(meta_files):

        meta_data = pickle.load(open(m, 'rb'));
        file_name = os.path.basename(m);
        file_name = file_name[:file_name.rfind('.')];

        if 'Spine' in meta_data.keys() and 'Ribs' in meta_data.keys():

            img_alt_name = replace_paranthes(file_name);
            if file_name in gt_img_list:
                idx = gt_img_list.index(file_name);
            elif img_alt_name in gt_img_list:
                idx = gt_img_list.index(img_alt_name);
            else:
                print(m);
                continue;

            lbl = gt_data_df.iloc[idx]['Symmetric Hemithoraces'];
            if lbl == 1:
                lbl = 0;
            elif lbl == 2:
                lbl = 1;
        

        
            lbl_list.append(lbl);
            spine_mask = cv2.imread(os.path.join(root, 'labels', meta_data['Spine'][2]), cv2.IMREAD_GRAYSCALE);
            spine_mask = np.where(spine_mask > 0, 1, 0).astype(np.bool8);
            ribs_mask = cv2.imread(os.path.join(root, 'labels', meta_data['Ribs'][2]), cv2.IMREAD_GRAYSCALE);
            ribs_mask = np.where(ribs_mask > 0, 1, 0).astype(np.bool8);
            mask = np.zeros(shape=(ribs_mask.shape[0], ribs_mask.shape[1], 1));
            mask[spine_mask] = 2;
            mask[ribs_mask] = 1;
            mask = np.int32(mask);

            #get image from images folder
            

            pickle.dump(mask, open(f'cache\\{file_name}.msk', 'wb'));
            mask_list.append(f'cache\\{file_name}.msk');
            if os.path.exists(os.path.join(root,'images', f"{file_name}.jpeg")):
                image_list.append(os.path.join(root,'images', f"{file_name}.jpeg"));
            elif os.path.exists(os.path.join(root,'images', f"{file_name}.png")):
                image_list.append(os.path.join(root,'images', f"{file_name}.png"));
            else:
                print(f"{file_name} does not exists");
        # else:
        #     print(m);
        
    image_list = np.array(image_list);
    mask_list = np.array(mask_list);
    lbl_list = np.array(lbl_list);

    return image_list, mask_list, lbl_list;

if __name__ == "__main__":
    # image_list, mask_list, lbl_list = preload_dataset('C:\\PhD\\Miscellaneous\\Spine and Ribs');
    # all_data = [image_list, mask_list, lbl_list];
    # pickle.dump(all_data, open('all_data.dmp', 'wb'));
    image_list, mask_list, lbl_list = pickle.load(open('all_data.dmp', 'rb'));


    nt = NetworkTrainer();
    # kfold = StratifiedKFold(5);
    # fold_cnt = 0;
    # for train_idx, test_idx in kfold.split(image_list, lbl_list):
    #     pickle.dump([train_idx,test_idx], open(f'{fold_cnt}.dmp', 'wb'));
    #     fold_cnt += 1;
    curr_fold = 4;
    train_idxs = [pickle.load(open('0.dmp', 'rb'))[0], pickle.load(open('1.dmp', 'rb'))[0], pickle.load(open('2.dmp', 'rb'))[0], pickle.load(open('3.dmp', 'rb'))[0], pickle.load(open('4.dmp', 'rb'))[0]]
    test_idxs = [pickle.load(open('0.dmp', 'rb'))[1], pickle.load(open('1.dmp', 'rb'))[1], pickle.load(open('2.dmp', 'rb'))[1], pickle.load(open('3.dmp', 'rb'))[1], pickle.load(open('4.dmp', 'rb'))[1]];
    #for train_idx, test_idx in zip(train_idxs, test_idxs):
        #pickle.dump([train_idx,test_idx], open(f'{fold_cnt}.dmp', 'wb'));
    print(f'===============Starting fold: {curr_fold}==================');

    img_train, mask_train = image_list[train_idxs[curr_fold]], mask_list[train_idxs[curr_fold]];
    img_test, mask_test = image_list[test_idxs[curr_fold]], mask_list[test_idxs[curr_fold]];

    if TRAIN:
        nt.train(curr_fold, [img_train, mask_train], [img_test, mask_test]);
    else:
        nt.eval('ckpt1.pt', [img_test, mask_test]);