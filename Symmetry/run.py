from copyreg import pickle
from glob import glob
import pickle
from re import L
import cv2
import numpy as np
import os
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import matplotlib.pyplot as plt
from thorax import segment_thorax
from utility import divide_image_symmetry_line, get_symmetry_line

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

def get_histogram(img, bins):
    temp_img = np.where(img == 255, 1, 0);
    h,w = img.shape;
    if h < bins:
        ph = bins;
        padded_img = np.zeros((ph,w));
        padded_img[:h,:] = img;
        img = padded_img;
        h = ph;

    rows_per_bin = int(h / bins);
    hist_horizontal = [];
    for i in range(0,h,rows_per_bin):
        s = temp_img[i:i+rows_per_bin,:];
        hist_horizontal.append(int(s.sum()));
    
    hist_horizontal = np.array(hist_horizontal, dtype=np.float32);
    hist_horizontal = np.expand_dims(hist_horizontal, axis=1);
    hist_horizontal = hist_horizontal / hist_horizontal.sum();

    hist_vertical = [];
    for i in range(0,w,rows_per_bin):
        s = temp_img[:,i:i+rows_per_bin];
        hist_vertical.append(int(s.sum()));
    
    hist_vertical = np.array(hist_vertical, dtype=np.float32);
    hist_vertical = np.expand_dims(hist_vertical, axis=1);
    hist_vertical = hist_vertical / hist_vertical.sum();
    
    return hist_horizontal, hist_vertical;

if __name__ == "__main__":
    meta_file = pickle.load(open('C:\\Users\\Admin\\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\DV13.meta', 'rb'));
    rib_mask_name = meta_file['Ribs'][2];
    spine_mask_name = meta_file['Spine'][2];
    rib_mask = cv2.imread(f'C:\\Users\\Admin\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{rib_mask_name}', cv2.IMREAD_GRAYSCALE);
    spine_mask = cv2.imread(f'C:\\Users\\Admin\OneDrive - University of Guelph\\Miscellaneous\\Spine and Ribs\\labels\\{spine_mask_name}', cv2.IMREAD_GRAYSCALE);
    rib_mask = np.where(rib_mask>0, 255, 0).astype("uint8");
    spine_mask = np.where(spine_mask>0, 255, 0).astype("uint8");

    sym_line = get_symmetry_line(spine_mask);
    ribs_left, ribs_right = divide_image_symmetry_line(rib_mask, sym_line);
    thorax_right = segment_thorax(ribs_right);
    hist_hor, hist_ver = get_histogram(thorax_right,64);
    hist_hor = np.squeeze(hist_hor);
    fig = plt.figure();
    fig.set_size_inches(20,20);
    ax = plt.axes();
    ax.bar(np.arange(0,64), hist_hor,);
    
    plt.show();
    thorax_left = segment_thorax(ribs_left);
    #whole_thorax = segment_thorax(rib_mask);




    
    image_list, mask_list, lbl_list = preload_dataset('C:\\PhD\\Miscellaneous\\Spine and Ribs');
    # all_data = [image_list, mask_list, lbl_list];
    # pickle.dump(all_data, open('all_data.dmp', 'wb'));
    image_list, mask_list, lbl_list = pickle.load(open('all_data.dmp', 'rb'));


    nt = NetworkTrainer();
    train_idxs = [pickle.load(open('0.dmp', 'rb'))[0], pickle.load(open('1.dmp', 'rb'))[0], pickle.load(open('2.dmp', 'rb'))[0], pickle.load(open('3.dmp', 'rb'))[0], pickle.load(open('4.dmp', 'rb'))[0]]
    test_idxs = [pickle.load(open('0.dmp', 'rb'))[1], pickle.load(open('1.dmp', 'rb'))[1], pickle.load(open('2.dmp', 'rb'))[1], pickle.load(open('3.dmp', 'rb'))[1], pickle.load(open('4.dmp', 'rb'))[1]];

    for curr_fold in range(5):
        print(f'===============Starting fold: {curr_fold}==================');

        img_train, mask_train = image_list[train_idxs[curr_fold]], mask_list[train_idxs[curr_fold]];
        img_test, mask_test = image_list[test_idxs[curr_fold]], mask_list[test_idxs[curr_fold]];

        if TRAIN:
            nt.train(curr_fold, [img_train, mask_train], [img_test, mask_test]);
        else:
            nt.eval('ckpt1.pt', [img_test, mask_test]);