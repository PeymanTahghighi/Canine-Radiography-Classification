import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from glob import glob

#---------------------------------------------------------
def update_folds(root_dataframe, num_folds = 5):

    #find intersection with spine and ribs since it hasn't been labelled yet
    spine_and_ribs = pickle.load(open('D:\\PhD\\Miscellaneous\\Spine and Ribs\\SpineandRibs.uog','rb'));
    img_list_all = list(root_dataframe['Image']);
    img_list_all = list(map(str, img_list_all));
    lbl_list_all = list(root_dataframe['Diagnosis']);

    img_list = [];
    lbl_list = [];

    for s in spine_and_ribs.keys():
        if spine_and_ribs[s][0]=='labeled':
            file_name = s[:s.rfind('.')];
            meta_file = pickle.load(open(f'D:\\PhD\\Miscellaneous\\Spine and Ribs\\labels\\{file_name}.meta', 'rb'));
            if 'Ribs' in meta_file.keys() and 'Spine' in meta_file.keys():
                idx = img_list_all.index(file_name);
                img_list.append(img_list_all[idx]);
                lbl_list.append(lbl_list_all[idx]);



    le = LabelEncoder();
    lbl_list =  le.fit_transform(lbl_list);

    skfold = StratifiedKFold(num_folds, shuffle=True, random_state=42);
    fold_cnt = 0;
    for train_idx, test_idx in skfold.split(img_list, lbl_list):
        pickle.dump([train_idx, test_idx], open(f'{fold_cnt}.fold', 'wb'));
        fold_cnt += 1;
#---------------------------------------------------------

#---------------------------------------------------------
def load_folds():
    fold_lst = glob('*.fold');
    folds = [];
    for f in fold_lst:
        folds.append(pickle.load(open(f, 'rb')));
    

    return folds;
#---------------------------------------------------------

if __name__ == "__main__":
    root_dataframe = pd.read_excel('G:\\My Drive\\dvvd_list_final.xlsx');

    #(1-1)
    update_folds(root_dataframe);
    #(1-2)
    folds = load_folds();





