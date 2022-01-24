import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon
import os
import pickle
import Config

def show_dialoge(icon, text, title, buttons, parent = None):
    if parent != None:
        msgBox = QMessageBox(parent);
    else:
        msgBox = QMessageBox()
    msgBox.setIcon(icon)
    msgBox.setText(text)
    msgBox.setWindowTitle(title)
    msgBox.setStandardButtons(buttons)

    returnValue = msgBox.exec()

def get_radiograph_label_meta(radiograph_root, mask_root):
    radiograph_list = os.listdir(radiograph_root);
    mask_list = os.listdir(mask_root);

    mask_names = [];
    radiograph_names = [];

    for m in mask_list:
        if m.find('meta') != -1:
            #Find file extension
            ext = "";
            file_name,_ = os.path.splitext(m);
            for r in radiograph_list:
                n,ext= os.path.splitext(r);
                if n == file_name:
                    break;
            mask_names.append(os.path.sep.join([mask_root,m]));
            radiograph_names.append(os.path.sep.join([radiograph_root,file_name+ext]));
    
    return radiograph_names, mask_names;

def load_radiograph_masks(radiograph_path, mask_path):
    radiograph_pixmap = QtGui.QPixmap(radiograph_path);
    #Open each mask in meta data and add them to list
    df = pickle.load(open(mask_path,'rb'));
    mask_pixmap_list = [];
    for k in df.keys():
        p = df[k][2];
        path = os.path.sep.join([Config.PROJECT_ROOT, 'labels', p]);
        mask_pixmap = QtGui.QPixmap(path);
        mask_pixmap_list.append([k,df[k][1],mask_pixmap]);

    return radiograph_pixmap, mask_pixmap_list;
