from copy import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import cv2
from glob import glob
import io
import os

GAP = 50;
IMG_WDITH = 512;
IMG_HEIGHT = 512;

def segment_thorax(img):

    #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
    img = cv2.resize(img, (IMG_WDITH,IMG_HEIGHT));
    h,w = img.shape;
    segment_thorax = np.ones(shape=img.shape, dtype=np.uint8);

    cols_sum = np.sum(img, axis = 0);
    rows_sum = np.sum(img, axis = 1);

    cols_sum = np.where(cols_sum > 0, 1, 0);
    cols_sum = cols_sum.tolist();
    x_start = cols_sum.index(1);
    x_end = len(cols_sum) - 1 - cols_sum[::-1].index(1);
    
    rows_sum = np.where(rows_sum > 0, 1, 0);
    rows_sum = rows_sum.tolist();
    y_start = rows_sum.index(1);
    y_end = len(rows_sum) - 1 - rows_sum[::-1].index(1);

    x_start -= GAP;
    x_end += GAP;
    y_start -= GAP;
    y_end += GAP;

    #union on beta=0.05 and 0.1
    #union on alpha=0.015 and 0.09

    x_length = (x_end - x_start);
    y_length = (y_end - y_start);
    range = [(0,0), (-x_length/4, -y_length/4), (-x_length/4, y_length/4), (x_length/4, -y_length/4), (x_length/4, y_length/4)];
    total_thorax = None;
    first = True;
    for v in range:
        s = np.linspace(0, 2*np.pi, 1000)
        c = x_start + x_length/2 + v[0] + x_length/2*np.cos(s)
        r = y_start + y_length/2 + v[1] +  y_length/2*np.sin(s)
        init = np.array([r, c]).T

        snake = active_contour(gaussian(img, 3, preserve_range=False),
                            init, alpha=0.09, beta=0.1, gamma=0.001, w_edge=15)
        
        snake = np.array(snake, dtype=np.int32);
        tmp = copy(snake);
        snake[:,0] = tmp[:, 1];
        snake[:,1] = tmp[:, 0];
        tmp = np.zeros(shape=img.shape, dtype=np.uint8);

        thorax_region = cv2.fillPoly(tmp, [snake], color = (255,255,255));


        if first is True:
            total_thorax = thorax_region;
            first = False;
        else:
            total_thorax = cv2.bitwise_or(total_thorax, thorax_region);
        # cv2.imshow("total", total_thorax);
        # cv2.waitKey();
        print("added new thorax...");

    return total_thorax;

if __name__ == "__main__":

    img_list = glob("ribs\\*.png");
    #img_list = reversed(img_list);

    for img_name in img_list:
        file_name = os.path.basename(img_name);
        file_name = file_name[:file_name.rfind('_')] + ".jpeg";
        #original_img = cv2.imread(os.path.sep.join(['ribs', file_name]), cv2.IMREAD_GRAYSCALE);
        #original_img = cv2.resize(original_img, (IMG_WDITH, IMG_HEIGHT));
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE);
        img = cv2.resize(img, (IMG_WDITH,IMG_HEIGHT));
        h,w = img.shape;

        cols_sum = np.sum(img, axis = 0);
        rows_sum = np.sum(img, axis = 1);

        cols_sum = np.where(cols_sum > 0, 1, 0);
        cols_sum = cols_sum.tolist();
        x_start = cols_sum.index(1);
        x_end = len(cols_sum) - 1 - cols_sum[::-1].index(1);
        
        rows_sum = np.where(rows_sum > 0, 1, 0);
        rows_sum = rows_sum.tolist();
        y_start = rows_sum.index(1);
        y_end = len(rows_sum) - 1 - rows_sum[::-1].index(1);

        x_start -= GAP;
        x_end += GAP;
        y_start -= GAP;
        y_end += GAP;

        #union on beta=0.05 and 0.1
        #union on alpha=0.015 and 0.09

        x_length = (x_end - x_start);
        y_length = (y_end - y_start);
        range = [(0,0), (-x_length/4, -y_length/4), (-x_length/4, y_length/4), (x_length/4, -y_length/4), (x_length/4, y_length/4)];
        snks = [];
        total_thorax = None;
        first = True;
        for v in range:
            s = np.linspace(0, 2*np.pi, 1000)
            c = x_start + x_length/2 + v[0] + x_length/2*np.cos(s)
            r = y_start + y_length/2 + v[1] +  y_length/2*np.sin(s)
            init = np.array([r, c]).T
            temp_shape = np.zeros(shape = img.shape, dtype=np.uint8);

            snake = active_contour(gaussian(img, 3, preserve_range=False),
                                init, alpha=0.09, beta=0.1, gamma=0.001, w_edge=15)
            
            snake = np.array(snake, dtype=np.int32);
            snks.append(snake);
            tmp = copy(snake);
            snake[:,0] = tmp[:, 1];
            snake[:,1] = tmp[:, 0];
            tmp = np.zeros(shape=img.shape, dtype=np.uint8);
            contours = np.array([[50,50], [50,150], [150,150], [150,50]])
            thorax_region = cv2.fillPoly(tmp, [snake], color = (255,255,255));


            if first is True:
                total_thorax = thorax_region;
                first = False;
            else:
                total_thorax = cv2.bitwise_or(total_thorax, thorax_region);
            
            print("added new thorax...");

            #focused_thorax = np.where(total_thorax == 255, original_img, 0);
        
        betas = [0.05, 0.1];
        for b in betas:
            s = np.linspace(0, 2*np.pi, 1000)
            c = x_start + x_length/2 + x_length/2*np.cos(s)
            r = y_start + y_length/2  +  y_length/2*np.sin(s)
            init = np.array([r, c]).T
            temp_shape = np.zeros(shape = img.shape, dtype=np.uint8);

            snake = active_contour(gaussian(img, 3, preserve_range=False),
                                init, alpha=0.015, beta=b, gamma=0.001, w_edge=15)
            
            snake = np.array(snake, dtype=np.int32);
            snks.append(snake);
            tmp = copy(snake);
            snake[:,0] = tmp[:, 1];
            snake[:,1] = tmp[:, 0];
            tmp = np.zeros(shape=img.shape, dtype=np.uint8);
            contours = np.array([[50,50], [50,150], [150,150], [150,50]])
            thorax_region = cv2.fillPoly(tmp, [snake], color = (255,255,255));

            total_thorax = cv2.bitwise_or(total_thorax, thorax_region);
            
            print("added new thorax...");

        #focused_thorax = np.where(total_thorax == 255, original_img, 0);
        
        #cv2.imwrite("res\\" + file_name + "_o.jpeg", original_img);
        #cv2.imwrite("res\\" + file_name + "_t.jpeg", focused_thorax);
        #cv2.imshow('t', focused_thorax);
        #cv2.imshow('tr', total_thorax);
        # cv2.imshow('treg', thorax_region);
        #cv2.waitKey();

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        #ax.plot(init[:, 1], init[:, 0], '--r', lw=1)
        #for idx,s in enumerate(snks):
        ax.plot(snks[0][:, 0], snks[0][:, 1], lw=1);

        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        plt.show()
        plt.savefig('p.png');

        