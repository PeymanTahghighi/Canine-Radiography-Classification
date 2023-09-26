from copy import deepcopy
import numpy as np
import cv2
from itertools import chain, combinations
import matplotlib.pyplot as plt
import scipy
from copy import copy
from config import IMAGE_SIZE

def get_symmetry_line(img):
    assert img.ndim == 2, "Image should be grayscale"
    
    w,h = img.shape;

    symmetry_line = np.zeros((w,2), dtype = np.int32);

    for i in range(w):
        first_cord = None;
        second_cord = None;
        for j in range(h):
            if img[i][j] != 0 and first_cord is None:
                first_cord = j;
            elif img[i][j] != 0:
                second_cord = j;

        if second_cord != None and first_cord != None:    
            symmetry_line[i] = ((i,(second_cord + first_cord) / 2));
    
    #check for missed points with zero values,
    #we use the average of ten points after and ten points before
    #as their value
    look_ahead = 10;
    for idx, s in enumerate(symmetry_line):
        if s[1] == 0:
            start_idx = idx;
            #attemp to estimte this value
            sum = 0;
            cnt_pos = 0;
            while(cnt_pos != look_ahead):
                if(symmetry_line[start_idx][1] != 0):
                    sum += symmetry_line[start_idx][1];
                    cnt_pos += 1;
                start_idx += 1;

                if start_idx >= w:
                    break;
            
            start_idx = idx;
            cnt_neg = 0;
            while(cnt_neg != look_ahead):
                if(symmetry_line[start_idx][1] != 0):
                    sum += symmetry_line[start_idx][1];
                    cnt_neg += 1;
                start_idx -= 1;

                if start_idx < 0:
                    break;
            sum /= cnt_neg + cnt_pos;

            symmetry_line[idx] = (idx, sum);
    
    return symmetry_line;

def divide_image_symmetry_line(img, sym_line):
    img_left = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.uint8);
    img_right = np.zeros(shape = (img.shape[0], img.shape[1]), dtype=np.uint8);

    w,h = img.shape;

    for s in sym_line:
        for j in range(h):
            if j < s[1]:
                img_left[s[0], j] = img[s[0], j];
            else:
                img_right[s[0], j] = img[s[0], j];
    
    return img_left, img_right;

def remove_outliers(lst):
    ret_lst = [];
    q1 = np.quantile(lst, axis=0, q=0.25);
    q3 = np.quantile(lst, axis=0, q=0.75);
    iqr = q3-q1;
    dist_list = [];
    total_dist = 0;
    for idx, p in enumerate(lst):
        x_range_start = q1[0] - 1.5*iqr[0];
        x_range_end = q3[0] + 1.5*iqr[0];

        y_range_start = q1[1] - 1.5*iqr[1];
        y_range_end = q3[1] + 1.5*iqr[1];
        if p[0]<x_range_end and p[0]>x_range_start and p[1] < y_range_end and p[1] > y_range_start:
            ret_lst.append(idx);
    
    return ret_lst;

def remove_outliers_spine(lst):
    ret_lst = [];
    q1 = np.quantile(lst, axis=0, q=0.25);
    q3 = np.quantile(lst, axis=0, q=0.75);
    iqr = q3-q1;
    dist_list = [];
    total_dist = 0;
    for idx, p in enumerate(lst):
        x_range_start = q1[0] - 1.5*iqr[0];
        x_range_end = q3[0] + 1.5*iqr[0];

        y_range_start = q1[1] - 1.5*iqr[1];
        y_range_end = q3[1] + 1.5*iqr[1];
        if p[0]<x_range_end and p[0]>x_range_start and p[1] < y_range_end and p[1] > y_range_start:
            ret_lst.append(idx);

    return ret_lst;
def remove_blobs_spine(spine):
    kernel = np.ones((5,5), dtype=np.uint8);
    opening = cv2.morphologyEx(spine, cv2.MORPH_OPEN, kernel);
    ret_img = np.zeros_like(opening);
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        all_area.append(area);

    biggest = np.max(all_area);
    for idx, a in enumerate(all_area):
        if a > 0.4*biggest:
            ret_img = cv2.drawContours(ret_img,contours, idx, (255,255,255), -1);
    return ret_img;

def remove_blobs(ribs):
    kernel = np.ones((5,5), dtype=np.uint8);
    opening = cv2.morphologyEx(ribs, cv2.MORPH_OPEN, kernel);
    ret_img = np.zeros_like(opening);

    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    mean_area = 0;
    for c in contours:
        mean_area += cv2.contourArea(c);
    
    mean_area /= len(contours);
    
    for idx, p in enumerate(contours):
        area = cv2.contourArea(p);
        if area > mean_area * 0.1:
            ret_img = cv2.drawContours(ret_img, [p],0, (255,255,255),-1);
    ret_img = (np.where(ret_img>0, 1, 0) * np.where(ribs>0, 1, 0)).astype("uint8")*255       
    return ret_img;

def remove_outliers_hist_ver(hist, img):
    hist_thresh = np.where(hist.flatten() != 0)[0];
    streak_cnt = 0;
    streak_start = -1;
    streak_end = 0;
    min_streak = 1024*1024;
    min_start = 0;
    min_end = 0;
    streak_list = [];
    for idx in range(len(hist_thresh)-1):
        if hist_thresh[idx+1] - hist_thresh[idx] < 10:
            streak_cnt += 1;
            if streak_start == -1:
                streak_start = hist_thresh[idx];
        else:
            streak_end = hist_thresh[idx]+1 if hist_thresh[idx] < 1024 else hist_thresh[idx];
            streak_list.append([streak_start,streak_end,streak_end - streak_start]);
            if streak_cnt < min_streak:
                min_streak = streak_cnt;
                min_start = streak_start;
                min_end = streak_end;
            streak_start = -1;
            streak_end = -1;
            streak_cnt = 0;
    
    streak_end = hist_thresh[-1];
    streak_list.append([streak_start,streak_end,streak_end - streak_start]);
    streak_list.sort(key=lambda x:x[2],reverse=True);
    streak_list = np.array(streak_list);
    avg = np.mean(streak_list,axis=0)[2];
    img_new = deepcopy(img);
    for i in range(0,len(streak_list)):
        if streak_list[i][2] < avg*0.65:
            img_new[:,streak_list[i][0]:streak_list[i][1]+1 if streak_list[i][1]+1 < 1024 else  streak_list[i][1]] = 0
    return img_new;

def remove_outliers_hist_hor(hist, img):
    hist_thresh = np.where(hist.flatten() != 0)[0];
    streak_cnt = 0;
    streak_start = -1;
    streak_end = 0;
    max_streak = 0;
    max_start = 0;
    max_end = 0;
    for idx in range(len(hist_thresh)-1):
        if hist_thresh[idx+1] - hist_thresh[idx] < 50:
            streak_cnt += 1;
            if streak_start == -1:
                streak_start = hist_thresh[idx];
        else:
            streak_end = hist_thresh[idx];
            if streak_cnt > max_streak:
                max_streak = streak_cnt;
                max_start = streak_start;
                max_end = streak_end;
            streak_start = -1;
            streak_end = -1;
            streak_cnt = 0;
    
    streak_end = hist_thresh[idx+1];
    if streak_cnt > max_streak:
        max_streak = streak_cnt;
        max_start = streak_start;
        max_end = streak_end;
    img_new = deepcopy(img);
    img_new[:max_start,:] = 0
    img_new[max_end+1:,:] = 0

    return img_new;

def smooth_boundaries(spine, dist):
    spine_cpy = deepcopy(spine);
    spine_thresh = np.where(spine_cpy[0,:]>0);
    h,w = spine_cpy.shape;
    left_bound = [];
    right_bound = [];
    start = -1;
    for i in range(h):
        if np.sum(spine_cpy[i,:]) > 0:
            if start == -1:
                start = i;
            spine_thresh = np.where(spine_cpy[i,:]>0);
            left_bound.append(spine_thresh[0][0]);
            right_bound.append(spine_thresh[0][-1]);

    local_minimas = [];
    local_maximas = [];
    for i in range(dist, len(left_bound)-dist):
        temp_arr = left_bound[i-dist:i+dist];
        m = np.min(temp_arr);
        if m == left_bound[i]:
            local_minimas.append([m,i+start]);
    
    for i in range(dist, len(right_bound)-dist):
        temp_arr = right_bound[i-dist:i+dist];
        m = np.max(temp_arr);
        if m == right_bound[i]:
            local_maximas.append([m,i+start]);
    
    ret = np.zeros_like(spine_cpy);
    for l in range(len(local_minimas)-1):
        spine_cpy = cv2.line(spine_cpy, (int(local_minimas[l][0]), int(local_minimas[l][1])), (int(local_minimas[l+1][0]), int(local_minimas[l+1][1])),(255,255,255),1);
    for l in range(len(local_maximas)-1):
        spine_cpy = cv2.line(spine_cpy, (int(local_maximas[l][0]), int(local_maximas[l][1])), (int(local_maximas[l+1][0]), int(local_maximas[l+1][1])),(255,255,255),1);
    
    spine_cpy = np.where(spine_cpy > 0, 1, 0);
    out = np.zeros_like(spine_cpy);
    for i in range(h):
        if np.sum(spine_cpy[i,:]) > 0:
            r = spine_cpy[i,:];
            r = np.where(r == 1);
            s = r[0][0];
            e = r[0][-1];
            if s != e:
                w = int((e - s) / 4);
                out[i, s:e] = 255;
            else:
                out[i,s] = 255;
    return out;

def scale_width(spine, multiplier):
    spine = np.where(spine > 0, 1, 0);
    h,w = spine.shape;
    out = np.zeros_like(spine);
    for i in range(h):
        if np.sum(spine[i,:]) > 0:
            r = spine[i,:];
            r = np.where(r == 1);
            s = r[0][0];
            e = r[0][-1];
            if s != e:
                w = int((e - s) / multiplier);
                out[i, s-w:e+w] = 255;
            else:
                out[i,s] = 255;
    return out;

def remove_blobs_spine(spine):
    kernel = np.ones((5,5), dtype=np.uint8);
    opening = cv2.morphologyEx(spine, cv2.MORPH_OPEN, kernel);
    ret_img = np.zeros_like(opening);
    contours, _ = cv2.findContours(opening, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);

    all_area = [];
    for c in contours:
        area = cv2.contourArea(c);
        all_area.append(area);

    biggest = np.max(all_area);
    for idx, a in enumerate(all_area):
        if a > 0.4*biggest:
            ret_img = cv2.drawContours(ret_img,contours, idx, (255,255,255), -1);
    return ret_img;

'''
    Extract sternum features for classification
    @param sternum_mask: Raw sternum mask output from segmentation model
    @param spine_mask: Raw spine_mask output from model, 
    make sure to remove blobs before passing spine to this function
    @param whole_thorax: segmented whole thorax region
'''
def extract_sternum_features(sternum_mask, spine_mask, whole_thorax):
    spine_mask = smooth_boundaries(spine_mask,10);
    spine_mask = smooth_boundaries(spine_mask,25);
    spine_mask = draw_missing_spine(spine_mask);
    spine_mask = scale_width(spine_mask,2);

    sternum_mask = ((np.where(sternum_mask >0, 1, 0) * np.where(whole_thorax>0, 1, 0))*255).astype("uint8");
    sternum_mask = postprocess_sternum(sternum_mask);

    sternum_contours = cv2.findContours(sternum_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    max_dist = 0;
    max_cnt = None;
    max_center = None;
    max_spine_center = None;
    if len(sternum_contours) != 0:
        for sc in sternum_contours:
            #tmp = cv2.drawContours(tmp, [sc], 0, (0,0,255), -1);
            bbox = cv2.boundingRect(sc);
            center = [bbox[0] + int(bbox[2]/2), int(bbox[1] + bbox[3]/2)];
            spine_thresh = np.where(spine_mask[center[1],:]!=0)[0];
            spine_center = spine_thresh[0] + (abs(spine_thresh[-1] - spine_thresh[0])/2);
            dist = abs(center[0] - spine_center) / (abs(spine_thresh[-1] - spine_thresh[0]));
            if dist >= max_dist:
                max_dist = dist;
                max_cnt = sc;
                max_center = center;
                max_spine_center = spine_center;


    sternum_mask = np.where(sternum_mask > 0, 1, 0);
    spine_mask = np.where(spine_mask > 0, 1, 0);
    masked_sternum = np.maximum(sternum_mask - spine_mask,np.zeros_like(sternum_mask));
    total_masked_pixels = np.sum(masked_sternum);
    total_sternum_pixels = np.sum(sternum_mask);

    return [max_dist, total_masked_pixels/(total_sternum_pixels+1e-6)] ,(sternum_mask*255).astype("uint8");

def postprocess_sternum(sternum):
    kernel = np.ones((9,9), np.uint8);
    opening = cv2.morphologyEx(sternum, cv2.MORPH_OPEN, kernel);
    return opening;

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

def draw_missing_spine(img):
    img_cpy = deepcopy(img);
    rows = np.sum(img, axis = 1);
    rows_thresh = np.where(rows > 0)[0];
    GROW = 10;
    if rows_thresh[0] != 0:
        avg_w = [];
        for r in range(int(len(rows_thresh)*0.1)):
            cols_thresh = np.where(img[rows_thresh[r],:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            avg_w.append(w);
        avg_w = np.mean(avg_w);

        for r in rows_thresh:
            cols_thresh = np.where(img[r,:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            if w > avg_w*0.9:
                img_cpy[:r,cols_thresh[0]-GROW:cols_thresh[-1]+GROW] = 255;
                break;
    if rows_thresh[-1] != img.shape[0]:
        rows_thresh = rows_thresh[::-1];
        avg_w = [];
        for r in range(int(len(rows_thresh)*0.1)):
            cols_thresh = np.where(img[rows_thresh[r],:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            avg_w.append(w);
        avg_w = np.mean(avg_w);
        for r in rows_thresh:
            cols_thresh = np.where(img[r,:]>0)[0];
            w = cols_thresh[-1] - cols_thresh[0];
            if w > avg_w*0.9:
                img_cpy[r:img.shape[0],cols_thresh[0]-GROW:cols_thresh[-1]+GROW] = 255;
                break;
    
    # cv2.imshow("orig", img);
    # cv2.imshow('af', img_cpy);
    # cv2.waitKey();
    return img_cpy;

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


def extract_cranial_features(spine_mask, ribs_mask):
    '''
        Extract cranial feature for classification.
        :param spine_mask: original spine mask output from model
        :param ribs_mask: original segmented ribs
        :param full_body_segmentation: original segmented full body label
    '''
    
    spine_mask = np.where(spine_mask > 0, 255, 0).astype("uint8");
    ribs_mask = np.where(ribs_mask > 0, 255, 0).astype("uint8");

    ribs_mask = cv2.resize(ribs_mask, (512,512));
    spine_mask = cv2.resize(spine_mask, (512,512));

    sum_rows = np.sum(ribs_mask, axis=1);
    sum_rows = np.where(sum_rows > 0);
    start_y = sum_rows[0][0];

    height_ratio = start_y/512;

    return height_ratio;

def get_max_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    
    if len(contours) == 0:
        return 0, 0;

    max_cnt = 0;
    max_area = 0;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c;
    
    return max_cnt, max_area;

def draw_max_contour(mask):
    max_cnt, _ = get_max_contour(mask);
    tmp = np.zeros_like(mask);
    tmp = cv2.drawContours(tmp, [max_cnt], 0, (255,255,255), -1);
    return tmp;

def get_center_point(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE);
    
    if len(contours) == 0:
        print("Warning: No contour exists to find center")
        return [0, 0];

    max_cnt = 0;
    max_area = 0;
    for c in contours:
        a = cv2.contourArea(c);
        if a > max_area:
            max_area = a;
            max_cnt = c;

    bbox = cv2.boundingRect(max_cnt);
    return [bbox[0] + bbox[2], bbox[1] + bbox[3]];

def extract_caudal_features(abdomen, whole_thorax, heart_mask, ribs):
    '''
    Extract caudal feature for classification.
    :param abdomen: original spine mask output from model
    :param whole_thorax: original segmented thorax label
    :param heart_mask: original segmented heart mask label
    :param ribs: original segmented ribs label
    '''
     
    abdomen = (np.where(abdomen > 0, 1, 0) * 255).astype("uint8");
    heart_mask = (np.where(heart_mask > 0, 1, 0) * 255).astype("uint8");
    ribs = (np.where(ribs > 0, 1, 0) * 255).astype("uint8");

    abdomen = cv2.resize(abdomen, (1024, 1024));
    whole_thorax = cv2.resize(whole_thorax, (1024, 1024));
    heart_mask = cv2.resize(heart_mask, (1024, 1024));
    ribs = cv2.resize(ribs, (1024, 1024));
    

    thorax_cnt, thorax_area = get_max_contour(whole_thorax);
    kernel = np.array([[1,0,1],[1,0,1],[1,0,1]], dtype=np.uint8);
    whole_thorax = cv2.erode(whole_thorax, kernel, iterations=5);
    whole_thorax = cv2.resize(whole_thorax, (abdomen.shape[1], abdomen.shape[0]))

    inner_thorax = (np.int32(whole_thorax) * np.where(ribs>0, 0, 1)).astype("uint8");

    heart_max_cnt, heart_max_cnt_area = get_max_contour(heart_mask);

    highest_y = 0;
    
    if heart_max_cnt_area != 0:
        heart_bbox = cv2.boundingRect(heart_max_cnt);
        abdomen[0:heart_bbox[1]+heart_bbox[3],:] = 0;

    diaphragm_max_cnt, diaphragm_max_area = get_max_contour(abdomen);
    if diaphragm_max_area == 0:
        return [0, 0, 0];
    diaphragm_bbox = cv2.boundingRect(diaphragm_max_cnt);
    highest_y = diaphragm_bbox[1];

    diaph_area = cv2.contourArea(diaphragm_max_cnt);
    diaphragm_max_cnt = cv2.convexHull(diaphragm_max_cnt);
    abdomen = cv2.drawContours(np.zeros_like(abdomen), [diaphragm_max_cnt], 0, (255,255,255), -1);

    residual = np.maximum(np.int32(inner_thorax) - np.int32(abdomen), np.zeros_like(abdomen)).astype("uint8");
    kernel = np.ones((5,5), np.uint8);
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, kernel);
    residual_contours = cv2.findContours(residual, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    lowest_y = 0;
    for c in residual_contours:
        b = cv2.boundingRect(c);
        if b[1]+b[3] > lowest_y:
            lowest_y = b[1]+b[3];
    
    abdomen_new = deepcopy(abdomen);
    abdomen_new[:lowest_y,:] = 0;

    diaph_new_cnt, diap_new_area = get_max_contour(abdomen_new);
    thorax_cnt, thorax_area = get_max_contour(whole_thorax);
    thorax_bbox = cv2.boundingRect(thorax_cnt);

    return [lowest_y/(thorax_bbox[1]+thorax_bbox[3]),  diap_new_area / thorax_area, diap_new_area/(diaph_area+1e-6)]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def extract_sp_feature(spine, spinous_process, whole_thorax, file_name, store = False):
    spinous_process = cv2.resize(spinous_process.astype("uint8"), (1024,1024));
    whole_thorax = cv2.resize(whole_thorax.astype("uint8"), (1024,1024));
    spine = cv2.resize(spine.astype("uint8"), (1024,1024));

    spine_mask_5 = smooth_boundaries(spine,10);
    spine_mask_5 = draw_missing_spine(spine_mask_5);
    spine_mask_5 = scale_width(spine_mask_5,13);

    spinous_process = ((np.where(spinous_process>0, 1, 0) * np.where(whole_thorax>0,1,0))*255).astype("uint8")

    contours = cv2.findContours(spinous_process.astype("uint8")*255,
    cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0];
    avg_area = [];
    for c in contours:
        avg_area.append(cv2.contourArea(c));
    p_sp = np.zeros_like(spinous_process);
    avg_area = np.mean(avg_area);
    for c in contours:
        if cv2.contourArea(c) > avg_area:
            p_sp = cv2.drawContours(p_sp, [c], 0, (255,255,255), -1);

    overlap = (np.where(p_sp>0,1,0) * np.where(spine_mask_5>0,0,1));
    if store is True:
        cv2.imwrite(f'tmp\\{file_name}.png', overlap.astype("uint8")*255);
        cv2.imwrite(f'tmp\\{file_name}_b.png', cv2.addWeighted(p_sp, 0.5, spine_mask_5.astype("uint8"), 0.5, 0.0));

    sp_features_1 = np.sum(overlap) / ((np.sum(whole_thorax)/255) + 1e-6);
    sp_features_2 = np.sum(overlap) / ((np.sum(spinous_process)/255) + 1e-6);


    return sp_features_1, sp_features_2;

def confidence_intervals(scores, rounds = 5):
    mean = np.mean(scores, axis = 0);
    confidence = 0.95  # Change to your desired confidence level
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, df=rounds - 1)

    sd = np.std(scores, ddof=1, axis = 0)
    se = sd / np.sqrt(rounds)

    ci_length = t_value * se

    ci_lower = mean - ci_length
    ci_upper = mean + ci_length

    print(mean);
    print(ci_lower, ci_upper)