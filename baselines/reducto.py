"""
Reducto
reference:
    https://github.com/reducto-sigcomm-2020/reducto/blob/master/reducto/differencer/diff_processor.py
"""
import cv2
from tqdm import tqdm
import imutils


def img_diff(img_path_list, feature):
    """img_diff
    image difference computing

    Args:
        img_path_list (list of str)
        feature (str): pixel/area/edge/corner
    
    Returns:
        score_list (list of float): feat_i - feat_{i-1}
    """
    if feature == "pixel":
        diff_func = pixel_diff
    elif feature == "area":
        diff_func = area_diff
    elif feature == "edge":
        diff_func = edge_diff
    elif feature == "corner":
        diff_func = corner_diff
    else:
        print("Invalid feature argument: {}. (expected in ['pixel', 'area', 'edge', 'corner'])".format(feature))
        return None
    
    total_num = len(img_path_list)

    prev_img_path = img_path_list[0]
    prev_img = cv2.imread(prev_img_path)
    
    score_list = []
    for idx in tqdm(range(1, total_num)):
        img_path = img_path_list[idx]
        img = cv2.imread(img_path)

        diff = diff_func(img, prev_img)
        score_list.append(diff)
        
        prev_img = img.copy()

    return score_list


# default configuration
# https://github.com/reducto-sigcomm-2020/reducto/blob/master/config/diff_config.ini
PIXEL_THRESH_LOW_BOUND = 21

AREA_BLUR_RAD = 11
AREA_BLUR_VAR = 0
AREA_THRESH_LOW_BOUND = 21

EDGE_BLUR_RAD = 11
EDGE_BLUR_VAR = 0
EDGE_CANNY_LOW = 101
EDGE_CANNY_HIGH = 255
EDGE_THRESH_LOW_BOUND = 21

CORNER_BLOCK_SIZE = 5
CORNER_KSIZE = 3
CORNER_K = 0.05
# ==========================


def pixel_diff(img, prev_img, 
               thresh=PIXEL_THRESH_LOW_BOUND):
    # 1. feature extraction
    feat, prev_feat = img, prev_img
    # 2. difference calculation
    total_pixels = feat.shape[0] * feat.shape[1]
    img_diff = cv2.absdiff(feat, prev_feat)
    img_diff = cv2.cvtColor(img_diff, cv2.COLOR_BGR2GRAY)
    img_diff = cv2.threshold(img_diff, thresh, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(img_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed


def area_diff(img, prev_img, 
              blur_rad=AREA_BLUR_RAD, blur_var=AREA_BLUR_VAR,
              thresh=AREA_THRESH_LOW_BOUND):
    # 1. feature extraction
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_feat = cv2.GaussianBlur(prev_img_gray, (blur_rad, blur_rad), blur_var)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feat = cv2.GaussianBlur(img_gray, (blur_rad, blur_rad), blur_var)

    # 2. difference calculation
    total_pixels = feat.shape[0] * feat.shape[1]
    img_delta = cv2.absdiff(feat, prev_feat)
    img_thresh = cv2.threshold(img_delta, thresh, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.dilate(img_thresh, None)
    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if not contours:
        return 0.0
    return max([cv2.contourArea(c)/total_pixels for c in contours])


def edge_diff(img, prev_img, 
              blur_rad=EDGE_BLUR_RAD, blur_var=EDGE_BLUR_VAR,
              canny_low=EDGE_CANNY_LOW, canny_high=EDGE_CANNY_HIGH,
              thresh=EDGE_THRESH_LOW_BOUND):
    # 1. feature extraction
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_img_blur = cv2.GaussianBlur(prev_img_gray, (blur_rad, blur_rad), blur_var)
    prev_feat = cv2.Canny(prev_img_blur, canny_low, canny_high)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur_rad, blur_rad), blur_var)
    feat = cv2.Canny(img_blur, canny_low, canny_high)

    # 2. difference calculation
    total_pixels = feat.shape[0] * feat.shape[1]
    img_diff = cv2.absdiff(feat, prev_feat)
    img_diff = cv2.threshold(img_diff, thresh, 255, cv2.THRESH_BINARY)[1]
    changed_pixels = cv2.countNonZero(img_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed


def corner_diff(img, prev_img, 
                block_size=CORNER_BLOCK_SIZE, ksize=CORNER_BLOCK_SIZE, k=CORNER_K,
                thresh=EDGE_THRESH_LOW_BOUND):
    # 1. feature extraction
    prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    prev_img_corner = cv2.cornerHarris(prev_img_gray, block_size, ksize, k)
    prev_feat = cv2.dilate(prev_img_corner, None)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_corner = cv2.cornerHarris(img_gray, block_size, ksize, k)
    feat = cv2.dilate(img_corner, None)

    # 2. difference calculation
    total_pixels = feat.shape[0] * feat.shape[1]
    img_diff = cv2.absdiff(feat, prev_feat)
    changed_pixels = cv2.countNonZero(img_diff)
    fraction_changed = changed_pixels / total_pixels
    return fraction_changed