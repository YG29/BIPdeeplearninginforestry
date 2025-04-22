#!/usr/bin/env python
# coding: utf-8

import skimage as ski
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sklearn as sk
import pandas as pd
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# === Feature Extraction ===

def expand(feature):
    """Add a singleton channel dimension to a 2D array."""
    return np.expand_dims(feature, axis=-1)


def add_features(tile):
    """
    Args:
        tile (ndarray): An HxWxC image with at least 4 bands (RGB + NIR).
    Returns:
        ndarray: Feature-enhanced image with additional channels.
    """
    red = tile[:, :, 0]
    green = tile[:, :, 1]
    blue = tile[:, :, 2]
    NIR = tile[:, :, 3]
    gray_tile = ski.color.rgb2gray(tile[:, :, 0:3])
    
    # Vegetation indices
    brightness = (red + green + blue + NIR) / 4
    NDVI = (NIR - red) / (NIR + red + 1e-6)
    EVI = 2.5 * ((NIR - red) / (NIR + 6 * red - 7.5 * blue + 1))
    SAVI = ((1 + 0.5) * (NIR - red)) / (NIR + red + 0.5)

    # Edge feature
    canny = ski.feature.canny(gray_tile)
    canny = np.clip(canny, 1e-5, None)
    TDGI = -np.log10(canny) * brightness

    # Filters
    dog = ski.filters.difference_of_gaussians(tile, 2, 10)
    edge_scharr = ski.filters.scharr(gray_tile)
    gaussian = ski.filters.gaussian(tile)
    sato = ski.filters.sato(gray_tile)

    features_list = [
        tile, dog, gaussian,
        expand(NDVI), expand(EVI), expand(SAVI), expand(TDGI),
        expand(canny), expand(edge_scharr), expand(sato)
    ]

    return np.concatenate(features_list, axis=-1)

# === Add features to a range of tiles ===

def features_img(range_img, path, output_dir='train_tiles'):
    """
    Adds features to each image in range_img and saves the resulting image.

    Args:
        range_img (list): List of tile indices to load (e.g., [0, 1, 2]).
        path (str): Directory where the original images are stored.
        output_dir (str): Directory where the feature-enhanced images will be saved.
    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range_img:
        img_path = f'{path}/tile_{i}.tif'
        img = ski.io.imread(img_path)

        # Add features
        img_with_features = add_features(img)

        # Save the image as .npy (NumPy format) to preserve channels
        output_path = os.path.join(output_dir, f'tile_{i}.npy')
        np.save(output_path, img_with_features)

        print(f"Features added and saved: {output_path}")


# === Create large images from tiles (with features) ===

def create_big_img(range_img, path, img_type):
    """
    Args:
        range_img (list): List of tile indices to load.
        path (str): Path prefix to the image files.
        img_type (str): Either '.npy' or '.tif'.
    
    Returns:
        np.ndarray: Stacked NumPy array of images.
    """
    big_img = []
    for i in range_img:
        if img_type == '.npy':
            img_path = f'{path}_{i}.npy'
            img = np.load(img_path)
        elif img_type == '.tif':
            img_path = f'{path}_{i}.tif'
            img = ski.io.imread(img_path)
        else:
            raise ValueError("img_type must be either '.npy' or '.tif'")

        big_img.append(img)
    return np.stack(big_img)



# === Model training ===

def train_model(X, y, model):
    """
    Train + save a machine learning model.
    Args:
        X (ndarray): Training features (NxD).
        y (ndarray): Labels (N,).
        model (str): 'KNN' or 'RF'.
    Returns:
        None
    """
    if model == 'KNN':
        clf = KNeighborsClassifier(3)
    elif model == 'RF':
        clf = RandomForestClassifier(4)
    else:
        raise ValueError("Model must be 'KNN' or 'RF'")

    clf.fit(X, y)
    joblib.dump(clf, f'model/{model}.joblib')


# === Model prediction ===

def make_prediction(range_img, model_file):
    """
    Args:
        range_img (list): List of tile indices to process.
        model_file (str): Path to the saved model (.joblib).

    Returns:
        tuple: (big_img, big_label, predicted_imgs)
    """
    all_img = []
    all_label = []
    predicted_imgs = []
    clf = joblib.load(model_file)

    for i in range_img:
        img_path = f'test_tiles/tile_{i}.npy'
        label_path = f'masks/mask_{i}.tif'
        img = np.load(img_path)
        label = ski.io.imread(label_path)

        predict = clf.predict(img.reshape(-1, img.shape[-1]))
        
        predicted_imgs.append(predict)
        all_label.append(label)
        all_img.append(img)

    return all_img, all_label, predicted_imgs


# === Morphology & Bounding Boxes ===

def make_erosion(mask):
    """
    Apply binary erosion to a mask.
    Returns:
        ndarray: Eroded mask.
    """
    return ski.morphology.binary_erosion(mask.copy())


def extract_bbox(mask, min_area):
    """
    Args:
        mask (ndarray): Binary mask.
        min_area (int): Minimum region size to consider.

    Returns:
        list: List of bboxes [(min_row, min_col, max_row, max_col), ...]
    """
    labels = ski.measure.label(mask.copy(), connectivity=1)
    region_props = ski.measure.regionprops(labels)
    return [region.bbox for region in region_props if region.area > min_area]


def visualize_and_save_bboxs(list_image, bboxes, output_dir='predict_images', visualise=False):
    """
    Args:
        list_image (list): List of images.
        bboxes (list): List of bbox lists per image.
        output_dir (str): Output directory for saved plots.
    """
    nb_img = len(list_image)
    nb_bboxs = len(bboxes)

    if nb_img != nb_bboxs :
        raise ValueError(f'Not the same number of images and bboxs : got {nb_img} images; {nb_bboxs} bboxes.')

    fig, ax = plt.subplots(1, nb_img)

    for idx in range(nb_img):
        axes = ax[idx]
        img = list_image[idx]
        img = img.reshape(1024,1024)

        axes.imshow(img / 255.0) # /255 for normalization

        for bbox in bboxes[idx]:
            y_min, x_min, y_max, x_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
            axes.add_patch(rect)

        axes.set_title(f'{len(bboxes[idx])}')
        axes.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir, bbox_inches='tight', pad_inches=0)
    
    if visualise == True : 
    	plt.show()


# === Accuracy ===

def get_bboxes_area(boxA, boxB):
    """
    Calculate overlap and area of two bounding boxes.

    Args:
        boxA, boxB (tuple): (min_row, min_col, max_row, max_col)

    Returns:
        tuple: (intersection area, area of boxA, area of boxB)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    overlap = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return overlap, boxA_area, boxB_area


def accuracy_bboxs(true_bboxs_perTiles, pred_bboxs_perTiles):
    """
    Calculate precision, recall, IoU and F1 across tile-wise bboxs.

    Args:
        true_bboxs_perTiles (list): List of bbox lists (ground truth).
        pred_bboxs_perTiles (list): List of bbox lists (predicted).

    Returns:
        dict: {'accuracy', 'IoU', 'recall', 'precision', 'F1'}
    """
    IoU, recall, precision, F1 = [], [], [], []

    for true_boxes, pred_boxes in zip(true_bboxs_perTiles, pred_bboxs_perTiles):
        iou_vals, rec_vals, prec_vals, f1_vals = [], [], [], []

        for true_box in true_boxes:
            for pred_box in pred_boxes:
                overlap, true_area, pred_area = get_bboxes_area(true_box, pred_box)
                iou = overlap / float(true_area + pred_area - overlap)
                rec = overlap / float(true_area)
                prec = overlap / float(pred_area)
                f1 = (2 * prec * rec) / (prec + rec + 1e-6)

                iou_vals.append(iou)
                rec_vals.append(rec)
                prec_vals.append(prec)
                f1_vals.append(f1)

        IoU.append(np.max(iou_vals) if iou_vals else 0)
        recall.append(np.max(rec_vals) if rec_vals else 0)
        precision.append(np.max(prec_vals) if prec_vals else 0)
        F1.append(np.max(f1_vals) if f1_vals else 0)

    accuracy = np.mean(F1)
    return {
        'accuracy': accuracy,
        'IoU': IoU,
        'recall': recall,
        'precision': precision,
        'F1': F1
    }


# === Save to CSV ===

def data2csv(data_dict, file_name='prediction_bboxs_res.csv'):
    """
    Args:
        data_dict (dict): Results data.
        file_name (str): Output filename.
    """
    df = pd.DataFrame(data_dict)
    df.to_csv(file_name, index=False)

