import cv2
import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import os.path as osp
import sys

from CBB.myTools import read_data
from CBB.myos import is_Exist
import nibabel as nib
import scipy
import SimpleITK as sitk
def Extract_max_VOI(data_path, save_path=None):
    is_Exist(data_path)
    file_name = osp.basename(data_path)
    path = osp.dirname(data_path)
    suffix = file_name.split(".")[-1]
    img = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(img)

    binary_data = (data > 0).astype(np.uint8)
    sitk_binary_data = sitk.GetImageFromArray(binary_data)

    labeled_data = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk_binary_data))

    label_volumes = np.bincount(labeled_data.flatten())[1:]
    largest_label = np.argmax(label_volumes) + 1

    largest_voi = np.zeros_like(data)
    largest_voi[labeled_data == largest_label] = 1

    largest_voi_img = sitk.GetImageFromArray(largest_voi)
    largest_voi_img.SetSpacing(img.GetSpacing())
    largest_voi_img.SetOrigin(img.GetOrigin())
    largest_voi_img.SetDirection(img.GetDirection())
    if save_path is None:
        sitk.WriteImage(largest_voi_img, osp.join(path, f"Max_{file_name}"))
    else:
        sitk.WriteImage(largest_voi_img, save_path)

def extract_top_N_vois(data_path, N=1, save_path=None):
    is_Exist(data_path)
    file_name = osp.basename(data_path)
    path = osp.dirname(data_path)
    suffix = file_name.split(".")[-1]
    img = sitk.ReadImage(data_path)
    data = sitk.GetArrayFromImage(img)
    binary_data = (data > 0).astype(np.uint8)
    sitk_binary_data = sitk.GetImageFromArray(binary_data)

    labeled_data = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk_binary_data))
    label_volumes = np.bincount(labeled_data.flatten())[1:]
    total_labels = len(label_volumes)

    if total_labels == 0:
        raise ValueError("No valid VOIs found in the input image.")
    elif total_labels < N:
        print(f"Warning: Only {total_labels} VOIs found, less than the requested top {N} VOIs.")
        N = total_labels  # Adjust N to the available numb

    top_labels = np.argsort(label_volumes)[-N:] + 1
    top_voi = np.zeros_like(data)
    for label in top_labels:
        top_voi[labeled_data == label] = 1

    top_voi_img = sitk.GetImageFromArray(top_voi)
    top_voi_img.SetSpacing(img.GetSpacing())
    top_voi_img.SetOrigin(img.GetOrigin())
    top_voi_img.SetDirection(img.GetDirection())

    # Save the resulting image
    if save_path is None:
        sitk.WriteImage(top_voi_img, osp.join(path, f"Top_{N}_{file_name}"))
    else:
        sitk.WriteImage(top_voi_img, save_path)


def get_VOIs(data_path):
    data = read_data(data_path)
    binary_data = (data > 0).astype(np.uint8)
    sitk_binary_data = sitk.GetImageFromArray(binary_data)
    labeled_data = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk_binary_data))
    label_volumes = np.bincount(labeled_data.flatten())[1:]
    return label_volumes

def get_max_VOIs(data_path):
    data = read_data(data_path)
    binary_data = (data > 0).astype(np.uint8)
    sitk_binary_data = sitk.GetImageFromArray(binary_data)
    labeled_data = sitk.GetArrayFromImage(sitk.ConnectedComponent(sitk_binary_data))
    label_volumes = np.bincount(labeled_data.flatten())[1:]
    max_voi = max(label_volumes)
    return max_voi

def get_Bounding_Box(data_path,mask_path, separate_clusters=False):
    is_Exist(data_path)
    is_Exist(mask_path)

    data = sitk.ReadImage(data_path)
    mask = sitk.ReadImage(mask_path)
    assert data.GetSize() == mask.GetSize(),"Error! The shape check failed, data has shape as {} but mask is {}".format(data.GetSize(), mask.GetSize())

    if separate_clusters:
        # Identify connected components in the mask
        connected_mask = sitk.ConnectedComponent(mask)
        mask_to_process = connected_mask
    else:
        mask_to_process = mask

    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(data,mask_to_process)
    labels = lsif.GetLabels()
    bounding_boxes = []
    for label in labels:
        bounding_box = lsif.GetBoundingBox(label)
        bounding_boxes.append(bounding_box)
    return bounding_boxes