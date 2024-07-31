import cv2
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

