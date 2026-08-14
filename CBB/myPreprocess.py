import os
import os.path as osp
import sys 
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from myTools import read_sitk_data

def n4BiasFieldCorrection(image, mask, output_image_path):
    # 读取输入图像

    input_image = read_sitk_data(image)
    input_mask = read_sitk_data(mask)

    # 创建N4偏置场校正器
    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    corrected_image = corrector.Execute(input_image, input_mask)

    # 保存校正后的图像
    sitk.WriteImage(corrected_image, output_image_path)