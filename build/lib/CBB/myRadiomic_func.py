#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : myRadiomic_func.py
@Time         : 2024/11/18/15:03
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the general functions for the calculations of radiomics features in CPU AND GPU

'''
import numpy as np
import SimpleITK as sitk
from CBB.myTools import load_yaml_file,read_sitk_data
from CBB.errors import *
from CBB.myCV import get_bin_edges

DEFAULT_PARAM = load_yaml_file(r"D:\Pycharm_workplace\CBB\param\Default_params.yaml")


class MyRadiomicsCore:
    def __init__(self, image, mask, **kwargs):
        if isinstance(image,str):
            self.image = read_sitk_data(image)
        else:
            self.image = image

        if isinstance(mask, str):
            self.mask = read_sitk_data(mask)
        else:
            self.mask = mask

        if "Params" not in kwargs:
            raise ValueError("Error! Cannot detect param file from the input!")

        if isinstance(kwargs['Params'], str):
            try:
                self.param = load_yaml_file(kwargs['Params'])
            except:
                raise ValueError("Cannot read the param file from path {}".format(kwargs['Params']))
        elif isinstance(kwargs['Params'], dict):
            self.param = kwargs['Params']
        else:
            raise ValueError("Cannot load parameter file from the given value")
        self.bin_width = self.param.get("setting",{}).get("binWidth",25)
        self.bin_count = self.param.get("setting",{}).get("binCount",None)
        self.bin_image, self.bin_edge = self._get_bin_image(self.image, self.mask,self.bin_width, self.bin_count)
    def _get_bin_image(self, image, mask, binwidth=25, bincount=None):
        if isinstance(image, sitk.Image):
            image = sitk.GetArrayFromImage(image)
        if isinstance(mask, sitk.Image):
            mask = sitk.GetArrayFromImage(mask)
        pm = np.zeros(image.shape, dtype="int")
        binEdges = get_bin_edges(image[mask], binWidth=binwidth, binCount=bincount)
        pm[mask] = np.digitize(image[mask],binEdges)
        return pm, binEdges


if __name__ == '__main__':
    test_data = r"D:\Pycharm_workplace\DATA_STORAGE\COVID\Data\005.nii.gz"
    test_mask = r"D:\Pycharm_workplace\DATA_STORAGE\COVID\Mask\005.nii.gz"
    test_para_file = r"D:\Pycharm_workplace\CBB\param\CT_Params.yaml"

    RTool = MyRadiomicsCore(test_data,test_mask, Params=test_para_file)
    print("Generator created successfully!")
