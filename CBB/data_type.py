#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : data_type.py
@Time         : 2024/02/14 14:14:02
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the functions that are related to the data type manipulations
'''
from CBB.errors import *
import numpy as np 
import pandas as pd 


def is_numpy(data):
    if not isinstance(data, np.ndarray):
        raise TypeError("Error! The given data is not a numpy ndarray!")
    else:
        return

def map_str2int(data : pd.DataFrame, name):
    assert name in data.columns.values,"Error! The mapping name is not in the Dataframe"

    unique_values = data[name].unique()
    if len(unique_values) < 2:
        print("Warning! The given column contains no more than 2 unique values, function return!")
    elif len(unique_values) == 2:
        print(unique_values)
        mapping = np.arange(len(unique_values))
        mapping = list(zip(unique_values,mapping))
        data[name] = data[name].map({mapping[i][0] : mapping[i][1] for i in range(len(mapping))}).astype(int)
        return data
    else:
        print("Warning! For more than 2 unique values in the list, the dataset is supposed to be changed using DataFrame.dummy()")
        return data
    
def is_cate(data):
    """This function is used to check whether the given data set is a categorical data 

    Args:
        data (numpy.ndarray): The 1-D like numpy.ndarray which 

    Returns:
        bool: True if the data pass the categorical check otherwise False
    """
    if isinstance(data, pd.Series):
        try:
            data = data.values
        except:
            raise TypeConvertionFailed("Error! The given data is not numpy.ndarray and it failed to be converted into numpy.ndarray!")
    elif not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except:
            raise TypeConvertionFailed("Error! The given data is not numpy.ndarray and it failed to be converted into numpy.ndarray!")
    

    from collections import Counter
    val_dict = Counter(data)
    if any(isinstance(x, str) and not x.isnumeric() for x in val_dict.keys()):
        print("The data is string-like, must be category")
        return True
    elif any(len(val_dict)//2 > count for _key , count in val_dict.items()):
        print("This data does not seems like a categorical type data, see val count {}".format(val_dict))
        return False
    else:
        print("Data length check passed! The data seems like a categorical database!")
        return True

def variable_type(column):
    unique_values = column.unique()
    num_unique_values = len(unique_values)
    total_values = len(column)
    ratio_unique_values = num_unique_values / total_values

    if ratio_unique_values < 0.05 and num_unique_values < 10:
        return 'Cat'
    else:
        return 'Con'

def round_down_to_nearest_half(number):
    dec = number%1

    if dec == 0:
        return number   
    elif dec - 0.5 > 0:
        return np.floor(number) + 0.5
    elif dec - 0.5 < 0:
        return np.floor(number)
    else:
        raise ValueError("Error! The given number is not valid!")
    
def round_up_to_nearest_half(number):
    dec = number%1
    if np.isinf(number):
        return 10000
    if dec == 0:
        return number   
    elif dec - 0.5 > 0:
        return np.ceil(number)
    elif dec - 0.5 < 0:
        return np.floor(number) + 0.5
    else:
        raise ValueError("Error! The given number is not valid!")


def extract_three_digit_numbers(text):
    import re
    # 定义匹配长度为3的数字字段的正则表达式
    pattern = re.compile(r'\b(\d{3})\b')
    # 使用 findall 函数匹配所有符合条件的子字符串并返回列表
    numbers = re.findall(pattern, text)
    return numbers