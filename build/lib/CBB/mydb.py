#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : mydb.py
@Time         : 2024/02/14 14:08:00
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains functions that are related to the pandas.DataFrame manipulations
'''
import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from CBB.data_type import is_cate


def show_nan(data : pd.DataFrame):
    from pprint import pprint
    pprint(data.isna().sum().to_dict())
    return data.isna().sum() 

def remove_nan(data):
    if isinstance(data, np.ndarray):
        data = data[~np.isnan(data)]
        return data
    elif isinstance(data, pd.Series):
        data = data.dropna()
        return data
    else:
        raise TypeError("The incoming type '{}' is not legal!".format(type(data)))
    
def if_Nan_Exists(data : pd.DataFrame):
    return data.isnull().values.any()

def Columns_has_Nan(data : pd.DataFrame):
    if not if_Nan_Exists(data):
        print("Data frame does not contain Nan value, function returned!")
        return 0
    if isinstance(data, pd.DataFrame):
        output_list = []
        names = data.columns.values
        for name in names:
            if data[name].isnull().values.any():
                output_list.append(name)
        return output_list
    elif isinstance(data, pd.Series):
        if data.isnull().values.any():
            return True
    else:
        raise TypeError("The given data is in type {} but pd.DataFrame or pd.Series is required".format(type(data)))

def show_num_and_ratio(data,name):
    if not is_cate(data[name]):
        raise TypeError("Error! The given data is not a 'Category' type data")
    assert name in data.columns,f"Error! The input name {name} is not in the colum names, function return"
    if if_Nan_Exists(data[name]):
        print(f"Warning! The named column {name} contains nan values, these value will not be counted!")
    val_dict = data[name].value_counts()

    total_num = data[name].shape[0]
    val_dict = dict(val_dict)
    for k,v in val_dict.items():
        ratio = float(v) / total_num
        val_dict[k] = str(v) + " / " + str(round(ratio, 4)*100) + "%"
    print(val_dict)
    return val_dict


def normalize_and_melt(df, id_vars=['label'], value_vars=None, var_name='variable', value_name='value',save_path=None):
    from CBB.myos import check_path
    import os.path as osp
    # 如果未指定value_vars，默认使用除去id_vars外的所有列
    if value_vars is None:
        value_vars = df.columns.difference(id_vars).tolist()

    # 复制原始DataFrame以避免原地修改
    df_copy = df.copy()

    # 提取连续变量列进行归一化处理
    scaler = MinMaxScaler()
    df_copy[value_vars] = scaler.fit_transform(df_copy[value_vars])

    # 使用melt函数将DataFrame转换为长格式
    melted_df = pd.melt(df_copy, id_vars=id_vars, value_vars=value_vars, var_name=var_name, value_name=value_name)

    if save_path is not None:
        melted_df.to_csv(save_path, index=None)
    else:
        default_melt_df =check_path(r"./melt_df")
        save_path = osp.join(default_melt_df, "./melt.csv")
        melted_df.to_csv(save_path,index=None)
    return melted_df




if __name__ == '__main__':

    data = pd.read_csv(r"D:\Pycharm_workplace\New_test\new_db.csv")
    print(show_nan(data))

    show_num_and_ratio(data,"sex")