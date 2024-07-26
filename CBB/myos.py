#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : myos.py
@Time         : 2024/02/14 13:53:20
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the system level file manipulation functions
'''

import re
import os
import os.path as osp

def get_file_name(path, suffix=True):
    if suffix:
        return os.path.basename(path)
    else:
        return os.path.splitext(os.path.basename(path))[0]

def check_path(path):
    if not osp.exists(path):
        print("The given path does not exist! Creat a new one")
        os.makedirs(path)
        return path
    else:
        return path


def is_Exist(file_path):
    if not osp.exists(file_path):
        raise FileExistsError(f"Error! The file path {file_path} is not valid")
    else:
        return file_path

def auto_save_file(path):
    directory, file_name = os.path.split(path)
    while os.path.isfile(path):
        pattern = '(\d+)\)\.'
        if re.search(pattern, file_name) is None:
            file_name = file_name.replace('.', '(0).')
        else:
            current_number = int(re.findall(pattern, file_name)[-1])
            new_number = current_number + 1
            file_name = file_name.replace(f'({current_number}).', f'({new_number}).')
        path = os.path.join(directory + os.sep + file_name)
    return path

def format_number(num):
    return "{:03d}".format(num)


def get_full_paths(directory, mode=0):
    """
    获取指定目录下所有文件和子目录的完整路径

    :param directory: 目标目录
    :param mode: 0 - 返回所有文件和子目录的路径，1 - 仅返回文件路径，2 - 仅返回子目录路径
    :return: 包含完整路径的列表
    """
    # 使用 os.listdir 获取目录下的所有文件名和子目录名
    items = os.listdir(directory)
    # 使用 os.path.join 将文件名或子目录名与目录路径拼接起来，得到完整路径
    full_paths = [os.path.join(directory, item) for item in items]

    if mode == 1:
        # 仅保留文件路径
        full_paths = [path for path in full_paths if os.path.isfile(path)]
    elif mode == 2:
        # 仅保留子目录路径
        full_paths = [path for path in full_paths if os.path.isdir(path)]

    return full_paths


def find_sample_files(directory, sample_names):
    """
    在指定目录及其子目录中搜索与样本名匹配的文件，并返回文件的完整路径。

    :param directory: 样本保存的目录
    :param sample_names: 样本名列表
    :return: 匹配样本名的文件的完整路径列表
    """
    matched_files = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 获取文件名（不带后缀）
            file_name, file_extension = os.path.splitext(file)
            if file_name in sample_names:
                # 匹配成功，记录文件的完整路径
                matched_files.append(os.path.join(root, file))

    return matched_files