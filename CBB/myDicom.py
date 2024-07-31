from CBB.myos import is_Exist, get_full_paths
import pydicom
from pydicom.uid import generate_uid
from pydicom.uid import UID
import os
import os.path as osp
import sys
import numpy as np
import re

def extract_numbers(s):
    # 使用正则表达式匹配字符串中的所有数字
    numbers = re.findall(r'\d+', s)
    # 将提取到的数字转换为int类型，并返回
    return [int(num) for num in numbers]

def add_patient_id_to_dicom(directory):
    # 获取文件夹名称作为Patient ID
    patient_id = os.path.basename(directory)

    # 遍历文件夹中的所有DICOM文件
    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):  # 确保只处理DICOM文件
            filepath = os.path.join(directory, filename)
            # 读取DICOM文件
            ds = pydicom.dcmread(filepath)

            # 添加或更新Patient ID
            ds.PatientID = patient_id

            # 保存修改后的DICOM文件
            ds.save_as(filepath)
            print(f"Updated Patient ID for {filename}")


def add_tags(dicom_dir):
    # 检查目录是否存在
    is_Exist(dicom_dir)

    # 生成新的 Series Instance UID 和 Study Instance UID
    new_series_instance_uid = generate_uid()
    new_study_instance_uid = generate_uid()

    # 遍历目录中的所有 DICOM 文件
    for filename in os.listdir(dicom_dir):
        if filename.endswith(".dcm"):
            file_path = os.path.join(dicom_dir, filename)
            ds = pydicom.dcmread(file_path)

            # 生成新的 SOP Instance UID
            new_sop_instance_uid = generate_uid()

            # 设置新的 UID
            ds.StudyInstanceUID = new_study_instance_uid
            ds.SeriesInstanceUID = new_series_instance_uid
            ds.SOPInstanceUID = new_sop_instance_uid

            # 添加一些常用的 Series 和 Study 信息
            ds.SeriesNumber = ds.get('SeriesNumber', 1)
            ds.SeriesDescription = ds.get('SeriesDescription', 'Series Description')
            ds.StudyID = ds.get('StudyID', '1')
            ds.StudyDescription = ds.get('StudyDescription', 'Study Description')
            patient_id = osp.basename(dicom_dir)
            ds.PatientID = patient_id

            # 检查并添加其他必要标签
            if not hasattr(ds, 'PatientName'):
                ds.PatientName = 'Unknown'
            if not hasattr(ds, 'StudyDate'):
                ds.StudyDate = '20240101'  # Example date
            if not hasattr(ds, 'SeriesDate'):
                ds.SeriesDate = '20240101'  # Example date

            # 保存修改后的 DICOM 文件
            ds.save_as(file_path)

    print("All DICOM files processed.")


def check_and_correct_instance_numbers(directory):
    mismatched_files = []
    dicom_number = osp.basename(directory)
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith(".dcm"):
            filepath = os.path.join(directory, filename)
            # 读取DICOM文件
            ds = pydicom.dcmread(filepath)
            # 从文件名中提取索引
            # index_str = os.path.splitext(filename)[0].replace('IM', '')
            index_str = filename.split("I")[1].split(".")[0]
            index = int(index_str)
            # 检查Instance Number是否匹配
            if ds.InstanceNumber != index:
                print(f"Mismatch found: {dicom_number},slice index: {index} InstanceNumber: {ds.InstanceNumber}")
                # 记录不匹配的文件
                mismatched_files.append((filepath, index))

    # 修改不匹配的文件
    if len(mismatched_files) == 0:
        print(f"Dicom : {dicom_number} is OK")
    else:
        for filepath, correct_index in mismatched_files:
            try:
                ds = pydicom.dcmread(filepath)
                ds.InstanceNumber = correct_index
                ds.save_as(filepath)
                print(f"Corrected Instance Number for {dicom_number}")
            except Exception as e:
                print(f"Error writing {filepath}: {e}")


def add_instance_number(dicom_dir):
    is_Exist(dicom_dir)
    print("Processing file : {}".format(osp.basename(dicom_dir)))
    z_list = []
    dcms = get_full_paths(dicom_dir)
    for dcm in dcms:
        ds = pydicom.dcmread(dcm)
        image_position = ds.ImagePositionPatient
        z_set = (osp.basename(dcm), image_position[-1])
        z_list.append(z_set)
    sorted_z_list = sorted(z_list, key=lambda x: float(x[1]), reverse=True)
    for idx, item in enumerate(sorted_z_list):
        act_idx = idx+1
        file_path = osp.join(dicom_dir, item[0])
        ds = pydicom.dcmread(file_path)
        ds.InstanceNumber = act_idx
        ds.save_as(file_path)


