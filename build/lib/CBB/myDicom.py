from CBB.myos import is_Exist
import pydicom
from pydicom.uid import generate_uid
from pydicom.uid import UID
import os
import os.path as osp
import sys
import numpy as np


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
            ds.Modality = ds.get('Modality', 'CT')

            # 检查并添加其他必要标签
            if not hasattr(ds, 'PatientID'):
                ds.PatientID = 'Unknown'
            if not hasattr(ds, 'PatientName'):
                ds.PatientName = 'Unknown'
            if not hasattr(ds, 'StudyDate'):
                ds.StudyDate = '20240101'  # Example date
            if not hasattr(ds, 'SeriesDate'):
                ds.SeriesDate = '20240101'  # Example date

            # 保存修改后的 DICOM 文件
            ds.save_as(file_path)

    print("All DICOM files processed.")