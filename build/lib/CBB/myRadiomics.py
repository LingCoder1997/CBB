import radiomics
import os.path as osp
import os
import sys
import numpy as np
import pandas as pd

from CBB.myTools import consist_check
from CBB.myos import find_sample_files, is_Exist, check_path
from feature_extraction import feature_extraction

def extract_features(dicom_dir, mask_dir, label_file, save_path=None):
    valid_samples, temp_db = consist_check(dicom_dir, mask_dir, label_file)
    print(valid_samples)

    total_db = None
    success, fail = 0, 0
    for dcm in valid_samples:
        matched_dicom = find_sample_files(dicom_dir,dcm)
        matched_mask = find_sample_files(mask_dir,dcm)
        assert len(matched_mask)==1 and len(matched_dicom)==1, f"Error! sample {dcm} has some problem"

        dicom_path,mask_path = osp.join(dicom_dir,matched_dicom[0]), osp.join(mask_dir, matched_mask[0])
        try:
            is_Exist(dicom_path)
            is_Exist(mask_path)
            print(f"Proessing file {dcm} ...")
            features = feature_extraction(data_path=dicom_path,mask_path=mask_path)
        except:
            print("Feature extraction failed at {}".format(id))
            fail += 1
            continue
        df = pd.DataFrame(features.values(), index=features.keys()).transpose()
        df['name'] = dcm
        total_db = df if total_db is None else pd.concat([total_db, df])
        success += 1

    total_db = pd.merge(left=temp_db, right=total_db, on="name", how="inner")
    if save_path is not None:
        total_db.to_csv(save_path,index=None)
    else:
        save_path = check_path(r"./Extracted_features")
        save_path = osp.join(save_path, "auto.csv")
        total_db.to_csv(save_path,index=None)
        print(f"Feature file saved into {save_path}")

    print("Feature extraction finished!")


if __name__ == '__main__':
    mask_dir = r"D:\Pycharm_workplace\COVID19\COVID_DATA\mask"
    dicom_dir =r"D:\Pycharm_workplace\COVID19\COVID_DATA\new_dcm"
    label_file = r"D:\Pycharm_workplace\COVID19\GT.txt"
    extract_features(dicom_dir,mask_dir,label_file)