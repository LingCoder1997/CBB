import radiomics
import os.path as osp
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from CBB.myTools import consist_check
from CBB.myos import find_sample_files, is_Exist, check_path
from feature_extraction import feature_extraction

class FeatureSelector:
    def __init__(self, name="Default", top_N=10, dul_check=False, **kwargs):
        self.name = name
        self.top_N = top_N
        self.dul_check = dul_check
        self.selector = None
        self.features = None

    def _init_selector(self):
        pass

    def _select_features(self, **kwargs):
        self.X = kwargs.get("X", None)
        self.y = kwargs.get("y", None)
        self._init_selector()
        self.selector.fit(self.X, self.y)
        self._get_selected_features()

    def _get_selected_features(self):
        pass

    def _show_features(self):
        if self.features is not None:
            for name, coef in self.features.items():
                print(f"{name}, coef: {coef}")
        else:
            print("No features selected yet.")

class LASSOSelector(FeatureSelector):
    def __init__(self, min_val=-3, max_val=1, cv=5, **kwargs):
        super().__init__(**kwargs)
        self.min_val = min_val
        self.max_val = max_val
        self.cv = cv

    def _init_selector(self):
        alphas = np.logspace(self.min_val, self.max_val, 50)
        self.selector = LassoCV(alphas=alphas, cv=self.cv, max_iter=300000)

    def _get_selected_features(self):
        coef_abs = np.abs(self.selector.coef_)
        feature_ranking = np.argsort(coef_abs)[::-1]
        num_features = self.top_N if isinstance(self.top_N, int) else int(len(self.X.columns) * self.top_N / 100)

        if self.dul_check:
            self.features, selected_features = self.find_unique_top_N(
                coef_list=coef_abs,
                column_names=self.X.columns,
                max_num=num_features
            )
        else:
            feature_ranking = feature_ranking[:num_features]
            selected_features = self.X.columns[feature_ranking]
            self.features = dict(zip(selected_features, self.selector.coef_[feature_ranking]))

    @staticmethod
    def find_unique_top_N(coef_list, column_names, max_num):
        # Implementation of the method to find unique top N features
        feature_dict = {}
        unique_features = []
        sorted_indices = np.argsort(coef_list)[::-1]

        for idx in sorted_indices:
            if len(unique_features) >= max_num:
                break
            if coef_list[idx] not in feature_dict:
                feature_dict[column_names[idx]] = coef_list[idx]
                unique_features.append(column_names[idx])

        return feature_dict, unique_features


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