import radiomics
import os.path as osp
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import levene, ttest_ind
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
        self._init_selector()
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

def select_features(X, y, FS_mode, mode="top-5", **kwargs):
    assert FS_mode in ['LASSO', 'TTS', 'RELF', 'GNRO', 'FAOV',
                       'FSCR'], "Error! The given key: {} is not supported currently".format(FS_mode)
    assert mode in ['top-5', 'top-10', 'top-5%'], "Error! The given mode {} is not currently not supported".format(mode)
    export_feature = kwargs.get("show_features", False)
    dul_check = kwargs.get("dul_check", False)

    if FS_mode == "LASSO":
        min, max = kwargs.get('min', -3), kwargs.get('max', 1)
        cv = kwargs.get("cv", 5)
        alphas = np.logspace(min, max, 50)
        model = LassoCV(alphas=alphas, cv=cv, max_iter=300000).fit(X, y)
        print("LASSO alpha = {}".format(model.alpha_))

        selected_features = X.columns[model.coef_ != 0]
        if len(selected_features) <= 3:
            print(f"Warning! The LASSO extracted features number {len(selected_features)} is below 3")
        coef_abs = np.abs(model.coef_)
        feature_ranking = np.argsort(coef_abs)[::-1]
        features = X.columns
        num_features = selected_features_number(features, mode) if len(selected_features) > 3 else len(
            selected_features)

        if dul_check:
            feature_dict, selected_features = find_unique_top_N(coef_list=coef_abs, column_names=features,
                                                                max_num=num_features)
        else:
            feature_ranking = feature_ranking[:num_features]
            selected_features = X.columns[feature_ranking]
            feature_dict = dict(zip(X.columns[feature_ranking], model.coef_[feature_ranking]))

        for name, coef in feature_dict.items():
            print(f"{name}, coef: {coef}")
        if export_feature:
            return selected_features, feature_dict
        else:
            return selected_features

    elif FS_mode == "TTS":
        selected_features = {}
        features = X.columns
        for feature in features:
            group1 = X[y == 0][feature]
            group2 = X[y == 1][feature]

            levene_test = levene(group1, group2)
            if levene_test.pvalue < 0.05:
                t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            else:
                t_stat, p_value = ttest_ind(group1, group2)
            if p_value < 0.05:
                selected_features[feature] = p_value

        selected_features = dict(sorted(selected_features.items(), key=lambda x: x[1], reverse=False))
        num_features_selected = len(selected_features)

        if num_features_selected <= 3:
            print(f"Warning! The T-test selected features number is {num_features_selected} below 3")

        num_features = selected_features_number(features, mode) if num_features_selected > 3 else num_features_selected
        if export_feature:
            if dul_check:
                feature_info, selected_features = find_unique_top_N(
                    coef_list=selected_features.values(),
                    column_names=selected_features.keys(),
                    max_num=num_features, reverse=False)
            else:
                feature_info = {k: selected_features[k] for k in list(selected_features)[:num_features]}
            return pd.Index(selected_features), feature_info
        else:
            return pd.Index(selected_features)

    elif FS_mode == "RELF":
        from skrebate import ReliefF
        from sklearn.feature_selection import SelectKBest
        relief = ReliefF()
        features = X.columns
        # Fit ReliefF to the data
        relief.fit(X.values, y)

        # Get feature scores
        feature_scores = relief.feature_importances_

        # Select the top features based on their scores
        num_selected_features = selected_features_number(features, mode)

        if dul_check:
            feature_dict, selected_features = find_unique_top_N(
                coef_list=feature_scores,
                column_names=features,
                max_num=num_selected_features)
        else:
            selected_feature_indices = np.argsort(feature_scores)[::-1][:num_selected_features]
            selected_features = X.columns[selected_feature_indices]
            feature_dict = {X.columns[idx]: feature_scores[idx] for idx in selected_feature_indices}

        print(f"RELIEF extracted {len(selected_features)} features")
        print(selected_features)
        if export_feature:
            return selected_features, feature_dict
        else:
            return selected_features

    elif FS_mode == "GNRO":
        from sklearn.feature_selection import mutual_info_classif
        features = X.columns
        feature_scores = mutual_info_classif(X, y)
        k = selected_features_number(features, mode)
        if dul_check:
            feature_info, selected_features = find_unique_top_N(
                coef_list=feature_scores,
                column_names=features,
                max_num=k)
        else:
            selected_feature_indices = np.argsort(feature_scores)[::-1][:k]
            selected_features = X.columns[selected_feature_indices]
            feature_info = {features[idx]: feature_scores[idx] for idx in selected_feature_indices}
        print(f"GNRO extracted {len(selected_features)} features")
        print(selected_features)

        if export_feature:
            return selected_features, feature_info
        else:
            return selected_features
    elif FS_mode == "FAOV":
        from sklearn.feature_selection import SelectKBest, f_classif
        features = X.columns
        k = selected_features_number(features, mode)
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        feature_scores = selector.scores_

        if dul_check:
            feature_info, selected_features = find_unique_top_N(
                coef_list=feature_scores,
                column_names=features,
                max_num=k)
        else:
            selected_feature_indices = selector.get_support(indices=True)
            selected_features = X.columns[selected_feature_indices]
            feature_info = dict(zip(selected_features, feature_scores[selected_feature_indices]))
        print(f"F-ANOVA extracted {len(selected_features)} features")
        print(selected_features)

        if export_feature:
            return selected_features, feature_info
        else:
            return selected_features

    elif FS_mode == "FSCR":
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.feature_selection import chi2
        features = X.columns
        k = selected_features_number(features, mode)
        MM_scaler = MinMaxScaler()
        X_scaled = MM_scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        fisher_scores, p_values = chi2(X_scaled, y)
        significant_fisher_scores = fisher_scores[p_values < 0.05]
        significant_indices = np.where(p_values < 0.05)[0]
        select_features = features[significant_indices]
        if dul_check:
            feature_info, selected_features = find_unique_top_N(
                coef_list=significant_fisher_scores,
                column_names=select_features,
                max_num=k)
        else:
            sorted_indices = np.argsort(significant_fisher_scores)[::-1]
            selected_feature_indices = sorted_indices[:k]

            selected_features = X.columns[selected_feature_indices]
            selected_scores = fisher_scores[selected_feature_indices]
            feature_info = dict(zip(selected_features, selected_scores))
        if export_feature:
            return selected_features, feature_info
        else:
            return selected_features
    else:
        raise KeyError("Error! The given key {} was not included in current version".format(FS_mode))

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

def selected_features_number(overall_features, mode):
    if mode == "top-5":
        return 5
    elif mode == "top-10":
        return 10
    elif mode == "top-5%":
        num_features = max(int(len(overall_features) * 0.05), 3)
        return num_features
    else:
        raise KeyError("Error! mode '{}' is not valid!".format(mode))

def find_unique_top_N(coef_list, column_names, max_num, reverse=True):
    assert len(coef_list) == len(column_names),"Error! The given coef {} is not the same length as the names {}".format(len(coef_list), len(column_names))
    feature_coef_dict = {name : coef for name, coef in zip(column_names,coef_list) }
    feature_coef_dict = sorted(feature_coef_dict.items(), key=lambda x: x[1], reverse=reverse)

    if len(feature_coef_dict) <= max_num:
        print("Warning! ThE OVERALL length is only {} cannot find {} number of top features".format(len(feature_coef_dict), max_num))
        temp_buffer,selected_names = filter_unique_names(feature_coef_dict, max_num)
        return temp_buffer,selected_names

    temp_buffer = feature_coef_dict[:max_num-1]
    feature_coef_dict = feature_coef_dict[max_num-1:]
    names = None
    while len(temp_buffer) < max_num and len(feature_coef_dict) != 0:
        addon = feature_coef_dict.pop(0)
        temp_buffer.append(addon)
        temp_buffer,names = filter_unique_names(temp_buffer, max_num)
    return temp_buffer, names

def filter_unique_names(feature_list,N):
    selected_features = []
    selected_names = []
    name_list = []
    for feature, coef in feature_list:
        if len(selected_features) >= N:
            break
        parts = feature.split('_')
        name = parts[-1]
        if name not in name_list:
            selected_features.append((feature, coef))
            selected_names.append(feature)
            name_list.append(name)

    return selected_features,selected_names


if __name__ == '__main__':
    mask_dir = r"D:\Pycharm_workplace\COVID19\COVID_DATA\mask"
    dicom_dir =r"D:\Pycharm_workplace\COVID19\COVID_DATA\new_dcm"
    label_file = r"D:\Pycharm_workplace\COVID19\GT.txt"
    extract_features(dicom_dir,mask_dir,label_file)