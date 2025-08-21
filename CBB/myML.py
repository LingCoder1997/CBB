#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : myML.py
@Time         : 2024/02/14 14:04:50
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the functions that are related to Machine learning algorithms
'''
import pickle

import torch
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from CBB.data_type import variable_type
from CBB.myos import *
from CBB.mydb import *
from CBB.myplot import generate_panel
from CBB.myMetrics import Cal_Metrics
from CBB.myDecorator import timer
import functools
# Reconstruct the customized model class which contains the name of the model, which will be easier to to analysis 
class Custom_SVM(SVC):
    def __init__(self, model_name='CustomSVM', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

class Custom_LR(LogisticRegression):
    def __init__(self, model_name='Custom_LogisticRegression', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

class Custom_RF(RandomForestClassifier):
    def __init__(self, model_name='CustomRF', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

class Custom_XGBoost(xgb.XGBClassifier):
    def __init__(self, model_name='CustomXGB', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

RF_param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

LR_param_grid_L1 = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1'],  # 正则化类型
            'solver': ['liblinear', 'saga'],
            'max_iter':[1000000,100000]
        }

LR_param_grid_L2 = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],  # 正则化类型
            'solver': ['lbfgs', 'newton-cg', 'sag'] ,
        }

SVM_param_grid_SRL = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['sigmoid','rbf','linear'],
            'gamma': [0.01, 0.1, 1, 10],
            'max_iter': [1000, 2000, 3000]
        }
SVM_param_grid_Poly = {
    'C': [0.1, 1, 10],
    'kernel': ['poly'],
    'degree': [2, 3, 4],
    'max_iter': [1000, 2000, 3000]
}

def show_model_coef(model,model_name,X_data,index=0,save_path=None):

    if isinstance(model, RandomForestClassifier) or isinstance(model, Custom_RF):
        feature_importance = model.feature_importances_ 
    elif isinstance(model, LogisticRegression) or isinstance(model, Custom_LR):
        feature_importance = model.coef_
        feature_importance = feature_importance.squeeze()
    else:
        raise TypeError(f"Error! The input model type is {model.__class__.__name__} which is not in the supported type list: ['LogisticRegression','RandomForestClassifier']")
    # 获取原始特征名称
    feature_names = X_data.columns
    
    # 将特征重要性进行排序
    sorted_idx = np.argsort(feature_importance)
    # 绘制水平条形图
    fig, ax = plt.subplots()
    bars = ax.barh(range(len(feature_importance)), feature_importance[sorted_idx], align='center')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Feature Name')
    ax.set_title('RandomForest Feature Importance')
    
    for bar, coef in zip(bars, feature_importance[sorted_idx]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{coef:.2f}', 
                va='center', ha='left', color='red')
    plt.tight_layout()
    plt.show() if save_path is None else plt.savefig(osp.join(save_path,f"{model_name}_coef.jpg"))
    plt.gcf()
    plt.figure(index)

def draw_ROC_lines(models, X_test, y_test,figure_size=(10,6),save_path=None):
    from sklearn.metrics import roc_curve,auc
    plt.figure(figsize = figure_size,dpi=600)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    counter = {}
    for i,model in enumerate(models):
        
        try:
            model_name = model.model_name
        except:
            print("Warning! Get the model name attribute failed, use default class name")
            model_type = model.__class__.__name__
            if model_type not in counter:
                counter[model_type] = 0
                model_name = model_type
            else:
                counter[model_type] += 1
                model_name = model_type + "_{}".format(counter[model_type])

        y_scores_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_scores_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=default_colors[i], lw=2, label='{} ROC curve (area = {:.2f})'.format(model_name, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
        plt.figure()

def generate_train_test_data(data,xkeys,ykey,class_name=None,ratio=0.25,random_state=None):
    from collections import Counter
    from sklearn.model_selection import train_test_split

    keys = xkeys + ykey
    temp_db = data[keys]    
    if if_Nan_Exists(temp_db):
        print(show_nan(temp_db))
        temp_db.dropna(inplace = True)

    X_data = temp_db[xkeys]
    y_data = temp_db[ykey]
    y_data = np.ravel(y_data)
    if len(Counter(y_data)) == 2:
        print("The incoming data is a binary classification")
        class_counts = np.unique(y_data, return_counts=True)
        positive_ratio = class_counts[1][1] / len(y_data)
        negative_ratio = class_counts[1][0] / len(y_data)
        print("Positive / Negative samples ratio: {}({:.4f}) / {}({:.4f})".format(class_counts[1][1],positive_ratio, class_counts[1][0], negative_ratio))
    
    if class_name is None:
        class_name = ["Cls_1","Cls_2"]
    if random_state is not None:
        X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=ratio,random_state=random_state)
    else:
        X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=ratio)

    return X_train, X_test, y_train, y_test

def compute_vif(X,export=False):
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif_data['VIF'] = vif_data["VIF"].round(4)
    if export:
        check_path("./VIF")
        vif_data.to_csv("./VIF/vif.csv",index=None)
    return vif_data
@timer
def ML_analysis(data,xkeys,ykey,model_list,class_name=None,metrics=False,cm=False,ROC=False,save_path=None,show_coef=False,adv_params=[]):
    import seaborn as sns 
    from pprint import pprint
    from sklearn.preprocessing import StandardScaler
    check_path(save_path)
    if isinstance(ykey, str):ykey = [ykey]
    X_train,X_test,y_train,y_test = generate_train_test_data(data=data, xkeys=xkeys,ykey=ykey)

    ss_X = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)

    model_num = len(model_list)
    if model_num == 0:
        print("No model detected! Function exit!")
        return
    name_dict = {}
    if cm:
        cm_fig, cm_axes, rows, cols = generate_panel(num=model_num)   
    if show_coef:
        sc_fig,sc_axes,rows,cols = generate_panel(num=model_num)   

    if cm and show_coef:
        cm_index = 0
        sc_index = 1
    else:
        cm_index = 0
        sc_index = 0
        
    dict_list = []

    for idx, model in enumerate(model_list):
        model_name = get_model_name(model,name_dict)
        print("Proceeding model {} . . .".format(model_name))
        if idx in adv_params:
            try:
                if "scoring" in adv_params[idx]:
                    scoring = adv_params[idx]['scoring']
                    adv_params[idx].pop('scoring')
                else:
                    scoring = 'f1'
                model = GridSearchCV(estimator=model, param_grid=adv_params[idx], cv=5, scoring=scoring, n_jobs=-1)
            except:
                print(f"Advanced setting for model {model_name} is failed!")
                continue
        model.fit(X_train,y_train)

        if isinstance(model,GridSearchCV):
            print(f"Model {model_name} has trained with GridSearch and the best params is as below: ")
            pprint(model.best_params_)
            model = model.best_estimator_
            model_list[idx] = model

        if show_coef:
            try:
                show_model_coef(model=model,model_name=model_name,X_data=X_train,index=sc_index,save_path=save_path)
            except:
                print("Model type {} does not support coef_check".format(model.__class__.__name__))
        result = model.predict(X_test)
        if metrics:
            m = Cal_Metrics(y_test,result)
            m["name"] = model_name 
            dict_list.append(m)
        if cm:
            matrix = confusion_matrix(y_test,result)
            ax = cm_axes[idx // cols, idx % cols] if len(cm_axes.shape) != 1 else cm_axes[idx]
            dataframe = pd.DataFrame(matrix, index=class_name, columns=class_name)
            sns.heatmap(dataframe, annot=True, annot_kws={"size": 20, "weight" : "bold"}, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f'Confusion Matrix of {model_name}')

    plt.tight_layout()
    if metrics:
        mets = pd.DataFrame(dict_list).set_index("name")
        if save_path is None:
            print(mets)
        else:
            mets.to_csv(osp.join(save_path,"metrics.csv") )
    if cm:
        plt.savefig(osp.join(save_path,"./CM_plots.jpg")) if save_path else plt.show()
        plt.figure(cm_index)
    
    if ROC:
        draw_ROC_lines(model_list, X_test=X_test, y_test=y_test,save_path=osp.join(save_path,"ROC_graph.jpg"))

def stats_model_analysis(data,xkeys,ykey):
    import statsmodels.api as sm 
    assert isinstance(data, pd.DataFrame),"Error! The incomming data is not a DataFrame"

    keys = xkeys+ykey if isinstance(ykey,list) else xkeys+[ykey]
    temp_data = data[keys].dropna()
    X_data = temp_data[xkeys]
    y_data = temp_data[ykey] if isinstance(ykey, str) else temp_data[ykey[0]]

    X = sm.add_constant(X_data)
    model = sm.OLS(y_data, X_data)

    results = model.fit()
    print(results.summary())

    return results

def show_ratio(y_data,class_name=None, show_ext_val=False):
    from collections import Counter
    from pprint import pprint
    assert isinstance(y_data, pd.Series) or isinstance(y_data, np.ndarray), "Error! The given data is not in the correct format"

    length = len(y_data)
    val_dict = Counter(y_data)

    ratio_dict = {key : val/length for key, val in val_dict.items()}
    
    if show_ext_val:
        for k,v in ratio_dict.items():
            ratio_dict[k] = (val_dict[k], v)

    if not class_name is None:
        temp_dict = {}
        for k,v in class_name.items():
            if v in ratio_dict.keys():
                temp_dict[k] = ratio_dict[v]
        ratio_dict = temp_dict

    pprint(ratio_dict)
    return ratio_dict

def get_model_name(model,name_dict):
    try:
        model_name = model.model_name
        return model_name
    except:
        print("Get model name failed! Using default name")
        model_cls = model.__class__.__name__
        if model_cls not in name_dict:
            name_dict[model_cls] = 0
            model_name = f'{model_cls}_model'
        else:
            name_dict[model_cls] += 1
            model_name = f'{model_cls}_model_{name_dict[model_cls]}'
        return model_name

def baseline_models():
    model_1 = Custom_LR(model_name="BaseLine_LR",class_weight="balanced")
    model_2 = Custom_RF(model_name="BaseLine_RF",class_weight="balanced")
    model_3 = Custom_SVM(model_name="BaseLine_SVM",class_weight="balanced",probability=True)

    return [model_1,model_2,model_3]


def plot_feature_importance(model, X, feature_names, angle=45):
    """
    绘制逻辑回归模型中每个特征的重要性（系数值）的条形图。

    参数：
    model : 已训练的逻辑回归模型 (LogisticRegression)
    X : 训练数据集特征
    feature_names : 特征名称列表，可选，默认为None，如果为None，则使用默认的X列名
    angle : 特征名称显示的旋转角度，默认为45度
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline

    # 获取逻辑回归模型的系数
    if isinstance(model, Pipeline):
        # 获取Pipeline中最后一步的模型
        model = model.named_steps['classifier']

    if hasattr(model, 'coef_'):
        coefficients = model.coef_[0]
    else:
        raise AttributeError("The model does not have 'coef_' attribute.")

    # 如果没有提供特征名称，则使用默认的列名
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            # 移除 'name' 和 'label' 列
            feature_columns = X.drop(columns=['name', 'label'], errors='ignore')
            feature_names = feature_columns.columns
        else:
            # 如果不是 DataFrame，直接生成特征名
            feature_names = [f'Feature {i + 1}' for i in range(X.shape[1])]

    # 创建一个DataFrame存储特征名和系数值
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })

    # 按照系数的绝对值排序特征
    feature_importance['Abs Coefficient'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values('Abs Coefficient', ascending=False)

    # 绘制条形图
    plt.figure(figsize=(12, 8))
    y_positions = np.arange(len(feature_importance))
    plt.barh(y_positions, feature_importance['Coefficient'], color='skyblue')
    plt.yticks(y_positions, feature_importance['Feature'], rotation=angle, fontsize=10)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance in Logistic Regression', fontsize=14)

    # 添加系数数值到条形图的末端
    for i, coef in enumerate(feature_importance['Coefficient']):
        plt.text(coef, i, f'{coef:.3f}', ha='left' if coef > 0 else 'right',
                 va='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.gca().invert_yaxis()  # 反转y轴，使重要性高的特征在顶部
    plt.show()


def train_and_save_models(
        data,
        db_name=None,
        label_name= None,
        k=20,
        output_file=None,
        no_train=False,
        mode=None,
        test_df =None,
        CM=False,
        ROC=False,
        show_error=False,
        show_youden=False,
        focus=None,
        use_best="No"):
    # Separate features and labels
    result_dir = fr"./result/{db_name}_{mode}_{k}_result"
    if CM or ROC:
        check_path(result_dir)
    data = data.fillna(0)
    X = data.drop(columns=[label_name,"name"])
    y = data[label_name]
    print(f"Find {data.shape[1]} number of features in general")

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    if not no_train:
        if mode=="var":
            var_thresh = VarianceThreshold(threshold=0.01)
            X_var = var_thresh.fit_transform(X_scaled)
            k_best = SelectKBest(score_func=f_classif, k=k)
            X_top_features = k_best.fit_transform(X_var, y)
            selected_features_indices = k_best.get_support(indices=True)
            top_features = X.columns[selected_features_indices]
            X_top_features = pd.DataFrame(X_top_features, columns=top_features, index=X.index)
        elif mode=="RF":
            # Using RF for feature selection
            rf = RandomForestClassifier(n_estimators=100)
            rfe = RFE(estimator=rf, n_features_to_select=k, step=1)
            X_rfe = rfe.fit_transform(X_scaled, y)
            top_features = X.columns[rfe.get_support()]
            X_top_features = pd.DataFrame(X_rfe, columns=top_features, index=X.index)
        elif mode == "LASSO":
            # Using LASSO for feature selection
            min, max = -3, 1
            alphas = np.logspace(min, max, 50)
            lasso = LassoCV(alphas=alphas, cv=5, max_iter=300000)
            lasso.fit(X_scaled, y)
            importance = np.abs(lasso.coef_)
            top_k_indices = np.argsort(importance)[-k:]
            top_features = X.columns[top_k_indices]
            X_top_features = X_scaled[top_features]
        else:
            raise KeyError("Error! Mode {} is not available currently".format(mode))

        models = {
            'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=10000),
            'SVM': SVC(probability=True,class_weight='balanced'),
            'KNN': KNeighborsClassifier(),
            'Neural Network': MLPClassifier(max_iter=10000),
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier()
        }

        # Train and save models
        trained_models = {}
        cross_val_results = {}
        best_thresholds = {}
        for name, model in models.items():
            pipeline = Pipeline([
                ('scaler', StandardScaler()),  # Each model has its own scaler for consistency
                ('classifier', model)
            ])
            cv_scores = cross_val_score(pipeline, X_top_features, y, cv=5, scoring='f1')
            cross_val_results[name] = cv_scores

            pipeline.fit(X_top_features, y)
            trained_models[name] = pipeline
            y_prob = pipeline.predict_proba(X_top_features)[:, 1]
            fpr, tpr, thresholds = roc_curve(y, y_prob)
            youden_index = tpr - fpr
            best_threshold_index = youden_index.argmax()
            best_threshold = thresholds[best_threshold_index]
            best_thresholds[name] = best_threshold
            print(f"{name}: Best Threshold (Youden Index): {best_threshold:.2f}")

        for name, scores in cross_val_results.items():
            print(f"{name} Cross-Validation AUC: {scores.mean():.2f} ± {scores.std():.2f}")

        if output_file is not None:
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'scaler': scaler,
                    'top_10_features': top_features,
                    'models': trained_models,
                    'best_thresholds' : best_thresholds
                }, f)

            print(f"Models and scaler saved to {output_file}")

    if test_df is not None:
        if not isinstance(test_df, pd.DataFrame):
            raise ValueError("test_df should be a pandas DataFrame")

        if no_train:
            with open(output_file, 'rb') as f:
                saved_data = pickle.load(f)
            top_features = saved_data['top_10_features']
            scaler = saved_data['scaler']
            trained_models = saved_data['models']
            best_thresholds = saved_data['best_thresholds']

        # Standardize test features
        X_test = test_df.drop(columns=[label_name])
        if "name" in list(X_test.columns):
            names = X_test["name"]
            X_test = X_test.drop(columns=['name'])
        else:
            names = None

        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        X_test_scaled = X_test_scaled[top_features]
        y_test = test_df[label_name]

        predictions = {}
        metrics = {}

        if CM:
            if focus is None:
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
                axes = axes.flatten()
            else:
                fig = plt.figure(figsize=(10,8))
                ax = fig.add_subplot(111)

        if ROC:
            if not focus:
                fig_roc, axes_roc = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
                axes_roc = axes_roc.flatten()
                roc_files = []
            else:
                fig_roc = plt.figure(figsize=(10,8))
                axes_roc = fig_roc.add_subplot(111)

        for i, (name, model) in enumerate(trained_models.items()):
            if show_error:
                status_dict = {
                    "name": [],
                    "status": []
                }
                status_map = {0: "TP", 1: "TN", 2: "FP", 3: "FN"}

            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            if use_best == "no":
                optimal_threshold = best_thresholds[name]
            elif use_best == "default":
                optimal_threshold = 0.5
            elif use_best == "yes":
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                specificity = 1 - fpr
                youden_index = tpr + specificity - 1
                optimal_idx = np.argmax(youden_index)
                optimal_threshold = thresholds[optimal_idx]

            y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
            # y_pred = model.predict(X_test_scaled)
            predictions[name] = y_pred_optimal

            cm = confusion_matrix(y_test, y_pred_optimal)
            tn, fp, fn, tp = cm.ravel()
            cm_reordered = np.array([[tp, fn], [fp, tn]])

            if show_error:

                # Identifying the indices of mispredicted samples
                mispredicted_indices = y_test.index[y_test != y_pred_optimal]
                mispredicted_samples = test_df.loc[mispredicted_indices]

                if names is not None:
                    mispredicted_names = names.loc[mispredicted_indices]
                else:
                    mispredicted_names = mispredicted_indices

                # Determine which are FP and FN
                fp_indices = y_test.index[(y_test == 0) & (y_pred_optimal == 1)]
                fn_indices = y_test.index[(y_test == 1) & (y_pred_optimal == 0)]
                tp_indices = y_test.index[(y_test == 1) & (y_pred_optimal == 1)]
                tn_indices = y_test.index[(y_test == 0) & (y_pred_optimal == 0)]

                if name == "Random Forest":
                    print("fp_indices : {}\nfn_indices : {}\ntp_indices : {}\ntn_indices : {}".format(len(fp_indices),len(fn_indices),len(tp_indices),len(tn_indices)))
                fp_names = names.loc[fp_indices] if names is not None else fp_indices
                fn_names = names.loc[fn_indices] if names is not None else fn_indices
                tp_names = names.loc[tp_indices] if names is not None else tp_indices
                tn_names = names.loc[tn_indices] if names is not None else tn_indices

                status_dict["name"].extend(tp_names)
                status_dict["status"].extend([0] * len(tp_names))

                status_dict["name"].extend(tn_names)
                status_dict["status"].extend([1] * len(tn_names))  # TN

                status_dict["name"].extend(fp_names)
                status_dict["status"].extend([2] * len(fp_names))  # FP

                status_dict["name"].extend(fn_names)
                status_dict["status"].extend([3] * len(fn_names))  # FN
                status_df = pd.DataFrame(status_dict)
                status_df["status"] = status_df["status"].map(status_map)

                status_df.to_csv(osp.join(result_dir, f"{name}_pred_status.csv"),index=None)
                # Display the results in a more readable format
                print(f"\n{'=' * 40}")
                print(f"Model: {name}")
                print(f"{'=' * 40}")
                print(f"Total Mispredicted Samples: {len(mispredicted_indices)}")
                print(f"False Positives (FP): {len(fp_indices)}")
                print(f"False Negatives (FN): {len(fn_indices)}")

                print("\nFalse Positives (FP) Samples:")
                print(fp_names.to_list())

                print("\nFalse Negatives (FN) Samples:")
                print(fn_names.to_list())

                print("\nTrue Positives (TP) Samples:")
                print(tp_names.to_list())

                print("\nTrue Negatives (TN) Samples:")
                print(tn_names.to_list())
                print(f"{'=' * 40}\n")

            if CM:
                if focus is None:
                    ax = axes[i]
                    cax = ax.matshow(cm_reordered, cmap='Blues')
                    fig.colorbar(cax, ax=ax)
                    ax.set_title(f'Confusion Matrix for {name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(['Positive', 'Negative'])
                    ax.set_yticklabels(['Positive', 'Negative'])
                    for (j, k), val in np.ndenumerate(cm_reordered):
                        ax.text(k, j, f'{val}', ha='center', va='center',
                                color='white' if val > cm_reordered.max() / 2 else 'black')
                elif focus == name:
                    cax = ax.matshow(cm_reordered, cmap='Blues')
                    fig.colorbar(cax, ax=ax)
                    ax.set_title(f'Confusion Matrix for {name}')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_xticks([0, 1])
                    ax.set_yticks([0, 1])
                    ax.set_xticklabels(['Positive', 'Negative'])
                    ax.set_yticklabels(['Positive', 'Negative'])
                    for (j, k), val in np.ndenumerate(cm_reordered):
                        ax.text(k, j, f'{val}', ha='center', va='center',
                                color='white' if val > cm_reordered.max() / 2 else 'black')


            if ROC:
                default_threshold = 0.5
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)

                specificity = 1 - fpr
                youden_index = tpr + specificity - 1
                optimal_idx = np.argmax(youden_index)
                optimal_threshold_valid = thresholds[optimal_idx]
                valid_point = (fpr[optimal_idx], tpr[optimal_idx])

                default_threshold = 0.5
                train_idx = np.argmin(np.abs(thresholds - default_threshold))
                train_point = (fpr[train_idx], tpr[train_idx])

                metrics[name] = {
                    "Optimal threshold (validation)": optimal_threshold_valid,
                    "Optimal threshold (training)": default_threshold,
                    "Youden's index": optimal_idx,
                    'Sensitivity': tpr[optimal_idx],
                    'Specificity': specificity[optimal_idx],
                    "AUC" : roc_auc
                }

                if focus is None:
                    ax = axes_roc[i]
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (area = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

                    if train_point is not None:
                        if show_youden:
                            ax.scatter(*train_point, color='red', label=f'Default Youden Index', zorder=5)
                            ax.annotate(
                                f'Default\n({train_point[0]:.2f}, {train_point[1]:.2f})',
                                xy=train_point,
                                xytext=(-50, 30),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", lw=1.5, color='red'),
                                color='red'
                            )
                            ax.scatter(*valid_point, color='blue', label=f'Validation Youden Index', zorder=5)
                            ax.annotate(
                                f'Validation\n({valid_point[0]:.2f}, {valid_point[1]:.2f})',
                                xy=valid_point,
                                xytext=(-50, -40),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'),
                                color='blue'
                            )

                    # 设置图形属性
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve for {name}')
                    ax.legend(loc='lower right')

                elif focus == name:
                    axes_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'{name} (area = {roc_auc:.2f})')
                    axes_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

                    if train_point is not None:
                        if show_youden:
                            axes_roc.scatter(*train_point, color='red', label=f'Default Youden Index', zorder=5)
                            axes_roc.annotate(
                                f'Default\n({train_point[0]:.2f}, {train_point[1]:.2f})',
                                xy=train_point,
                                xytext=(-50, 30),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", lw=1.5, color='red'),
                                color='red'
                            )
                            axes_roc.scatter(*valid_point, color='blue', label=f'Validation Youden Index', zorder=5)
                            axes_roc.annotate(
                                f'Validation\n({valid_point[0]:.2f}, {valid_point[1]:.2f})',
                                xy=valid_point,
                                xytext=(-50, -40),
                                textcoords='offset points',
                                arrowprops=dict(arrowstyle="->", lw=1.5, color='blue'),
                                color='blue'
                            )

                    # 设置图形属性
                    axes_roc.set_xlim([0.0, 1.0])
                    axes_roc.set_ylim([0.0, 1.05])
                    axes_roc.set_xlabel('False Positive Rate')
                    axes_roc.set_ylabel('True Positive Rate')
                    axes_roc.set_title(f'ROC Curve for {name}')
                    axes_roc.legend(loc='lower right')

                    # 保存图像

        if CM:
            plt.tight_layout()
            plt.savefig(osp.join(result_dir,"CM.jpg"))
            plt.close()

        if ROC:
            plt.tight_layout()
            plt.savefig(osp.join(result_dir,"ROC.jpg"))
            plt.close()

        if metrics:
            metrics = pd.DataFrame(metrics).T
            metrics.to_csv(osp.join(result_dir,"metrics.csv"))

        if test_df is None:
            return predictions, None
        else:
            return predictions, metrics


