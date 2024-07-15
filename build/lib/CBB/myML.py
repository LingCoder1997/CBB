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

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

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

def NN_Analysis(data,xkeys,ykey,con_list = None,num_epochs=100, ratio=0.25,test=False,
                metrics=False,ROC=False, save_path=None, random_state=42):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    for k in xkeys:
        if k not in data.columns:
            raise KeyError("Error! The given key {} is not included in the column names".format(k))
    if isinstance(ykey, str):
        ykey = [ykey]
    keys = xkeys + ykey
    temp_db = data[keys]
    if if_Nan_Exists(temp_db):
        print(show_nan(temp_db))
        temp_db.dropna(inplace=True)

    scaler = StandardScaler()
    if con_list is None:
        con_list = []
        for col in xkeys:
            if variable_type(data[col]) == "Cat":
                continue
            else:
                con_list.append(col)

    data[con_list] = scaler.fit_transform(data[con_list])
    X = temp_db[xkeys].values
    y = temp_db[ykey].values  # lung_nodule_label 是是否有大于100立方毫米的肺结节的标签

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    if test ==False:
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=ratio, random_state=random_state)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(X_tensor, y_tensor, test_size=0.4, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)

    input_size = len(xkeys)  # 除去标签列
    model = Net(input_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_f1_score = 0.0
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())

        val_predictions = [1 if pred >= 0.5 else 0 for pred in val_predictions]
        current_f1_score = f1_score(val_targets, val_predictions)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss.item()}, F1 Score: {current_f1_score}')

        # 保存表现最好的模型
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_model_state = model.state_dict()

    # 保存分数最高的模型
    if test==True:
        test_predictions = []
        test_targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                test_predictions.extend(outputs.cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())

        test_predictions = [1 if pred >= 0.5 else 0 for pred in test_predictions]
        test_f1_score = f1_score(test_targets, test_predictions)

        print(f'Final Test F1 Score: {test_f1_score}')

        # 保存分数最高的模型
        if save_path is None:
            save_path = check_path("./DN_result")

        torch.save(best_model_state, osp.join(save_path,'best_model.pth'))

        if metrics:
            conf_matrix = confusion_matrix(test_targets, test_predictions)
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='white')
            plt.title('Confusion Matrix')
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.savefig(osp.join(save_path,"confusion_matrix.jpg"))

        if ROC:
            from sklearn.metrics import roc_curve,auc
            # 绘制ROC曲线
            fpr, tpr, thresholds = roc_curve(test_targets, test_predictions)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()
            plt.savefig(osp.join(save_path,"ROC.jpg"))

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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
class Net(nn.Module):
    def __init__(self,input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


