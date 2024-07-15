#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : myMetrics.py
@Time         : 2024/02/14 14:15:08
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the functions that is useful to ML metrics calculations
'''
import csv

import pandas as pd 
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import numpy as np 
from CBB.data_type import *
from CBB.mydb import if_Nan_Exists, if_Nan_Exists
from CBB.myos import check_path

def cal_Edistance(p1, p2):
    '''

    :param p1: point one in format numpy.ndarray
    :param p2:
    :return:
    '''
    if not is_numpy(p1) and is_numpy(p2):
        raise TypeError("Error! The given points are not in numpy.ndarray type")
    if not len(p1) == 2 and len(p2) == 2:
        raise DataShapeError("Error! The given two points must in shape [0,2]")
    return np.linalg.norm(p1 - p2)

def calculate_hausdorff(c1, c2, get_max=False):
    from scipy.spatial.distance import directed_hausdorff
    """
    description: This function will calculate the hausdorff distance of two contours
    :param c1: C1 is the first input contour set
    :param c2: C2 is the second input contour set
    :param get_max: is the bool controller which will tell if the function needs to return the max distance or both
                    direction
    :return: res1, res2:
            res1 is the hausdorff from contour 1 to contour 2
            res2 is the hausdorff from contour 2 to contour 1
    """
    res1 = directed_hausdorff(c1, c2)
    res2 = directed_hausdorff(c2, c1)
    if get_max:
        return res1 if res1[0] > res2[0] else res2
    else:
        return res1, res2

def dice_coef(y_true, y_pred):
    assert np.max(y_true) == 1 and np.max(y_pred) == 1, "Error! The given matrix is not in 0/1"
    smooth = 1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def Cal_specificity(targets, preds):
    
    conf_matrix = confusion_matrix(targets, preds)
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]

    macro_specificity = (tn / (tn + fp)).mean()
    micro_specificity = tn / (tn + fp + 1e-15)
    
    return macro_specificity, micro_specificity

def Cal_sensitivity(targets, preds):
    conf_matrix = confusion_matrix(targets, preds)
    tp = conf_matrix.diagonal()
    fn = conf_matrix.sum(axis=1) - tp

    macro_sensitivity = (tp / (tp + fn)).mean()
    micro_sensitivity = tp.sum() / (tp.sum() + fn.sum() + 1e-15)

    return macro_sensitivity, micro_sensitivity

def Cal_Metrics(targets,preds,multi=None):
    mode = "binary" if multi is None else 'macro'
    accuracy = accuracy_score(targets,preds)
    precision = precision_score(targets,preds,average=mode)
    f1 = f1_score(targets,preds, average=mode)
    macro_sensitivity, micro_sensitivity = Cal_sensitivity(targets,preds)
    macro_specificity, micro_specificity = Cal_specificity(targets,preds)

    result = {
        "accuracy" : accuracy,
        "precision" : precision,
        "f1_score" : f1,
        "Macro sensitivity" : macro_sensitivity, 
        "Micro sensitivity" : micro_sensitivity,
        "Macro specificity" : macro_specificity, 
        "Micro specificity" : micro_specificity,
    }

    return result


def calculate_mean_rsd(matrices):
    if not isinstance(matrices, list):
        raise TypeError("Input must be a list of matrices.")
    # 将四个矩阵堆叠在一起形成一个4x6x8的数组

    first_shape = matrices[0].shape
    for i, matrix in enumerate(matrices):
        if not isinstance(matrix, np.ndarray):
            raise TypeError(f"Matrix {i + 1} is not a numpy array.")
        if matrix.shape != first_shape:
            raise ValueError(f"Matrix {i + 1} does not have the same shape as the first matrix.")

    stacked_matrices = np.stack(matrices)

    # 计算每个位置的均值
    mean_matrix = np.mean(stacked_matrices, axis=0)

    # 计算每个位置的标准偏差
    std_matrix = np.std(stacked_matrices, axis=0)

    # 计算每个位置的相对标准偏差（RSD）
    rsd_matrix = (std_matrix / mean_matrix) * 100

    return mean_matrix, rsd_matrix

def calculate_mean_rsd_from_csv(files):
    # 读取所有 CSV 文件并将它们转换为 DataFrame
    dfs = [pd.read_csv(file, index_col=0) for file in files]

    # 确保所有 DataFrame 形状一致
    first_shape = dfs[0].shape
    for i, df in enumerate(dfs):
        if df.shape != first_shape:
            raise ValueError(f"DataFrame {i + 1} does not have the same shape as the first DataFrame.")

    # 将 DataFrame 转换为 NumPy 数组并堆叠为一个 3D 数组
    stacked_arrays = np.stack([df.to_numpy() for df in dfs])

    # 计算每个位置的均值
    mean_matrix = np.mean(stacked_arrays, axis=0)

    # 计算每个位置的标准偏差
    std_matrix = np.std(stacked_arrays, axis=0)

    # 计算每个位置的相对标准偏差（RSD）
    rsd_matrix = (std_matrix / mean_matrix) * 100

    # 将结果转换为 DataFrame
    mean_df = pd.DataFrame(mean_matrix, columns=dfs[0].columns, index=dfs[0].index)
    rsd_df = pd.DataFrame(rsd_matrix, columns=dfs[0].columns, index=dfs[0].index)

    return mean_df, rsd_df

# This function is usually used on dataset whose size is below 30
def Cal_CI_30(data,score=0.95):
    mean = np.mean(data)
    std_error = stats.sem(data)

    lower,upper = stats.t.interval(score,len(data)-1,loc=mean,scale=std_error)
    return lower, upper

# This function is usually used on dataset whose size is beyond 30
def Cal_CI(data,score=0.95):
    mean = np.nanmean(data)
    std_error = stats.sem(data)
    lower, upper = stats.norm.interval(0.95, loc=mean, scale=std_error)
    return lower, upper

def pearson_correlation_multiple(*arrays):
    arrays = [np.array(arr) for arr in arrays]
    if len(set(len(arr) for arr in arrays)) > 1:
        raise ValueError("The input array does not have the same length!")
    correlation_matrix = np.corrcoef(arrays)
    return correlation_matrix


def pearson_correlation_spss_style(dataframe, columns, alpha=0.05, export_path=None):
    from scipy.stats import pearsonr
    # 从DataFrame中提取指定列
    N = len(dataframe)
    arrays = [dataframe[column].values for column in columns]

    # 检查输入数组的长度是否一致
    if len(set(len(arr) for arr in arrays)) > 1:
        raise ValueError("输入数组长度不一致")

    # 计算相关系数矩阵和p-值矩阵
    correlation_matrix, p_value_matrix = np.corrcoef(arrays), np.zeros_like(np.corrcoef(arrays))

    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            r, p_value = pearsonr(arrays[i], arrays[j])
            correlation_matrix[i, j], correlation_matrix[j, i] = r, r
            p_value_matrix[i, j], p_value_matrix[j, i] = p_value, p_value

    # 获取变量名称
    variable_names = [f"{column}" for column in columns]

    # 创建DataFrame用于存储相关性和显著性水平
    df_correlation = pd.DataFrame(correlation_matrix, index=variable_names, columns=variable_names)
    df_p_values = pd.DataFrame(p_value_matrix, index=variable_names, columns=variable_names)

    # 将相关系数矩阵和显著性水平合并到一个DataFrame中
    df_result = pd.DataFrame(index=variable_names, columns=variable_names)

    for i in range(len(variable_names)):
        for j in range(i, len(variable_names)):
            r = df_correlation.iloc[i, j]
            p_value = df_p_values.iloc[i, j]

            # 根据显著性水平添加 "*" 或 " " 到相关系数矩阵
            correlation_with_significance = f"{r:.4f}"

            correlation_with_significance += "* ({:e})".format(p_value)

            df_result.iloc[i, j] = correlation_with_significance
            df_result.iloc[j, i] = correlation_with_significance
    df_result['mean'] = dataframe[variable_names].mean()
    df_result['Std. Deviation'] = dataframe[variable_names].std()
    
    if export_path is not None:
        df_result.to_csv(export_path, index=variable_names)
    return df_result

def KsNormDetect(data):
    """This function will autoatically calculate if the distribution of the data is in normalization

    Args:
        data (numpy.ndarray/pd.Series): The incoming 1-D Array like data
    """
    from scipy.stats import kstest
    u = np.nanmean(data)
    # 计算标准差
    std = np.nanstd(data)
    # 计算P值
    print(kstest(data, 'norm', (u, std)))
    res = kstest(data, 'norm', (u, std))[1]
    print('均值为：%.2f, 标准差为：%.2f' % (u, std))
    # 判断p值是否服从正态分布，p<=0.05 拒绝原假设 不服从正态分布
    if res <= 0.05:
        print('该列数据不服从正态分布')
        print("-" * 66)
        return True
    else:
        print('该列数据服从正态分布')
        return False
    
def Cal_IQR(data,low=0.25,high=0.75,show_graph=False,save=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from myos import auto_save_file

    if show_graph==False:
        save=False
    data = sorted(data)
    top = np.quantile(data,high)
    down = np.quantile(data,low)
    outlier_top = top+1.5*(top-down)
    outlier_bot = down-1.5*(top-down)
    print("QRI-25: {} / QRI-75: {}".format(down,top))
    if show_graph:
        sns.boxplot(data,palette="Blues")
        if save:
            plt.savefig(auto_save_file("./Box_graph.jpg"))
        else:
            plt.show()
    return down,top

def Cal_P_score(g1,g2,mode="default"):
    """This function will calculate the p score between group 1 and group 2, to check whether these two groups 
       are different in statistical field.

    Args:
        g1 (numpy.ndarray): The first input group with 1-D array
        g2 (numpy.ndarray): The second input group with 1-D array 
    """

    from scipy import stats

    assert isinstance(g1, np.ndarray) and isinstance(g2, np.ndarray), "Error! The input data should be numpy.ndarray not {} and {}".format(type(g1),type(g2))
    # Run the levene test first to check whether the two groups data have the same homogeneity of variances
    st, pv = stats.levene(g1, g2)
    if pv < 0.05:
        print("Warning! The given two sets of data does not pass the Levene test, p = {}".format(pv))
        equal_var = False
    else:
        print("The two sets of data pass the Levene test, p = {}".format(pv))
        print("Run student t-test only")
        equal_var = True

    if mode == "default":
        t,p = stats.ttest_ind(g1,g2, equal_var= equal_var)      # If the data which does not assume equal population variance then the ttest will be Welch's t-test
    elif mode.lower() == "mann" or mode == "Mann-Whitney" and equal_var == "False":
        t,p = stats.mannwhitneyu(g1,g2,alternative="two-sided")
    else:
        raise ValueError("Error! The test mode {} is not supported in the current version".format(mode))
    
    if mode == "default" and equal_var:
        mode = "T-test"
    elif mode == "default" and not equal_var:
        mode = "Welch's"
    else:
        mode = mode

    print("After the {} test, the result is t: {}; p: {}".format(mode,t,p))
    return t,p 

def iqr(x):
    q75, q25 = np.percentile(x, [75, 25])
    return q75 - q25

def chi_square_test(data,observe_keys,prob=0.95):
    """This function will calculate the Chi-square-test based on the key given to the contigency table 

    Args:
        data (pd.Dataframe): The contigency table of the original database
        observe_keys (list:string): The observed columns needed for chi-square
        prob (float, optional): The confident interval. Defaults to 0.95.
    """

    assert isinstance(data, pd.DataFrame),"Error! The given data is not in the type of Dataframe"
    observed_data = data[observe_keys]
    from scipy.stats import chi2
    from scipy.stats import chi2_contingency

    stat,p,dof,expected = stats.chi2_contingency(observed=observed_data)
    print("\n####################### Chi-Square result of {} #######################".format(observe_keys))
    print('dof=%d'%dof)
    critical = chi2.ppf(prob,dof)
    print('probality=%.3f,critical=%.3f,stat=%.3f '%(prob,critical,stat))
    if abs(stat)>=critical:
        print('reject H0:Dependent')
    else:
        print('fail to reject H0:Independent')

    alpha = 1-prob
    print('significance=%.3f,p=%.3f'%(alpha,p))
    if p<alpha:
        print('reject H0:Dependent')
    else:
        print('fail to reject H0:Independent')
    
    return stat,p

def two_individual_ttest(data,xkey,ykey):
    from collections import Counter
    if if_Nan_Exists(data[[xkey,ykey]]):
        data = data[[xkey,ykey]].dropna()

    if not is_cate(data[xkey]):
        print("The given X-data does not seems like a categorical data! Function returned!")
        return None
    else:
        cates = Counter(data[xkey]).keys()
        cp_dict = {}
        for k in cates:
            print(k)
            temp_list = data[data[xkey] == k][ykey].values
            cp_dict[k] = temp_list
        if not len(cp_dict) == 2:
            print("This categorical data comparison contains more than two categories, which is not supoorted yet")
            return None
        else:
            t,p = Cal_P_score(list(cp_dict.values())[0], list(cp_dict.values())[1], mode="default")
            print("stat: {}; p-value: {}".format(t,p))

def make_xmap(l):
    from numpy import array
    ll = len(l)
    dy = 1.0 / (ll-1)
    def f(l, i):
        if i == 0 : return "0.0"
        y0 = i*dy-dy
        x0, x1 = l[i-1:i+1]
        return '%r+%r*(x-%r)/%r'%(y0,dy,x0,x1-x0)
    fmt = 'numpy.where(x<%f,%s%s'
    body = ' '.join(fmt%(j,f(l,i),"," if i<(ll-1) else ", 1.0") for i, j in enumerate(l))
    tail = ')'*ll
    def xm(x):
        x = array(x)
        return eval(body+tail)
    return xm


def generate_S_table(data, ykey, xkeys, cate_list=None, median=False, save_path=None):
    """
    This is the function that will generate all the statistic result that is useful according to the keys that is given
    to function. This function has integrated several statistic calculation methods that might be useful for statistical
    analysis.

    For example:    Chi-square test (For categorical variables)
                    T-test (For continuous data satisfy homogeneity of variances)
                    Welch's (ANOVA) Test (For continuous data that does not satisfy homogeneity of variances)
                    ANOVA Test (Normal ANOVA test for variables that satisfy the homogeneity of variances)

    :param data:    The Overall database that contains the keys
    :param ykey:    The independent keys for the datbase
    :param xkeys:   The dependent variable which is normally in categorical type
    :param cate_list:   The list that contains the types of the xkeys (Optional : but recommend to provide)
    :param save_path:   The saving path for the result (With xlsx / xls ending)
    :return: pd.DataFrame   The statistic table for all variables that are been calculated
    """
    from scipy.stats import ttest_ind
    from scipy.stats import chi2_contingency
    from scipy.stats import f_oneway, levene, kruskal

    assert isinstance(data, pd.DataFrame), "Error! The incoming data is not a type of dataframe"
    assert isinstance(xkeys, list), "Error! The incoming xkeys are supposed in a list"
    if not isinstance(ykey, list):
        ykey = [ykey]
    for k in ykey + xkeys:
        if k not in data.columns:
            raise KeyError(f"Error! The key: {k} is not in the dataframe")

    if cate_list is None or len(cate_list) == 0:
        print("Warning! The category list is not given, Predicting the type of the keys......")
        cate_list = []
        for k in xkeys:
            d = data[k]
            print("Key : {} / ({})".format(k, variable_type(d)))

            cate_list.append(variable_type(d))
    temp = pd.DataFrame(data[ykey].value_counts().rename(ykey[0]).reset_index(name="Count"))
    temp = temp.set_index(ykey)
    freq = temp['Count'].sum()
    temp['Count'] = temp['Count'].apply(lambda x: f"{x}({x / freq * 100:.2f}%)")
    output_df = temp.T

    for idx, k in enumerate(xkeys):
        out_row = None
        sub_cate = [k] + ykey
        if cate_list[idx] == "Cat":
            frequency_table = data.groupby(sub_cate).size().reset_index(name='count')
            pivot_table = frequency_table.pivot(index=k, columns=ykey[0], values='count').fillna(0)

            percentage_table = pivot_table.apply(lambda x: x / x.sum() * 100, axis=0)
            pivot_str = pivot_table.astype(str)

            percentage_str = percentage_table.round(2).astype(str)
            merged_str = pivot_str + "/" + percentage_str + "%"
            chi2, p, _, _ = chi2_contingency(pivot_table)

            for column in pivot_table.columns[1:]:
                contingency_table = pivot_table[[0, column]]
                chi2_stat_ind, p_value_ind, _, _ = chi2_contingency(contingency_table)
                if p_value_ind < 0.01:
                    merged_str[column][1] = merged_str[column][1] + "**"
                elif 0.01 < p_value_ind < 0.05:
                    merged_str[column][1] = merged_str[column][1] + "*"
            if p <= 0.001:
                p = "<0.001"
            else:
                p = f"{p.round(3)}"
            # import pdb;pdb.set_trace()
            out_row = merged_str
            out_row["P"] = p
            out_row["P"][0] = "-"
            out_row.index = out_row.apply(lambda row: f"{k}_{row.name}", axis=1)

        elif cate_list[idx] == "Con":
            temp_db = data.groupby(ykey)
            if median == True:
                avg = temp_db[k].median().round(2).astype(str)
                quantile = temp_db[k].quantile([0.25, 0.75]).round(2).astype(str)
                adding = ' (' + quantile.loc[:, 0.25] + ', ' + quantile.loc[:, 0.75] + ')'
            else:
                avg = temp_db[k].mean().round(2).astype(str)
                adding = "±" + temp_db[k].sem().round(2).astype(str)

            cell = avg + adding
            group_data = data.groupby(ykey[0])[k]
            t_test_results = {}
            auto_correct = False
            try:
                control_group = group_data.get_group(0)
            except:
                print(f"Key : {ykey[0]} seems has a string like category")
                reference_negatives = ['neg', 'negative', '0', 'low', 'quit smoker', 'no', 'not', 'level_0', 'None',
                                       'lower']
                from Levenshtein import distance
                threshold = 0.5
                pred_key = None
                keys = data[ykey[0]].unique()
                key_dict = {}
                for key in keys:
                    min_distance = np.inf
                    for ref_key in reference_negatives:
                        dist = distance(key, ref_key)
                        if dist < min_distance:
                            min_distance = dist
                            pred_key = key
                    similarity = 1 - min_distance / max(len(key), len(ref_key))
                    key_dict[key] = similarity

                pred_key = max(key_dict, key=key_dict.get)
                pred_value = key_dict[pred_key]

                if pred_value >= threshold:
                    control_group = group_data.get_group(pred_key)
                else:
                    pred_key = input(
                        "We cannot find a control group key automatically, maybe give me one? (Type here): ")
                    control_group = group_data.get_group(pred_key)
                auto_correct = True
            for group_name, group_values in group_data:
                if auto_correct:
                    control_key = pred_key
                else:
                    control_key = 0

                if group_name != control_key:
                    t_stat, p_value = ttest_ind(control_group, group_values)
                    t_test_results[group_name] = {'t_stat': t_stat, 'p_value': p_value}
            if len(data[ykey[0]].unique()) == 2:
                if auto_correct:
                    tempkey = keys.tolist()
                    tempkey.remove(control_key)
                    pos_key = tempkey[0]
                else:
                    pos_key = 1
                final_p = t_test_results[pos_key]['p_value']
                if final_p < 0.01:
                    cell.loc[pos_key] += "**"
                elif 0.01 < final_p < 0.05:
                    cell.loc[pos_key] += "*"

                if final_p < 0.001:
                    final_p = "<0.001"
                elif 0.001 < final_p:
                    final_p = "{:.3f}".format(final_p)
                else:
                    final_p = final_p.round(3)
                cell.loc['P'] = final_p
                out_row = pd.DataFrame(cell).T
            else:
                for idx, result in t_test_results.items():
                    temp_p = result['p_value']
                    if temp_p < 0.01:
                        cell.loc[idx] += "**"
                    elif 0.01 < temp_p < 0.05:
                        cell.loc[idx] += "*"
                group_list = [list(group) for name, group in group_data]
                levene_statistic, levene_pvalue = levene(*group_list)
                if levene_pvalue > 0.05:
                    anova_statistic, p_value = f_oneway(*group_list)
                else:
                    try:
                        import pingouin as pg
                        result = pg.welch_anova(dv="age", between=ykey, data=data)
                        p_value = result['p-unc'][0]
                    except ValueError:
                        print("Cannot perform the Welch ANOVA test!")
                        print("Recalculate the p_value of ANOVA by Kruskal test. . .")
                        anova_statistic, p_value = kruskal(*group_list)

                if p_value < 0.001:
                    p_value = "<0.001"
                elif 0.001 <= p_value < 0.05:
                    p_value = f"{p_value:.3f}"
                else:
                    p_value = p_value.round(3)
                cell.loc['P'] = p_value
                out_row = pd.DataFrame(cell).T
        else:
            raise KeyError("Error! Got the incoming key: {} which is not supported!".format(cate_list[idx]))
        if output_df is None:
            output_df = out_row
        else:
            print("Concate {} info".format(k))
            output_df = pd.concat([output_df, out_row])
    numeric_columns = sorted([col for col in output_df.columns if isinstance(col, (int, float, complex))])
    if len(numeric_columns) == 0:
        sorted_columns = [control_key]
        rest_key = [x for x in output_df.columns if x not in [control_key, 'P']]
        rest_key = sorted(rest_key)
        sorted_columns = sorted_columns + rest_key + ['P']
    else:
        sorted_columns = numeric_columns + ['P']
    output_df = output_df[sorted_columns]
    if save_path:
        try:
            output_df.to_excel(save_path, encoding='utf-8')
        except:
            raise DataFrameGenerationFail("Filed to generate the DataFrame to path: {}".format(save_path))
    else:
        print(
            "The saving path is not given, automatically save the file to the './General_table/{}_general_({})_table.xlsx'".format(
                ykey[0], "AVG" if not median else "MED"))
        check_path("./General_table")
        output_df.to_excel('./General_table/{}_general_({})_table.xlsx'.format(ykey[0], "AVG" if not median else "MED"))
    return output_df

    if save_path:
        try:
            output_df.to_excel(save_path,encoding='utf-8')
        except:
            raise DataFrameGenerationFail("Filed to generate the DataFrame to path: {}".format(save_path))
    else:
        print("The saving path is not given, automatically save the file to the './General_table/{}_general_table.xlsx'".format(ykey[0]))
        check_path("./General_table")
        output_df.to_excel('./General_table/{}_general_table.xlsx'.format(ykey[0]))
    return output_df

if __name__ == '__main__':
    data = pd.read_csv(r"D:\Pycharm_workplace\New_test\new_db.csv",engine='python')
    xkeys = ['age','sex','Packyears','Smoker','Emphysema_score','Max_Nodule_size',"Nodule_above_100"]
    ykey = "CAC_Above_100"
    cate_list = ['Con','Cat','Con','Cat','Con','Con','Cat']
    S_table = generate_S_table(data=data,xkeys=xkeys,ykey=ykey,cate_list=cate_list,save_path=r"D:\Pycharm_workplace\New_test\general_table.xlsx".format(ykey[0]))

