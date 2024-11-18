#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : myplot.py
@Time         : 2024/02/14 13:50:39
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the self-designed matplotlib plotting functions that maight be useful for data analysis 

'''
import os 
import os.path as osp 
import matplotlib as mpl
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline

from CBB.cfg import *
from CBB.cfg import _reset_figure

from CBB.data_type import is_cate, round_down_to_nearest_half, round_up_to_nearest_half, variable_type
from CBB.myMetrics import Cal_CI, Cal_IQR, KsNormDetect
from CBB.mydb import if_Nan_Exists, remove_nan
from CBB.myos import auto_save_file, check_path, get_file_name
from CBB.errors import *
from pprint import pprint

def generate_panel(num=1,rows=1,cols=1,W=4,H=3,figure_size=(12,8),DPI=600):
    """This function will generate the subplot panel for the model evaluation

    Args:
        num (int, optional): The overall number of subplots. Deaults to 1.
        rows (int, optional): The number of rows . Defaults to 1.
        cols (int, optional): The number of cols. Defaults to 1.
        W (int, optinal): The width unit of the panel. Defaults to 4.
        H (int, optinal): The height unit of the panel. Defaults to 3
        figure_size (tuple, optional): The size of the overall panel. Defaults to (12,8).
        DPI (int, optional): The dpi of the panel. Defaults to 600.
    """     

    # The input value can be 0 or below, reassign the val
    r = int(np.ceil(np.sqrt(num)))
    c = int(np.ceil(num / r))
    rows = max(rows,r)
    cols = max(cols,c)

    e_w,e_h = max(cols*W,figure_size[0]),max(r*H,figure_size[1])
    fig, axe = plt.subplots(rows, cols, figsize=(e_w, e_h))
    return fig, axe, rows, cols


def forest_plot(data,xkeys,ykey,cate_keys=None,order=None, save_path=None,verbose=False):
    """This function will generate the forest plot based on the logistic regression result of the model

    Args:
        data (pd.DataFrame): The overall database 
        xkeys (list): The variables that will be treated as indendent variables
        ykey (str/list): The variables that will be treaded as a dependent variable 
        cate_keys (list, optional): The categorical variables from the xkeys. Defaults to None.
        order (list, optional): The show order of the forest plot. Defaults to None.
        save_path (str, optional): The saving path of the plot if given. Defaults to None.
        verbose (bool, optional): Show the details of intermediate states if True otherwise show the final result only. Defaults to False.

    Raises:
        KeyError: ykey is included in the cate_key
        ValueError: cate_key contain values that are not included by xkeys
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    import myforestplot as mfp
    assert isinstance(data,pd.DataFrame),"Error!The given data is not in the type of DataFrame!"
    
    def add_stars_based_on_pvalue(row):
        pvalue = float(row['pvalues'])

        if pvalue < 1e-2:
            return f"{row['risk_pretty'].replace(' (', '** (')}"
        elif 1e-2 <= pvalue < 0.05:
            return f"{row['risk_pretty'].replace(' (', '* (')}"
        else:
            return row['risk_pretty']
    
    if isinstance(ykey, list) and len(ykey) != 1:
        raise KeyError(f"Error! The given ykey: {ykey} contains more than one element which is not allowed!")
    elif isinstance(ykey, str):
        ykey = [ykey]
    
    if cate_keys is not None:
        assert not ykey[0] in cate_keys,"Error! The cate_keys should not contain ykey"
    else:
        cate_keys = []
        for col in xkeys:
            if variable_type(data[col]) == "Cat":
                cate_keys.append(col)
            else:
                continue
        print("The cate_key is not given, auto-pred the cate-keys as below: \n{}".format(cate_keys))
    if len(set(cate_keys).difference(set(xkeys))) != 0:
        raise ValueError("The given cate_keys contains other keys {} that is not included by xkeys!".format(set(cate_keys).difference(set(xkeys))))

    data = data[xkeys+ykey].dropna()
    data['Intercept'] = 1

    if verbose:
        print("Independent variables:",end=" ")
        print(*xkeys, sep=" ") 
        print(f"dependent variable: {ykey[0]}")
        print("Categorical variables:",end=" ")
        print(*cate_keys, sep=" ")

    rule = f"{ykey[0]} ~ " + " + ".join(xkeys)

    res = smf.logit(formula=rule,data=data).fit()

    p_values = res.pvalues
    formatted_p_values = ["{:.2e}".format(p) for p in p_values]
    res.pvalues = pd.Series(formatted_p_values, index=res.pvalues.index)

    order = order if order is not None else xkeys
    cont_cols = set(xkeys).difference(set(cate_keys))

    df_sum = mfp.statsmodels_pretty_result_dataframe(
        data,
        res,
        order=order,
        cont_cols=cont_cols,
        fml=".3f"
    )
    if verbose:
        print(df_sum)

    df = df_sum.copy()
    df["nobs"] = (df["nobs"]
              .replace(np.nan, data.shape[0])
              .astype(int)
              )

    plt.rcParams["font.size"] = 8
    fp = mfp.ForestPlot(df=df,
                        ratio=[5,5,3],
                        fig_ax_index=[2],
                        dpi=600,
                        figsize=(12,6),
                        yticks_show=False,
                        vertical_align=True)
    
    fp.errorbar(index=2, errorbar_color="red",errorbar_kwds={})

    low,high = fp.df[0].min(),fp.df[1].max()
    low,high = min(0,round_down_to_nearest_half(low)),max(1,round_up_to_nearest_half(high))
    fp.axd[2].set_xlim([low, high])
    fp.axd[2].set_xticks(np.arange(low,high+0.5,0.5))
    fp.axd[2].set_xlabel("OR")
    fp.axd[2].axvline(x=1, ymin=0, ymax=1.0, color="black", alpha=0.5)

    fp.df["risk_pretty"] = fp.df.apply(add_stars_based_on_pvalue,axis=1)

    fp.axd[1].set_xlim([0.50, 1.0])
    fp.embed_cate_strings(1, "category", 0.5, header="Category",
                    text_kwds=dict(fontweight="bold"),
                    header_kwds=dict(fontweight="bold"),
                    )
    fp.embed_strings(1, "item", 0.55, header="", replace={"age":""})
    fp.embed_strings(1, "nobs", 0.86, header="N")
    fp.embed_strings(3, "risk_pretty", 0.1, header="OR (95% CI)")
    fp.embed_strings(3, "pvalues", 0.8, header="P-value", replace={"pvalue": ""})
    fp.horizontal_variable_separators()

    y_below = -(fp.df['category'].unique().size + fp.df['category'].size + 6/(fp.df['category'].unique().size + len(xkeys)))
    fp.draw_horizontal_line(y=1+abs(6/(fp.df['category'].unique().size + len(xkeys))),scale=.25)
    fp.draw_horizontal_line(y=y_below,scale=.25)    

    # fp.axd[1].text(-0.1, -0.12, "Line 1", ha='center', va='center', fontsize=5, fontweight='bold', transform=fp.axd[1].transAxes)
    # fp.axd[1].text(-0.1, -0.14, "Line 2", ha='center', va='center', fontsize=5, fontweight='bold', transform=fp.axd[1].transAxes)
    check_path(osp.join(FPLOTS))
    if not save_path:
        save_path = osp.join(FPLOTS,f"{ykey[0]}_forestplot.jpg")
    plt.savefig(save_path)


def pt2pixel(pt):
    return float(DPI/72*pt)

def pixel2pt(pixel):
    return float(pixel*72/DPI)


def show_distribution(data, mode="line", name=None, show_metric=False, save=False, cate=False):
    '''
    Description: This function will draw the distribution of the data automatically based on the
                    shape of the data and the mode type
    :param data: The input data as type: numpy.ndarray
    :param mode: The drawing mode that is selected depending on the data shape
                    if 2-d: "line","points","shade-line"
                    if 1-d: "bar","line","hist
    :return: The ploted diagram
    '''
    import seaborn as sns
    assert data is not None, "Error! The given data is None and can not be analysed"
    _reset_figure()
    # Set some default seaborn parameters 
    palette = plt.get_cmap('tab20c')#'Pastel2')      # 'Set1'

    dim = len(data.shape)

    if dim == 1:    
        if mode == "bar":
            ax1 = sns.histplot(data, color=sns.desaturate("yellow", .8), alpha=1)
        elif mode == "line":
            ax1 = sns.kdeplot(data)
        elif mode == "hist":
            sns.set_palette("hls")
            mpl.rc("figure", figsize=(15, 5))
            ax1 = sns.histplot(data,kde=True, shrink = 1,color = palette.colors[0], edgecolor = palette.colors[-1])
            add_values(ax1)
        elif mode=="pie":
            v_count = data.value_counts()
            labels = ["level_{}".format(int(x)) for x in v_count.index ]
            plt.pie(x = v_count.values,labels=labels ,autopct = '%3.2f%%',colors = ['red','yellowgreen','lightskyblue','yellow'])
            plt.legend()
        else:
            raise KeyError("Error! Currently only support 'line','bar' mode! Function return")
        if name and mode != "pie":
            ax1.set_title("{}_{}_graph".format(name, mode))
        if show_metric:
            data = remove_nan(data)
            avg = np.nanmean(data)
            median = np.nanmedian(data)
            std = np.nanstd(data)
            plt.axvline(avg, color='green', lw=2, alpha=0.7)
            plt.axvline(median, color='red', lw=2, alpha=0.7)
            CI_low,CI_high = Cal_CI(data)
            Norm = KsNormDetect(data)
            IQR_low,IQR_high = Cal_IQR(data)
            print("Mean val: {} / Median val: {} / standard deviation: {}".format(avg,median,std))
            print("CI_Low and CI_High: {} / {}".format(CI_low,CI_high))
            print("IQR_Low and IQR_High: {} / {}".format(IQR_low,IQR_high))


    elif dim == 2:
        if isinstance(data,pd.DataFrame):
            names = data.columns.values
            X,Y = data[names[0]],data[names[1]]
        else:
            X,Y = data[:,0],data[:,1]
        if mode == "shade-line":
            mpl.rc("figure", figsize=(6, 6))
            sns.kdeplot(x=X, y=Y, shade=True, bw="scott", gridsize=50, clip=(-11, 11))
        elif mode == "line":
            mpl.rc("figure", figsize=(6, 6))
            sns.kdeplot(x=X, y=Y)
        elif mode == "dots" or mode == "points":
            plt.scatter(X,Y)
    else:
        print("Sorry the function does not support data more than 2 dims")
        return
    plt.tight_layout()
    if save:
        if name:
            save_path = "./{}_{}_graph.jpg".format(name,mode)
        else:
            save_path = "./{}_graph.jpg".format(mode)
        p = auto_save_file(save_path)
        plt.savefig(p)
        plt.figure()
    else:
        plt.show()

def show_dataframe_distribution(data, key, mode="bar",save=False,bars=None):
    """ This function is the alternative generation of 'show_distribution', which will take dataframe 
        as input not Series or numpy.array 

    Args:
        data (pd.DataFrame): The overall dataframe of the entire database
        key (string): The desired key that want to the illustared
        mode (str, optional): The type of the output graph. Defaults to "bar".
        save (bool, optional): If saving the graph. Defaults to False.
    """
    import seaborn as sns
    _reset_figure(800)
    if not isinstance(data,pd.DataFrame):
        raise TypeError("Error! The function only allow Dataframe as input but not {}".format(type(data)))
    if mode =="bar":
        sns.set(style="darkgrid")
        data = pd.DataFrame(data[key].dropna())
        ax1 = sns.countplot(x=key,data=data)
        plt.xticks(np.arange(4),['Level_0','Level_1','Level_2','Level_3'])
        for p in ax1.patches:
            ax1.annotate(
                text=f"{p.get_height():1.0f}",
                xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
                xycoords='data',
                ha='center', 
                va='center', 
                fontsize='medium', 
                color='black',
                xytext=(0,7), 
                textcoords='offset points',
                clip_on=True,
            )
    if mode == 'box':
        sns.set(style="darkgrid")
        data = pd.DataFrame(data[key].dropna())
        ax1 = sns.boxplot(y=data[key],log_scale=False)
    if mode == "hist":
        sns.set(style="darkgrid")
        data = pd.DataFrame(data[key].dropna())
        bins = np.histogram_bin_edges(data[key], bins='auto')
        ax1 = sns.histplot(data, kde=True, shrink = 1, bins=bars if bars else 30)
        add_values(ax1)
    if save:
        name = r"./{}_{}_graph.jpg".format(key,mode)
        p = auto_save_file(name)
        plt.savefig(p)

def plot_3D_points(data,keys,save_path=None):
    from mpl_toolkits.mplot3d import Axes3D
    assert isinstance(data, pd.DataFrame),"Error! The incoming data is not pd.DataFrame type "
    assert len(keys) == 3 ,"Error! The key list length is not 3!"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data[keys[0]]
    y = data[keys[1]]
    z = data[keys[2]]

    ax.scatter(x, y, z, cmap='viridis', marker='o',s=5)
    ax.set_xlabel(keys[0])
    ax.set_ylabel(keys[1])
    ax.set_zlabel(keys[2])
    ax.set_title('Three-Dimensional Scatter Plot')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_3D_Adv(data,keys,point_size=8,save_path=None):
    import seaborn as sns
    fig = plt.figure(figsize=(12, 8))
    plt.rcParams['axes.titlesize'] = 10

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(data[keys[0]], data[keys[1]], data[keys[2]],alpha=0.8, marker='o',s=point_size//2)
    ax1.set_xlabel(keys[0])
    ax1.set_ylabel(keys[1])
    ax1.set_zlabel(keys[2])
    ax1.set_title('Three-Dimensional Scatter Plot')

    ax2 = fig.add_subplot(2, 2, 2)
    sns.scatterplot(x=data[keys[0]], y=data[keys[1]], data=data,alpha=0.8, ax=ax2,s=point_size)
    sns.regplot(x=data[keys[0]], y=data[keys[1]], data=data, ax=ax2, scatter=False, color='red')
    ax2.set_xlabel(keys[0])
    ax2.set_ylabel(keys[1])
    ax2.set_title(f'2D Projection with Regression Line ({keys[0]} vs {keys[1]})')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax3 = fig.add_subplot(2, 2, 3)
    sns.scatterplot(x=data[keys[0]], y=data[keys[2]], data=data,alpha=0.8, ax=ax3,s=point_size)
    sns.regplot(x=data[keys[0]], y=data[keys[2]], data=data, ax=ax3, scatter=False, color='red')
    ax3.set_xlabel(keys[0])
    ax3.set_ylabel(keys[2])
    ax3.set_title(f'2D Projection with Regression Line ({keys[0]} vs {keys[2]})')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = fig.add_subplot(2, 2, 4)
    sns.scatterplot(x=data[keys[1]], y=data[keys[2]], data=data,alpha=0.8, ax=ax4,s=point_size)
    sns.regplot(x=data[keys[1]], y=data[keys[2]], data=data, ax=ax4, scatter=False, color='red')
    ax4.set_xlabel(keys[1])
    ax4.set_ylabel(keys[2])
    ax4.set_title(f'2D Projection with Regression Line ({keys[1]} vs {keys[2]})')

    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def show_distribution_cate(data, mode="bar"):
    if isinstance(data, pd.Series):
        unique_data = data.unique()
        data = data.to_list()
    elif isinstance(data,np.ndarray):
        if not len(data.shape) == 1:
            print("Error! The input numpy ndarray is not flattened!")
            return 0
        unique_data = np.unique(data)
    else:
        raise UnSupportTypeError("The incoming type {} is not supported in the current function".format(type(data)))

    name_dict = {name : 0 for name in unique_data}
    for name in name_dict:
        if name != name:
            name_dict['nan'] = name_dict.pop(name)
            break
        else:
            continue

    for val in data:
        if val != val:
            val = "nan"
        if not val in name_dict:
            raise KeyError("Val {} is not listed in the dictionary".format(val))
        else:
            name_dict[val] += 1
    pprint(name_dict)

    if mode == "bar":
        X = np.arange(len(name_dict.keys()))
        Y = list(name_dict.values())
        if len(Y) * FONT_SIZE > DPI:
            print("Graph reshape!")
            fig = plt.figure(figsize=((len(Y)+2)//3, len(Y)//3), dpi=DPI//2)
        plt.barh(X,Y,height=BAR_WIDTH)
        plt.ylabel('Keys')
        plt.xlabel('Values')
        plt.yticks(X,list(name_dict.keys()))
        for a,b,i in zip(name_dict.keys(), name_dict.values(), range(len(name_dict.keys()))):
            plt.text(b+DPI,list(name_dict.keys()).index(a)-BAR_WIDTH/4,"%.2f" % list(name_dict.values())[i], ha="left",fontsize=FONT_SIZE)
        # for a, b, i in zip(name_dict.keys(), name_dict.values(), range(len(name_dict.keys()))):
        #     plt.text(a, b + 0.01, "%.2f" % name_dict.values()[i], ha='center', fontsize=20)
        plt.show()

def outliers_proc(data, col_name, scale=3):
    import seaborn as sns
    """
    用于清洗异常值, 默认box_plot(scale=3)进行清洗
    param data: 接收pandas数据格式
    param col_name: pandas列名
    param scale: 尺度
    """

    def box_plot_outliers(data_ser, box_scale):
        """
        利用箱线图去除异常值
        :param data_ser: 接收 pandas.Series 数据格式
        :param box_scale: 箱线图尺度
        """
        iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
        val_low = data_ser.quantile(0.25) - iqr
        val_up = data_ser.quantile(0.75) + iqr
        rule_low = (data_ser < val_low)
        rule_up = (data_ser > val_up)
        return (rule_low, rule_up), (val_low, val_up)

    data_n = data.copy()
    data_serier = data_n[col_name]
    rule, value = box_plot_outliers(data_serier, box_scale=scale)
    index = np.arange(data_serier.shape[0])[rule[0] | rule[1]]
    print("Delete number is:{}".format(len(index)))
    data_n = data_n.drop(index)
    data_n.reset_index(drop=True, inplace=True)
    print("Now column number is:{}".format(data_n.shape[0]))
    index_low = np.arange(data_serier.shape[0])[rule[0]]
    outliers = data_serier.iloc[index_low]
    print("Description of data less than the lower bound is:")
    print(pd.Series(outliers).describe())
    index_up = np.arange(data_serier.shape[0])[rule[1]]
    outliers = data_serier.iloc[index_up]
    print("Description of data larger than the upper bound is:")
    print(pd.Series(outliers).describe())

    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
    p = r"./{}_box_graph.jpg".format(col_name)
    p = auto_save_file(p)
    plt.savefig(p)
    return data_n

def show_correlation_dataframe(data,xkey,ykey, mode="point",log=False, save=False):
    import seaborn as sns
    import matplotlib.pyplot as plt
    assert isinstance(data,pd.DataFrame),"Error! The loaded data must be in type pd.DataFrame"
    if if_Nan_Exists(data[[xkey,ykey]]):
        data = data[[xkey,ykey]].dropna()

    if mode == "point" or mode == "points":
        ax1 = sns.scatterplot(data,x=data[xkey],y=data[ykey])
    elif mode.lower() == "box":
        if not is_cate(data[xkey]):
            print("The given X-data does not seems like a categorical data! Function returned!")
            return None
        else:
            if log:
                yticks_values = [10, 100, 1000, 10000,100000]
                plt.yscale('log')
                plt.yticks(yticks_values, [str(val) for val in yticks_values])
            ax1 = sns.boxplot(x = data[xkey], y = data[ykey], palette="Blues")
    else:
        raise KeyError("The mode {} is currently not supported. Function return".format(mode))
    if save:
        check_path("./plots/")
        path = "./plots/{}_{}_correlation_{}_graph.jpg".format(xkey,ykey,mode)
        path = auto_save_file(path)
        plt.savefig(path)
        plt.figure()


def add_values(chart,fontsize=None):

    for p in chart.patches:
        chart.annotate(
            text=f"{p.get_height():1.0f}",
            xy=(p.get_x() + p.get_width() / 2., p.get_height()), 
            xycoords='data',
            ha='center', 
            va='center', 
            fontsize= 35 / np.sqrt(len(chart.patches)) if not fontsize else fontsize, 
            color='black',
            xytext=(0,7), 
            textcoords='offset points',
            clip_on=True,
        )

def draw_boxplot(data, xkeys, ykey, jitter=False, sample_fraction=0.1, name=None, legend=None, save_path=None, palette=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))

    if palette is None:
        ax = sns.boxplot(x=xkeys, y=ykey, data=data, palette='tab10', width=0.5)
    else:
        ax = sns.boxplot(x=xkeys, y=ykey, data=data, palette=palette,width=0.5)
    if jitter:
        # 在箱型图上添加散点图
        temp_table_long = data.sample(frac=sample_fraction, random_state=1)
        sns.stripplot(x=xkeys, y=ykey, data=temp_table_long, jitter=True,
                      color='black', alpha=0.5)

    if name is None:
        plt.title(f'Boxplot of {xkeys} / {ykey}', fontsize=16)
    else:
        plt.title(f'Boxplot of {name}', fontsize=16)

    if legend is not None:
        # 手动创建图例
        handles, labels = ax.get_legend_handles_labels()
        unique_xkeys = list(legend.values())
        handles = [plt.Line2D([0], [0], color=ax.patches[i].get_facecolor(), lw=4) for i in range(len(unique_xkeys))]
        plt.legend(handles, unique_xkeys, title=xkeys, loc='upper left', bbox_to_anchor=(1, 1))

    plt.xlabel(xkeys)
    plt.ylabel(ykey)
    if legend is not None:
        plt.xticks(range(len(legend)), list(legend.values()))

    plt.tight_layout()
    if not save_path:
        save_dir = check_path("./feature_plots")
        if name is None:
            save_path = osp.join(save_dir, f"{xkeys}_{ykey}_box_plot.jpg")
        else:
            save_path = osp.join(save_dir, f"{name}_box_plot.jpg")
    plt.savefig(save_path)
    plt.close()


def df2heatmap(df, title=None, fmt=".4f", figsize=(10, 8), annot=True, cmap="YlGnBu", save_path=None, **kwargs):
    plt.figure(figsize=figsize)
    xlabel = kwargs.get("xlabel", "X")
    ylabel = kwargs.get("ylabel", "Y")
    highlight_min = kwargs.get("highlight_min", False)
    highlight_max = kwargs.get("highlight_max", False)

    if isinstance(df, str):
        if title is None:
            title = osp.splitext(osp.basename(df))[0]
        if save_path is None:
            save_dir = osp.dirname(df)
            save_path = osp.join(save_dir, f"{title}.jpg")
        df = pd.read_csv(df, engine='python')

    try:
        heatmap = sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap)
    except ValueError as e:
        if "could not convert string to float" in str(e):
            # 如果异常为无法将字符串转换为浮点数，则将第一列设置为索引并重新绘制热力图
            df.set_index(df.columns[0], inplace=True)
            heatmap = sns.heatmap(df, annot=annot, fmt=fmt, cmap=cmap)
        else:
            # 如果异常不是无法将字符串转换为浮点数，则抛出异常
            raise e

    # 高亮最小值和最大值
    if highlight_min or highlight_max:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                value = df.iloc[i, j]
                if highlight_min and value == df.min().min():
                    heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
                if highlight_max and value == df.max().max():
                    heatmap.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='blue', lw=3))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if save_path is None:
        save_dir = "./heatmaps"
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        save_path = osp.join(save_dir, f"heatmap_{title}.jpg")

    plt.savefig(save_path)

def plot_line_graph(data, xkey, ykey, smoothness=0, axe=None):
    """
    Plot a smooth line chart from a dataframe based on the specified x and y keys.

    Parameters:
    - data (DataFrame): The input dataframe containing the data to plot.
    - xkey (str): The column key in the dataframe to plot on the x-axis.
    - ykey (str): The column key in the dataframe to plot on the y-axis.
    - smoothness (float, optional): The smoothing factor used in UnivariateSpline (default is 1.0).
    - axe (AxesSubplot or None, optional): The axes object to plot onto. If None, create a new figure.

    Returns:
    - axe (AxesSubplot): The axes object containing the plot.
    """
    if axe is None:
        fig, axe = plt.subplots()

    x = data[xkey]
    y = data[ykey]

    if smoothness == 0:
        # Plot a regular line plot
        sns.lineplot(x=x, y=y, ax=axe, marker='o', label='Original Data')
    else:
        # Create a univariate spline with the specified smoothness
        spline = UnivariateSpline(x, y, s=smoothness)
        x_smooth = np.linspace(x.min(), x.max(), 1000)
        y_smooth = spline(x_smooth)
        axe.plot(x_smooth, y_smooth, label='Smooth Line', marker='o')
        axe.plot(x, y, 'o', label='Original Data')

    # Customize labels, ticks, and title as needed
    axe.set_xlabel(xkey)
    axe.set_ylabel(ykey)
    axe.set_title('Smooth Line Plot')

    return axe

