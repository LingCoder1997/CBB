#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File Name    : cfg.py
@Time         : 2024/02/14 13:56:00
@Author       : L WANG
@Contact      : wang@i-dna.org
@Version      : 0.0.0
@Description  : This file contains the configuration info of the packages (Some of the variables need to be fixed depending on the )
'''

import matplotlib.pyplot as plt

PLOT_PATH="./plots"
METRICS_PATH = "./ML_result"
FPLOTS = "./forest_plot"
BAR_WIDTH=0.5
FONT_SIZE=10
DPI=600

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

def _reset_figure(dpi=600):
    fig = plt.figure(figsize=(10,8),dpi=dpi)
    return fig