# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:02:45 2018

@author: jinyx
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

feature_list = ['_mean', '_median', '_std', '_min', '_max', '_range',
                '_minRatio', '_maxRatio', '_1Diff_mean', '_1Diff_median',
                '_1Diff_std', '_1Diff_min', '_1Diff_max', '_1Diff_range',
                '_1Diff_minRatio', '_1Diff_maxRatio', '_2Diff_std',
                '_2Diff_min', '_2Diff_max', '_2Diff_range', '_2Diff_minRatio',
                '_2Diff_maxRatio', '_fft_mean', '_fft_median', '_fft_std',
                '_fft_min', '_fft_max', '_fft_range']


def rsp_mean_(df):
    return df.mean(axis=1)


def rsp_median_(df):
    return df.median(axis=1)


def rsp_std_(df):
    return df.std(axis=1)


def rsp_min_(df):
    return df.min(axis=1)


def rsp_max_(df):
    return df.max(axis=1)


def rsp_range_(df_max, df_min):
    return df_max['_max'] - df_min['_min']


# 最小值比率 = Mmin/N
def rsp_minRatio_(all_df, rsp_min):
    all_df_T = all_df.T
    rsp_min_T = rsp_min.T
    rsp_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len(all_df_T[i][all_df_T[i] == rsp_min_T.at['_min', i]])
        rsp_minRatio_dict.update({i: num_min / 8064.0})
    rsp_minRatio_df = pd.DataFrame.from_dict(data=rsp_minRatio_dict, orient='index')
    rsp_minRatio_df.columns = ['_minRatio']
    return rsp_minRatio_df


# 最大值比率 = Nmax/N
def rsp_maxRatio_(all_df, rsp_max):
    all_df_T = all_df.T
    rsp_max_T = rsp_max.T
    rsp_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len(all_df_T[i][all_df_T[i] == rsp_max_T.at['_max', i]])
        rsp_maxRatio_dict.update({i: num_max / 8064.0})
    rsp_maxRatio_df = pd.DataFrame.from_dict(data=rsp_maxRatio_dict, orient='index')
    rsp_maxRatio_df.columns = ['_maxRatio']
    return rsp_maxRatio_df


# RSP一阶差分均值
def rsp1Diff_mean_(all_df):
    rsp1Diff_mean = all_df.diff(periods=1, axis=1).dropna(axis=1).mean(axis=1)
    return rsp1Diff_mean


# RSP一阶差分中值
def rsp1Diff_median_(all_df):
    rsp1Diff_median = all_df.diff(periods=1, axis=1).dropna(axis=1).median(axis=1)
    return rsp1Diff_median


# RSP一阶差分标准差
def rsp1Diff_std_(all_df):
    rsp1Diff_std = all_df.diff(periods=1, axis=1).dropna(axis=1).std(axis=1)
    return rsp1Diff_std


def rsp1Diff_min_(all_df):
    rsp1Diff_min = all_df.diff(periods=1, axis=1).dropna(axis=1).min(axis=1)
    return rsp1Diff_min


def rsp1Diff_max_(all_df):
    rsp1Diff_max = all_df.diff(periods=1, axis=1).dropna(axis=1).max(axis=1)
    return rsp1Diff_max


def rsp1Diff_range_(rsp1Diff_max, rsp1Diff_min):
    return rsp1Diff_max['_1Diff_max'] - rsp1Diff_min['_1Diff_min']


def rsp1Diff_minRatio_(all_df, rsp1Diff_min):
    all_df_Diff_T = all_df.diff(periods=1, axis=1).dropna(axis=1).T
    rsp1Diff_min_T = rsp1Diff_min.T
    rsp1Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len(all_df_Diff_T[i][all_df_Diff_T[i] == rsp1Diff_min_T.at['_1Diff_min', i]])
        rsp1Diff_minRatio_dict.update({i: num_min / 8063.0})
    rsp1Diff_minRatio_df = pd.DataFrame.from_dict(data=rsp1Diff_minRatio_dict, orient='index')
    return rsp1Diff_minRatio_df


def rsp1Diff_maxRatio_(all_df, rsp1Diff_max):
    all_df_Diff_T = all_df.diff(periods=1, axis=1).dropna(axis=1).T
    rsp1Diff_max_T = rsp1Diff_max.T
    rsp1Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len(all_df_Diff_T[i][all_df_Diff_T[i] == rsp1Diff_max_T.at['_1Diff_max', i]])
        rsp1Diff_maxRatio_dict.update({i: num_max / 8063.0})
    rsp1Diff_maxRatio_df = pd.DataFrame.from_dict(data=rsp1Diff_maxRatio_dict, orient='index')
    return rsp1Diff_maxRatio_df


def rsp2Diff_std_(all_df):
    rsp2Diff_std = all_df.diff(periods=2, axis=1).dropna(axis=1).std(axis=1)
    return rsp2Diff_std


def rsp2Diff_min_(all_df):
    rsp2Diff_min = all_df.diff(periods=2, axis=1).dropna(axis=1).min(axis=1)
    return rsp2Diff_min


def rsp2Diff_max_(all_df):
    rsp2Diff_max = all_df.diff(periods=2, axis=1).dropna(axis=1).max(axis=1)
    return rsp2Diff_max


def rsp2Diff_range_(rsp2Diff_max, rsp2Diff_min):
    rsp2Diff_range = rsp2Diff_max['_2Diff_max'] - rsp2Diff_min['_2Diff_min']
    return rsp2Diff_range


def rsp2Diff_minRatio_(all_df, rsp2Diff_min):
    all_df_2Diff_T = all_df.diff(periods=2, axis=1).dropna(axis=1).T
    rsp2Diff_min_T = rsp2Diff_min.T
    rsp2Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len(all_df_2Diff_T[i][all_df_2Diff_T[i] == rsp2Diff_min_T.at['_2Diff_min', i]])
        rsp2Diff_minRatio_dict.update({i: num_min / 8062.0})
    rsp2Diff_minRatio_df = pd.DataFrame.from_dict(data=rsp2Diff_minRatio_dict, orient='index')
    return rsp2Diff_minRatio_df


def rsp2Diff_maxRatio_(all_df, rsp2Diff_max):
    all_df_2Diff_T = all_df.diff(periods=2, axis=1).dropna(axis=1).T
    rsp2Diff_max_T = rsp2Diff_max.T
    rsp2Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len(all_df_2Diff_T[i][all_df_2Diff_T[i] == rsp2Diff_max_T.at['_2Diff_max', i]])
        rsp2Diff_maxRatio_dict.update({i: num_max / 8062.0})
    rsp2Diff_maxRatio_df = pd.DataFrame.from_dict(data=rsp2Diff_maxRatio_dict, orient='index')
    return rsp2Diff_maxRatio_df


# RSP DFT(FFT)频域数据
def rspfft_(all_df):
    rspfft_df = pd.DataFrame()
    for i in all_df.index.tolist():
        temp_rspfft = pd.DataFrame(np.fft.fft(all_df.loc[i, :].values)).T
        temp_rspfft.index = [i]
        rspfft_df = rspfft_df.append(temp_rspfft)
    return rspfft_df


# RSP 频域中值
def rspfft_mean_(rspfft_df):
    rspfft_mean = rspfft_df.mean(axis=1)
    return rspfft_mean


def rspfft_median_(rspfft_df):
    rspfft_median = rspfft_df.median(axis=1)
    return rspfft_median


def rspfft_std_(rspfft_df):
    rspfft_std = rspfft_df.std(axis=1)
    return rspfft_std


def rspfft_min_(rspfft_df):
    rspfft_min = rspfft_df.min(axis=1)
    return rspfft_min


def rspfft_max_(rspfft_df):
    rspfft_max = rspfft_df.max(axis=1)
    return rspfft_max


def rspfft_range_(rspfft_max, rspfft_min):
    rspfft_range = rspfft_max['_fft_max'] - rspfft_min['_fft_min']
    return rspfft_range


def get_123count_(df):
    tmp_df = pd.DataFrame()
    for i in range(0, 40, 1):
        num_1 = len(df[i][df[i] == 1])
        num_2 = len(df[i][df[i] == 2])
        num_3 = len(df[i][df[i] == 3])
        list_num = [num_1, num_2, num_3]
        tmp_df = pd.concat([tmp_df, pd.DataFrame(list_num)], axis=1)
    tmp_df.columns = range(0, 40, 1)
    tmp_df.index = ['num_1', 'num_2', 'num_3']
    return tmp_df


def extract_features(features, df_data_x, ch_id=""):
    if len(features) == 0 or len(
            {'mean', 'median', 'std', 'min', 'max', 'range', 'minRatio', 'maxRatio'} & set(features)) > 0:
        _mean = pd.DataFrame(rsp_mean_(df_data_x), columns=['_mean'])
        _median = pd.DataFrame(rsp_median_(df_data_x), columns=['_median'])
        _std = pd.DataFrame(rsp_std_(df_data_x), columns=['_std'])
        _min = pd.DataFrame(rsp_min_(df_data_x), columns=['_min'])
        _max = pd.DataFrame(rsp_max_(df_data_x), columns=['_max'])
        _range = pd.DataFrame(rsp_range_(_max, _min), columns=['_range'])
        _minRatio = pd.DataFrame(rsp_minRatio_(df_data_x, _min), columns=['_minRatio'])
        _maxRatio = pd.DataFrame(rsp_maxRatio_(df_data_x, _max), columns=['_maxRatio'])

    if len(features) == 0 or len(
            set(features) & {'1Diff_mean', '1Diff_median', '1Diff_std', '1Diff_min', '1Diff_max', '1Diff_range',
                             '1Diff_minRatio', '1Diff_maxRatio'}) > 0:
        _1Diff_mean = pd.DataFrame(rsp1Diff_mean_(df_data_x), columns=['_1Diff_mean'])
        _1Diff_median = pd.DataFrame(rsp1Diff_median_(df_data_x), columns=['_1Diff_median'])
        _1Diff_std = pd.DataFrame(rsp1Diff_std_(df_data_x), columns=['_1Diff_std'])
        _1Diff_min = pd.DataFrame(rsp1Diff_min_(df_data_x), columns=['_1Diff_min'])
        _1Diff_max = pd.DataFrame(rsp1Diff_max_(df_data_x), columns=['_1Diff_max'])
        _1Diff_range = pd.DataFrame(rsp1Diff_range_(_1Diff_max, _1Diff_min), columns=['_1Diff_range'])
        _1Diff_minRatio = rsp1Diff_minRatio_(df_data_x, _1Diff_min)
        _1Diff_minRatio.columns = ['_1Diff_minRatio']
        _1Diff_maxRatio = rsp1Diff_maxRatio_(df_data_x, _1Diff_max)
        _1Diff_maxRatio.columns = ['_1Diff_maxRatio']

    if len(features) == 0 or len(
            set(features) & {'2Diff_std', '2Diff_min', '2Diff_max', '2Diff_range', '2Diff_minRatio',
                             '2Diff_maxRatio'}) > 0:
        _2Diff_std = pd.DataFrame(rsp2Diff_std_(df_data_x), columns=['_2Diff_std'])
        _2Diff_min = pd.DataFrame(rsp2Diff_min_(df_data_x), columns=['_2Diff_min'])
        _2Diff_max = pd.DataFrame(rsp2Diff_max_(df_data_x), columns=['_2Diff_max'])
        _2Diff_range = pd.DataFrame(rsp2Diff_range_(_2Diff_max, _2Diff_min), columns=['_2Diff_range'])
        _2Diff_minRatio = rsp2Diff_minRatio_(df_data_x, _2Diff_min)
        _2Diff_minRatio.columns = ['_2Diff_minRatio']
        _2Diff_maxRatio = rsp2Diff_maxRatio_(df_data_x, _2Diff_max)
        _2Diff_maxRatio.columns = ['_2Diff_maxRatio']

    if len(features) == 0 or len(
            set(features) & {'fft_mean', 'fft_median', 'fft_std', 'fft_min', 'fft_max', 'fft_range'}) > 0:
        _fft_df = rspfft_(df_data_x)
        _fft_mean = pd.DataFrame(rspfft_mean_(_fft_df), columns=['_fft_mean'])
        _fft_median = pd.DataFrame(rspfft_median_(_fft_df), columns=['_fft_median'])
        _fft_std = pd.DataFrame(rspfft_std_(_fft_df), columns=['_fft_std'])
        _fft_min = pd.DataFrame(rspfft_min_(_fft_df), columns=['_fft_min'])
        _fft_max = pd.DataFrame(rspfft_max_(_fft_df), columns=['_fft_max'])
        _fft_range = pd.DataFrame(rspfft_range_(_fft_max, _fft_min), columns=['_fft_range'])

    temp_feature_df = pd.DataFrame()
    if len(features) > 0:
        for i in features:
            if "_" + i in feature_list:
                temp_feature_df = pd.concat([locals()["_" + i], temp_feature_df], axis=1)
    else:
        for i in feature_list:
            temp_feature_df = pd.concat([locals()[i], temp_feature_df], axis=1)

    tmp_cols = temp_feature_df.columns
    temp_feature_df.columns = [ch_id + name for name in tmp_cols]

    print('--- RSP features ---')
    print(temp_feature_df.shape)
    return temp_feature_df
