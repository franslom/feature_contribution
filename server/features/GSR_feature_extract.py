# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 21:02:45 2018
@author: jinyx
"""

import pandas as pd
import numpy as np

feature_list = ['_mean', '_median', '_std', '_min', '_max', '_range',
                '_minRatio', '_maxRatio', '_1Diff_mean', '_1Diff_median',
                '_1Diff_std', '_1Diff_min', '_1Diff_max', '_1Diff_range',
                '_1Diff_minRatio', '_1Diff_maxRatio', '_2Diff_std',
                '_2Diff_min', '_2Diff_max', '_2Diff_range', '_2Diff_minRatio',
                '_2Diff_maxRatio', '_fft_mean', '_fft_median', '_fft_std',
                '_fft_min', '_fft_max', '_fft_range']


def sc_mean_(df):
    return df.mean(axis=1)


def sc_median_(df):
    return df.median(axis=1)


def sc_std_(df):
    return df.std(axis=1)


def sc_min_(df):
    return df.min(axis=1)


def sc_max_(df):
    return df.max(axis=1)


def sc_range_(df_max, df_min):
    return df_max['_max'] - df_min['_min']


# 最小值比率 = Mmin/N
def sc_minRatio_(all_df, sc_min):
    all_df_T = all_df.T
    sc_min_T = sc_min.T
    sc_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len(all_df_T[i][all_df_T[i] == sc_min_T.at['_min', i]])
        sc_minRatio_dict.update({i: num_min / 8064.0})
    sc_minRatio_df = pd.DataFrame.from_dict(data=sc_minRatio_dict, orient='index')
    sc_minRatio_df.columns = ['_minRatio']
    return sc_minRatio_df


# 最大值比率 = Nmax/N
def sc_maxRatio_(all_df, sc_max):
    all_df_T = all_df.T
    sc_max_T = sc_max.T
    sc_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len(all_df_T[i][all_df_T[i] == sc_max_T.at['_max', i]])
        sc_maxRatio_dict.update({i: num_max / 8064.0})
    sc_maxRatio_df = pd.DataFrame.from_dict(data=sc_maxRatio_dict, orient='index')
    sc_maxRatio_df.columns = ['_maxRatio']
    return sc_maxRatio_df


# GSR一阶差分均值
def sc1Diff_mean_(all_df):
    sc1Diff_mean = all_df.diff(periods=1, axis=1).dropna(axis=1).mean(axis=1)
    return sc1Diff_mean


# GSR一阶差分中值
def sc1Diff_median_(all_df):
    sc1Diff_median = all_df.diff(periods=1, axis=1).dropna(axis=1).median(axis=1)
    return sc1Diff_median


# GSR一阶差分标准差
def sc1Diff_std_(all_df):
    sc1Diff_std = all_df.diff(periods=1, axis=1).dropna(axis=1).std(axis=1)
    return sc1Diff_std


def sc1Diff_min_(all_df):
    sc1Diff_min = all_df.diff(periods=1, axis=1).dropna(axis=1).min(axis=1)
    return sc1Diff_min


def sc1Diff_max_(all_df):
    sc1Diff_max = all_df.diff(periods=1, axis=1).dropna(axis=1).max(axis=1)
    return sc1Diff_max


def sc1Diff_range_(sc1Diff_max, sc1Diff_min):
    return sc1Diff_max['_1Diff_max'] - sc1Diff_min['_1Diff_min']


def sc1Diff_minRatio_(all_df, sc1Diff_min):
    all_df_Diff_T = all_df.diff(periods=1, axis=1).dropna(axis=1).T
    sc1Diff_min_T = sc1Diff_min.T
    sc1Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len(all_df_Diff_T[i][all_df_Diff_T[i] == sc1Diff_min_T.at['_1Diff_min', i]])
        sc1Diff_minRatio_dict.update({i: num_min / 8063.0})
    sc1Diff_minRatio_df = pd.DataFrame.from_dict(data=sc1Diff_minRatio_dict, orient='index')
    return sc1Diff_minRatio_df


def sc1Diff_maxRatio_(all_df, sc1Diff_max):
    all_df_Diff_T = all_df.diff(periods=1, axis=1).dropna(axis=1).T
    sc1Diff_max_T = sc1Diff_max.T
    sc1Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len(all_df_Diff_T[i][all_df_Diff_T[i] == sc1Diff_max_T.at['_1Diff_max', i]])
        sc1Diff_maxRatio_dict.update({i: num_max / 8063.0})
    sc1Diff_maxRatio_df = pd.DataFrame.from_dict(data=sc1Diff_maxRatio_dict, orient='index')
    return sc1Diff_maxRatio_df


def sc2Diff_std_(all_df):
    sc2Diff_std = all_df.diff(periods=2, axis=1).dropna(axis=1).std(axis=1)
    return sc2Diff_std


def sc2Diff_min_(all_df):
    sc2Diff_min = all_df.diff(periods=2, axis=1).dropna(axis=1).min(axis=1)
    return sc2Diff_min


def sc2Diff_max_(all_df):
    sc2Diff_max = all_df.diff(periods=2, axis=1).dropna(axis=1).max(axis=1)
    return sc2Diff_max


def sc2Diff_range_(sc2Diff_max, sc2Diff_min):
    sc2Diff_range = sc2Diff_max['_2Diff_max'] - sc2Diff_min['_2Diff_min']
    return sc2Diff_range


def sc2Diff_minRatio_(all_df, sc2Diff_min):
    all_df_2Diff_T = all_df.diff(periods=2, axis=1).dropna(axis=1).T
    sc2Diff_min_T = sc2Diff_min.T
    sc2Diff_minRatio_dict = {}
    for i in all_df.index.tolist():
        num_min = len(all_df_2Diff_T[i][all_df_2Diff_T[i] == sc2Diff_min_T.at['_2Diff_min', i]])
        sc2Diff_minRatio_dict.update({i: num_min / 8062.0})
    sc2Diff_minRatio_df = pd.DataFrame.from_dict(data=sc2Diff_minRatio_dict, orient='index')
    return sc2Diff_minRatio_df


def sc2Diff_maxRatio_(all_df, sc2Diff_max):
    all_df_2Diff_T = all_df.diff(periods=2, axis=1).dropna(axis=1).T
    sc2Diff_max_T = sc2Diff_max.T
    sc2Diff_maxRatio_dict = {}
    for i in all_df.index.tolist():
        num_max = len(all_df_2Diff_T[i][all_df_2Diff_T[i] == sc2Diff_max_T.at['_2Diff_max', i]])
        sc2Diff_maxRatio_dict.update({i: num_max / 8062.0})
    sc2Diff_maxRatio_df = pd.DataFrame.from_dict(data=sc2Diff_maxRatio_dict, orient='index')
    return sc2Diff_maxRatio_df


# GSR DFT(FFT)频域数据
def scfft_(all_df):
    scfft_df = pd.DataFrame()
    for i in all_df.index.tolist():
        temp_scfft = pd.DataFrame(np.fft.fft(all_df.loc[i, :].values)).T
        temp_scfft.index = [i]
        scfft_df = scfft_df.append(temp_scfft)
    return scfft_df


# GSR 频域中值
def scfft_mean_(scfft_df):
    scfft_mean = scfft_df.mean(axis=1)
    return scfft_mean


def scfft_median_(scfft_df):
    scfft_median = scfft_df.median(axis=1)
    return scfft_median


def scfft_std_(scfft_df):
    scfft_std = scfft_df.std(axis=1)
    return scfft_std


def scfft_min_(scfft_df):
    scfft_min = scfft_df.min(axis=1)
    return scfft_min


def scfft_max_(scfft_df):
    scfft_max = scfft_df.max(axis=1)
    return scfft_max


def scfft_range_(scfft_max, scfft_min):
    scfft_range = scfft_max['_fft_max'] - scfft_min['_fft_min']
    return scfft_range


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
        _mean = pd.DataFrame(sc_mean_(df_data_x), columns=['_mean'])
        _median = pd.DataFrame(sc_median_(df_data_x), columns=['_median'])
        _std = pd.DataFrame(sc_std_(df_data_x), columns=['_std'])
        _min = pd.DataFrame(sc_min_(df_data_x), columns=['_min'])
        _max = pd.DataFrame(sc_max_(df_data_x), columns=['_max'])
        _range = pd.DataFrame(sc_range_(_max, _min), columns=['_range'])
        _minRatio = pd.DataFrame(sc_minRatio_(df_data_x, _min), columns=['_minRatio'])
        _maxRatio = pd.DataFrame(sc_maxRatio_(df_data_x, _max), columns=['_maxRatio'])

    if len(features) == 0 or len(
            set(features) & {'1Diff_mean', '1Diff_median', '1Diff_std', '1Diff_min', '1Diff_max', '1Diff_range',
                             '1Diff_minRatio', '1Diff_maxRatio'}) > 0:
        _1Diff_mean = pd.DataFrame(sc1Diff_mean_(df_data_x), columns=['_1Diff_mean'])
        _1Diff_median = pd.DataFrame(sc1Diff_median_(df_data_x), columns=['_1Diff_median'])
        _1Diff_std = pd.DataFrame(sc1Diff_std_(df_data_x), columns=['_1Diff_std'])
        _1Diff_min = pd.DataFrame(sc1Diff_min_(df_data_x), columns=['_1Diff_min'])
        _1Diff_max = pd.DataFrame(sc1Diff_max_(df_data_x), columns=['_1Diff_max'])
        _1Diff_range = pd.DataFrame(sc1Diff_range_(_1Diff_max, _1Diff_min), columns=['_1Diff_range'])
        _1Diff_minRatio = sc1Diff_minRatio_(df_data_x, _1Diff_min)
        _1Diff_minRatio.columns = ['_1Diff_minRatio']
        _1Diff_maxRatio = sc1Diff_maxRatio_(df_data_x, _1Diff_max)
        _1Diff_maxRatio.columns = ['_1Diff_maxRatio']

    if len(features) == 0 or len(
            set(features) & {'2Diff_std', '2Diff_min', '2Diff_max', '2Diff_range', '2Diff_minRatio',
                             '2Diff_maxRatio'}) > 0:
        _2Diff_std = pd.DataFrame(sc2Diff_std_(df_data_x), columns=['_2Diff_std'])
        _2Diff_min = pd.DataFrame(sc2Diff_min_(df_data_x), columns=['_2Diff_min'])
        _2Diff_max = pd.DataFrame(sc2Diff_max_(df_data_x), columns=['_2Diff_max'])
        _2Diff_range = pd.DataFrame(sc2Diff_range_(_2Diff_max, _2Diff_min), columns=['_2Diff_range'])
        _2Diff_minRatio = sc2Diff_minRatio_(df_data_x, _2Diff_min)
        _2Diff_minRatio.columns = ['_2Diff_minRatio']
        _2Diff_maxRatio = sc2Diff_maxRatio_(df_data_x, _2Diff_max)
        _2Diff_maxRatio.columns = ['_2Diff_maxRatio']

    if len(features) == 0 or len(
            set(features) & {'fft_mean', 'fft_median', 'fft_std', 'fft_min', 'fft_max', 'fft_range'}) > 0:
        _fft_df = scfft_(df_data_x)
        _fft_mean = pd.DataFrame(scfft_mean_(_fft_df), columns=['_fft_mean'])
        _fft_median = pd.DataFrame(scfft_median_(_fft_df), columns=['_fft_median'])
        _fft_std = pd.DataFrame(scfft_std_(_fft_df), columns=['_fft_std'])
        _fft_min = pd.DataFrame(scfft_min_(_fft_df), columns=['_fft_min'])
        _fft_max = pd.DataFrame(scfft_max_(_fft_df), columns=['_fft_max'])
        _fft_range = pd.DataFrame(scfft_range_(_fft_max, _fft_min), columns=['_fft_range'])

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

    print('--- GSR features ---')
    print(temp_feature_df.shape)
    return temp_feature_df
