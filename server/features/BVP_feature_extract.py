# Features based on https://github.com/PGomes92/pyhrv
# https://pyhrv.readthedocs.io/en/latest/index.html

import pandas as pd
import biosppy
import pyhrv.time_domain as td
import pyhrv
import warnings

warnings.filterwarnings("ignore")

feature_list = ['_nni_mean', '_nni_min', '_nni_max', '_hr_mean', '_hr_min', '_hr_max', '_hr_std',
                '_nni_diff_mean', '_nni_diff_min', '_nni_diff_max', '_sdnn',
                '_rmssd', '_sdsd', '_nn50', '_pnn50', '_nn20', '_pnn20'
]


def bvp_features(signals, col_names, hrv_func):
    _features = pd.DataFrame(columns=col_names, index=signals.index)
    for i in range(len(signals)):
        try:
            rpeaks = biosppy.signals.ecg.ecg(signals.iloc[i].tolist(), show=False)[2]
            result = hrv_func(rpeaks=rpeaks)
            _features.iloc[i] = dict(result)
        except:
            print("Exception in:", i)
            continue
    return _features

"""
def bvp_frequency_features(signals):
    _features = pd.DataFrame(columns=['fft_peak_vlf', 'fft_peak_lf', 'fft_peak_hf',
                'fft_abs_vlf', 'fft_abs_lf', 'fft_abs_hf',
                'fft_rel_vlf', 'fft_rel_lf', 'fft_rel_hf',
                'fft_log_vlf', 'fft_log_lf', 'fft_log_hf',
                'fft_norm_lf', 'fft_norm_hf', 'fft_ratio'], index=signals.index)
    for i in range(len(signals)):
        try:
            result = pyhrv.frequency_domain.frequency_domain(signal=signals.iloc[i].tolist())
            _features.iloc[i]['fft_peak_vlf'] = result["fft_peak"][1]
            plt.close('all')
        except:
            print("exception in:", i)
            continue
    print(_features)
    return _features
"""


def extract_features(features, df_data_x, ch_id=""):
    df_features = pd.DataFrame()

    if len(features) == 0 or len(set(features) & {'nni_mean', 'nni_min', 'nni_max'}) > 0:
        df_tmp = bvp_features(df_data_x, ['nni_mean', 'nni_min', 'nni_max'], td.nni_parameters)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'nni_diff_mean', 'nni_diff_min', 'nni_diff_max'}) > 0:
        df_tmp = bvp_features(df_data_x, ['nni_diff_mean', 'nni_diff_min', 'nni_diff_max'], td.nni_differences_parameters)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'hr_mean', 'hr_min', 'hr_max', 'hr_std'}) > 0:
        df_tmp = bvp_features(df_data_x, ['hr_mean', 'hr_min', 'hr_max', 'hr_std'], td.hr_parameters)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'sdnn'}) > 0:
        df_tmp = bvp_features(df_data_x, ['sdnn'], td.sdnn)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'rmssd'}) > 0:
        df_tmp = bvp_features(df_data_x, ['rmssd'], td.rmssd)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'sdsd'}) > 0:
        df_tmp = bvp_features(df_data_x, ['sdsd'], td.sdsd)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'nn20', 'pnn20'}) > 0:
        df_tmp = bvp_features(df_data_x, ['nn20', 'pnn20'], td.nn20)
        df_features = pd.concat([df_features, df_tmp], axis=1)
    if len(features) == 0 or len(set(features) & {'nn50', 'pnn50'}) > 0:
        df_tmp = bvp_features(df_data_x, ['nn50', 'pnn50'], td.nn50)
        df_features = pd.concat([df_features, df_tmp], axis=1)

    tmp_cols = df_features.columns
    df_features.columns = [ch_id + "_" + name for name in tmp_cols]

    print('--- BVP features ---')
    print(df_features.shape)
    return df_features
