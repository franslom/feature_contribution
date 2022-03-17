import pandas as pd
import numpy as np
import os
from sklearn import decomposition
from multiprocessing import Pool
from functools import partial
from server.features import GSR_feature_extract as gsr_fex
from server.features import EEG_feature_extract as eeg_fex
from server.features import RSP_feature_extract as rsp_fex
from server.features import BVP_feature_extract as bvp_fex


PREFIX_FILE = "features"
CHUNKSIZE = 200
FINAL_FILE = "all_features_x"

featureExtractor = {
    'EEG': eeg_fex,
    'GSR': gsr_fex,
    'RSP': rsp_fex,
    'BVP': bvp_fex
    #'EOG': None,
    #'EMG': None,
    #'TEMP': None,
}

featureSelector = {
    'pca': decomposition.PCA(n_components='mle')
}


def get_original_features(data_folder):
    df_data = pd.read_csv(data_folder + FINAL_FILE, index_col=0)
    return df_data


def preprocess_features(df_features):
    # remove columns with complex
    df_abs = df_features.select_dtypes(["object"])
    for col in df_abs.columns:
        df_features[col] = df_features[col].str.replace('i', 'j').apply(lambda x: np.abs(np.complex(x)))
    # remove columns with all data as NaN values
    # df_features.dropna(axis=0, how='all', inplace=True)
    df_features.dropna(axis=1, how='all', inplace=True)
    # refill NaN values with the average
    return df_features.fillna(df_features.mean())


def get_feature_list(id_signal, id_channel):
    f_list = featureExtractor[id_signal].feature_list
    return [id_signal + "_" + id_channel + fname for fname in f_list]


def extract_features_from_signal(id_signal, id_channel, features, data_signal):
    return featureExtractor[id_signal].extract_features(features, data_signal, id_signal + "_" + id_channel)


def extract_features_per_channel(ch_data, src_folder, dst_folder):
    sig_ch = ch_data["channel"]
    features = ch_data["features"]
    signal, channel = sig_ch.split('_')
    print("Extracting features from " + sig_ch)

    o_filename = ""
    if signal in featureExtractor.keys():
        o_filename = dst_folder + PREFIX_FILE + "_" + sig_ch
        df_data = pd.read_csv(src_folder + sig_ch + '_df_x', chunksize=CHUNKSIZE, index_col=0)
        mode = 'w'
        header = True
        for data_signal in df_data:
            tmp_df = extract_features_from_signal(signal, channel, features, data_signal)
            tmp_df.to_csv(o_filename, mode=mode, header=header)
            if header:
                header = False
                mode = 'a'
    return o_filename


def generate_file_allfeatures(files, dst_folder):
    # Generating a file which contains all extracted features
    features = []
    files = [elem for elem in files if elem != ""]
    for fname in files:
        features.append(pd.read_csv(fname, chunksize=CHUNKSIZE, index_col=0))

    mode = 'w'
    header = True
    for item in zip(*features):
        df_sdata = pd.concat(item, axis=1)
        df_sdata.to_csv(dst_folder + FINAL_FILE, mode=mode, header=header)
        if header:
            header = False
            mode = 'a'
    return dst_folder + FINAL_FILE


def extract_features(data_in, data_folder):
    selected_chs = data_in["channels"]

    # Computing features for the selected channels
    pool = Pool(os.cpu_count() - 2)
    tmp_fn = partial(extract_features_per_channel, src_folder=data_folder, dst_folder=data_folder)
    files = pool.map(tmp_fn, selected_chs)

    return generate_file_allfeatures(files, data_folder)


def select_features(fselector, features):
    if fselector in featureSelector.keys():
        tech = featureSelector[fselector]
        tech.fit(features)
        features = tech.transform(features)
    else:
        features = features.values
    return features


if __name__ == "__main__":
    data = {'channels': ['GSR_GSR', 'EEG_P8', 'RESP_Respiration'], 'winSize': 63, 'winIni': 0, 'sampleSize': 128}
    out_folder = '../../datasets/data_files/'
    """
    tmp_df = extract_features(data, out_folder)
    print("previous features: ", tmp_df.shape)
    new_feat = select_features('pca', tmp_df)
    print("selected features", new_feat.shape)
    """
    features = get_original_features(out_folder)
    print(features.shape)
