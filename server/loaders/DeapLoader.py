import _pickle
import json
import pandas as pd
from multiprocessing import Pool
from functools import partial
import os

"""
data	40 x 40 x 8064	video/trial x channel x data
labels	40 x 4	        video/trial x label (valence, arousal, dominance, liking)
* Valence	The valence rating (float between 1 and 9).
* Arousal	The arousal rating (float between 1 and 9).
"""

signalData = {
    'EEG': { 'channels': ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
                          'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2',
                          'P4', 'P8', 'PO4', 'O2'],
             'ini': 0 },
    'EOG': { 'channels': ['hEOG', 'vEOG'], 'ini': 32 },
    'EMG': { 'channels': ['zEMG', 'tEMG'], 'ini': 34 },
    'GSR': { 'channels': ['GSR'], 'ini': 36 },
    'RESP': { 'channels': ['Respiration'], 'ini': 37 },
    'BVP': {'channels': ['Plethysmograph'], 'ini': 38},
    'TEMP': {'channels': ['Temperature'], 'ini': 39},
}


def load_info(folder):
    return json.load(open(folder + "dataset_info.json", "r"))


def load_channels(folder_path):
    info = load_info(folder_path)
    channels = [{'id': ch['signal'] + '_' + ch['channel'], 'label': ch['signal'] + ' - ' + ch['channel']} for ch in info['channels']]
    return {'channels': channels, 'sampleSize': info['sampleSize']}


def create_labels_file(db_folder, dst_folder, subjects):
    header = True
    mode = 'w'
    for subj in subjects:
        sXX = _pickle.load(open(db_folder + subj + '.dat', 'rb'), encoding='latin1')
        sXX_df = pd.DataFrame(sXX['labels'][:, 0:2])
        sXX_df = sXX_df.round(0)
        sXX_df.columns = ['valence', 'arousal']
        sXX_df.index = [subj + '_v' + str(j) + "_0" for j in range(40)]
        sXX_df.to_csv(dst_folder + "all_df_y", mode=mode, header=header)
        if header:
            header = False
            mode = 'a'


def create_channels_files(in_data, subjects):
    out_fname = in_data[0]
    ch = in_data[1]
    db_folder = in_data[2]

    print("> File " + out_fname + " ...")
    mode = 'w'
    header = True
    for subj in subjects:
        sXX = _pickle.load(open(db_folder + subj + '.dat', 'rb'), encoding='latin1')
        channel_df = pd.DataFrame(sXX['data'][:, ch, :])
        channel_df.index = [subj + '_v' + str(j) + "_0" for j in range(40)]
        channel_df.to_csv(out_fname, mode=mode, header=header)
        if header:
            header = False
            mode = 'a'


def convert_dataset(path_db, output_folder):
    if 'info' not in globals():
        globals()['info'] = load_info(path_db)
    channels = globals()['info']['channels']
    subjects = globals()['info']['subjects']

    print("Generating file for the labels")
    create_labels_file(path_db, output_folder, subjects)

    # Names of the files to be generated
    signal_files = []
    for channel in channels:
        idCh = channel['signal'] + "_" + channel['channel'] + "_df_x"
        signal_files.append([output_folder + idCh, channel['idx'], path_db])

    print("Files for channels: ", len(signal_files))

    pool = Pool(os.cpu_count() - 2)
    #pool.map(create_channels_files, signal_files)
    tmp_fn = partial(create_channels_files, subjects=subjects)
    pool.map(tmp_fn, signal_files)


"""
    all_df_y = pd.DataFrame()
    channels_df_x = {}
    for subj in subjects:
        print("Loading " + subj + " data ...")
        sXX = _pickle.load(open(path_db + subj + '.dat', 'rb'), encoding='latin1')

        # save labels
        sXX_df = pd.DataFrame(sXX['labels'][:, 0:2])
        sXX_df = sXX_df.round(0)
        sXX_df.columns = ['valence', 'arousal']
        temp_index = []
        for j in range(40):
            temp_index.append(subj + '_' + str(j))
        sXX_df.index = temp_index
        all_df_y = pd.concat([all_df_y, sXX_df])

        # save channels
        for channel in channels:
            channel_df = pd.DataFrame(sXX['data'][:, channel['idx'], :])
            channel_df.index = temp_index
            idCh = channel['signal'] + "_" + channel['channel'] + "_df_x"
            if idCh not in channels_df_x:
                channels_df_x[idCh] = pd.DataFrame()
            channels_df_x[idCh] = pd.concat([channels_df_x[idCh], channel_df])

    _pickle.dump(all_df_y, open(output_folder + "all_df_y", "wb"))
    for fname in channels_df_x:
        _pickle.dump(channels_df_x[fname], open(output_folder + fname, "wb"))
"""

if __name__ == "__main__":
    data_folder = '../../datasets/deap_preprocessed/'
    convert_dataset(data_folder, '../../datasets/data_files/')
    #data = _pickle.load(open('../../datasets/data_files/all_df_y', 'rb'))
    #data = load_channels(data_folder)
    print(data)
