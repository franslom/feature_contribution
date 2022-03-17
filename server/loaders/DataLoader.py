import server.loaders.DeapLoader as DeapLoader

loader = {
    'deap': DeapLoader.convert_dataset,
    'deap_ch': DeapLoader.load_channels
}


def load_channels(dataset, dataset_folder):
    if dataset in loader.keys():
        return loader[dataset + '_ch'](dataset_folder)
    return []


def convert_dataset(dataset, dataset_folder, out_folder):
    if dataset not in loader.keys():
        return
    loader[dataset](dataset_folder, out_folder)
