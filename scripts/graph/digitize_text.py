import torch
import sys, json, os, pickle
sys.path.insert(0, '../../')
from ct.data import ROOT_DIR, BagOfWordsBasic, FeaturizerChannelHuggingFace
from ct.data.graph import CTDictToPyGGraph, DatasetCTGraph, DatasetCTGraphInMemory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', default='./config_set.json')
config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

features_type = config["features"]["type"]

if 'bow' in features_type.lower():
    with open(config["features"]["init"]["path"], 'rb') as file:
        bow_obj = pickle.load(file)

    class Featurizer:
        def __init__(self, bow_obj):
            self.bow_obj = bow_obj

        def __call__(self, raw):
            feat = self.bow_obj.transform_project([raw]).reshape(-1)
            return feat

    featurizer_obj = Featurizer(bow_obj)

elif 'lm' in features_type.lower():
    model_name = config["features"]["init"]["model_name"]
    tokenizer_name = config["features"]["init"]["tokenizer_name"]
    max_len = config["features"]["init"]["max_len"]

    device = torch.device(config["global"]["device"])

    featurizer_obj = FeaturizerChannelHuggingFace(model_name=model_name, tokenizer_name=tokenizer_name,
                                                  device=device,
                                                  truncation=True, max_length=512)

    class Featurizer:
        def __init__(self, f_o):
            self.f_o = f_o

        def __call__(self, raw):
            return self.f_o([raw]).reshape(-1).cpu().numpy()


    featurizer_obj = Featurizer(featurizer_obj)

pyg_obj = CTDictToPyGGraph(featurizer_obj=featurizer_obj, feature_key='x', root_to_leaf=False,
                           exclude_fields=tuple(config["data"]["exclude_fields"]))

source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
split_train_path = os.path.join(ROOT_DIR, config["data"]["split_train_path"])
split_valid_path = os.path.join(ROOT_DIR, config["data"]["split_valid_path"])
split_test_path = os.path.join(ROOT_DIR, config["data"]["split_test_path"])

graph_set_path_train = os.path.join(ROOT_DIR, config["data"]["target_train_dir"])
graph_set_path_valid = os.path.join(ROOT_DIR, config["data"]["target_valid_dir"])
graph_set_path_test = os.path.join(ROOT_DIR, config["data"]["target_test_dir"])

exclude_fields = config["data"]["exclude_fields"]
root_to_leaf = config["data"]["root_to_leaf"]

dataset_base_train = DatasetCTGraph(featurizer_obj, source_dir, split_train_path, feature_key='x',
                                    root_to_leaf=root_to_leaf, exclude_fields=tuple(exclude_fields))

dataset_base_valid = DatasetCTGraph(featurizer_obj, source_dir, split_valid_path, feature_key='x',
                                    root_to_leaf=root_to_leaf, exclude_fields=tuple(exclude_fields))

dataset_base_test = DatasetCTGraph(featurizer_obj, source_dir, split_test_path, feature_key='x',
                                   root_to_leaf=root_to_leaf, exclude_fields=tuple(exclude_fields))

print('Creating the in-memory dataset for the training set at {}'.format(graph_set_path_train))
dataset_train = DatasetCTGraphInMemory(graph_set_path_train, dataset_base_train)

print('Creating the in-memory dataset for the validation set at {}'.format(graph_set_path_valid))
dataset_valid = DatasetCTGraphInMemory(graph_set_path_valid, dataset_base_valid)

print('Creating the in-memory dataset for the test set at {}'.format(graph_set_path_test))
dataset_test = DatasetCTGraphInMemory(graph_set_path_test, dataset_base_test)

print('All done.')
