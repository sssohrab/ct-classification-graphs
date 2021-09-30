import json, os
import pickle
import h5py
import sys
import torch

sys.path.insert(0, '../../')
from ct.data import ROOT_DIR, FeaturizerChannelHuggingFace
from ct.data.flat import DatasetFeatureText
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', default='./config_train.json')

config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)


source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
train_path = os.path.join(ROOT_DIR, config["data"]["train_path"])
valid_path = os.path.join(ROOT_DIR, config["data"]["valid_path"])


features_key = config["features"]["key"]    # whether a blob of text (BasicToText), or with 9 fields separated
features_type = config["features"]["type"]  # whether BOW-based or LM-based

if "bow" in features_type.lower():

    dim = config["features"]["init"]["proj_dim"]
    nnz_proj = config["features"]["init"]["nnz_proj"]
    ngram_low = config["features"]["init"]["ngram_low"]
    ngram_high = config["features"]["init"]["ngram_high"]
    max_features = config["features"]["init"]["max_features"]

    featurizer_obj_path = os.path.join(
        '../../storage', 'bow_obj_dim{}_nnzp{}_l{}_h{}_max{}.pkl'.format(
            dim, nnz_proj, ngram_low, ngram_high, max_features))

    if os.path.exists(featurizer_obj_path):
        with open(featurizer_obj_path, 'rb') as file:
            featurizer_obj = pickle.load(file)
    else:
        print("Didn't find the BOW featurizer object.. ")

    h5_path_train = os.path.join('../../storage', 'train_BOW_dim{}_nnzp{}_l{}_h{}_max{}.h5'.format(
        dim, nnz_proj, ngram_low, ngram_high, max_features))

    h5_path_valid = os.path.join('../../storage', 'valid_BOW_dim{}_nnzp{}_l{}_h{}_max{}.h5'.format(
        dim, nnz_proj, ngram_low, ngram_high, max_features))

elif "lm" in features_type.lower():
    device = torch.device(config["global"]["device"])

    model_name = config["features"]["init"]["model_name"]
    tokenizer_name = config["features"]["init"]["tokenizer_name"]

    max_len = config["features"]["init"]["max_len"]
    featurizer_obj = FeaturizerChannelHuggingFace(model_name=model_name, tokenizer_name=tokenizer_name, device=device,
                                              truncation=True, max_length=max_len)

    h5_path_train = os.path.join('../../storage', 'train_LM_max-length{}.h5'.format(512))
    h5_path_valid = os.path.join('../../storage', 'valid_LM_max-length{}.h5'.format(512))


dataset_train = DatasetFeatureText(featurizer_obj, source_dir, split_label_path=train_path, ext='.json',
                                   features_key=features_key)
dataset_valid = DatasetFeatureText(featurizer_obj, source_dir, split_label_path=valid_path, ext='.json',
                                   features_key=features_key)


if not os.path.exists(h5_path_train):
    print('Saving numerical arrays of the train set to disk..')
    hf_train = h5py.File(h5_path_train, 'w')

    for i_sample, _sample in enumerate(dataset_train):
        if i_sample % 1000 == 0:
            print(i_sample + 1)
        inp = _sample['features'].cpu().numpy()
        nctid = _sample['docids']
        hf_train.create_dataset(nctid, data=inp)

    hf_train.close()
    print('Done.')


if not os.path.exists(h5_path_valid):
    print('Saving numerical arrays of the valid set to disk..')
    hf_valid = h5py.File(h5_path_valid, 'w')

    for i_sample, _sample in enumerate(dataset_valid):
        if i_sample % 1000 == 0:
            print(i_sample + 1)
        inp = _sample['features'].cpu().numpy()
        nctid = _sample['docids']
        hf_valid.create_dataset(nctid, data=inp)

    hf_valid.close()
    print('Done.')
