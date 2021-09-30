import json, os
import pickle
import sys
import argparse

sys.path.insert(0, '../../')
from ct.data import ROOT_DIR, BagOfWordsBasic

parser = argparse.ArgumentParser()
parser.add_argument('-c', default='./config_fit_BOW.json')
config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)


source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
train_path = os.path.join(ROOT_DIR, config["data"]["train_path"])

with open(train_path, 'r') as f:
    nctids_list = [nctid.strip().split()[0] for nctid in f.readlines()]


cts_dict_list = []
for nct in nctids_list:
    with open(os.path.join(source_dir, nct + '.json'), 'r') as f:
        ct = json.load(f)
        cts_dict_list.append(ct['Features'][config["features"]["key"]])

print('Fitting the BOW object ..')

dim = config["features"]["init"]["proj_dim"]
nnz_proj = config["features"]["init"]["nnz_proj"]
ngram_low = config["features"]["init"]["ngram_low"]
ngram_high = config["features"]["init"]["ngram_high"]
max_features = config["features"]["init"]["max_features"]

bow_obj = BagOfWordsBasic(ngram_range=(ngram_low, ngram_high), dim=dim, nnz_proj=nnz_proj, max_features=max_features)
bow_obj.fit(cts_dict_list)

bow_obj_path = os.path.join('../../storage', 'bow_obj_dim{}_nnzp{}_l{}_h{}_max{}.pkl'.format(
    dim, nnz_proj, ngram_low, ngram_high, max_features))

with open(bow_obj_path, 'wb') as f:
    pickle.dump(bow_obj, f, pickle.HIGHEST_PROTOCOL)

print('Fitted and saved a bag-of-words dictionary with ' + str(len(bow_obj.vectorizer.vocabulary_)) + ' tokens.')