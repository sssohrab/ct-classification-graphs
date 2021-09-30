import os, json
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import sys


DEFAULT_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_DTYPE)

class DatasetTextRaw(Dataset):
    """
    This is the basic data loader that simply reads the ith document from storage to memory.
    """
    def __init__(self, source_dir, split_label_path, ext='.json', features_key='Basic'):
        if ext[0] != '.':
            ext = '.' + ext
        self.ext = ext
        self.features_key = features_key

        self.source_dir = source_dir
        with open(split_label_path, 'r') as f:
            lines = f.readlines()
        self.docids_list = [l.strip().split()[0] for l in lines]
        try:
            self.labels_list = [l.strip().split()[1] for l in lines]
        except:
            self.labels_list = [None for l in lines]

    def __len__(self):
        return len(self.docids_list)

    def __getitem__(self, item):
        label = self.labels_list[item]
        docid = self.docids_list[item]
        doc_path = os.path.join(self.source_dir, docid + self.ext)
        if self.ext == '.json':
            with open(doc_path, 'r') as f:
                doc = json.load(f)["Features"][self.features_key]
        if self.ext == '.xml':
            # TODO: Ask Alex!
            pass

        sample = dict()
        sample['docs'] = doc
        sample['docids'] = docid
        sample['labels'] = label

        return sample


class DatasetFeatureText(DatasetTextRaw):
    """
        Reads directly from text source (in json format) and converts to arrays based on a BOW object.
        """

    def __init__(self, featurizer_obj, source_dir, split_label_path, ext='.json', features_key='Basic'):
        super().__init__(source_dir, split_label_path, ext, features_key)
        self.featurizer_obj = featurizer_obj

    def __getitem__(self, item):
        _sample = super().__getitem__(item)

        if isinstance(_sample['docs'], dict):
            _doc = [str(v) for k, v in _sample['docs'].items()]
        if isinstance(_sample['docs'], str):
            _doc = [_sample['docs']]

        features = self.featurizer_obj(_doc)

        sample = dict()
        sample['features'] = features
        sample['docids'] = _sample['docids']
        sample['labels'] = int(_sample['labels']) if _sample['labels'] is not None else None
        sample['docs'] = _sample['docs']

        return sample


class DatasetFeatureArray(Dataset):
    """
    Having digitized all documents offline to numerical arrays using a featurizer object in h5 format, simply reads
    them for each id. Otherwise use DatasetFeatureText which is slower.
    """
    def __init__(self, source_h5_path, split_label_path):
        self.source_h5_path = source_h5_path
        self.split_label_path = split_label_path

        with open(split_label_path, 'r') as f:
            lines = f.readlines()
        self.docids_list = [l.strip().split()[0] for l in lines]
        self.labels_list = [l.strip().split()[1] for l in lines]

    def __len__(self):
        return len(self.docids_list)

    def __getitem__(self, item):
        label = self.labels_list[item]
        docid = self.docids_list[item]
        hf = h5py.File(self.source_h5_path, 'r')
        tfidf = np.array(hf.get(docid))

        sample = dict()
        sample['features'] = torch.from_numpy(tfidf).squeeze(0)
        sample['docids'] = docid
        sample['labels'] = int(label) if label is not None else None

        hf.close()
        return sample


def collate_fn(batch):

    batch = {k: [d[k] for d in batch] for k in batch[0]}
    if 'docs' in batch.keys():
        del batch['docs']
    batch['features'] = torch.tensor([b.tolist() for b in batch['features']])
    _b, _d = batch['features'].shape[0], batch['features'].shape[-1]
    batch['features'] = batch['features'].view(_b, -1, _d)
    batch['labels'] = torch.tensor(batch['labels']) if batch['labels'][0] is not None else [None] * len(batch['labels'])

    return batch
