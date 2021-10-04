import networkx as nx
import json, os, sys
import pickle

import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data import Dataset as Dataset_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx



class CTDictToPyGGraph:
    """
    From a raw hierarchical dictionary loaded from json-api, creates a torch_geometric graph data class.
    """

    def __init__(self, featurizer_obj=None, exclude_fields=('StatusModule', 'ResultsSection', 'DerivedSection'),
                 root_to_leaf=True, feature_key='x'):
        self.featurizer_obj = featurizer_obj  # Whether BOW-based, or transformer-based
        self.exclude_fields = exclude_fields  # Label-sensitive content
        self.root_to_leaf = root_to_leaf   # The direction of the graph edges
        self.feature_key = feature_key

        self.G_raw = nx.DiGraph()

    @staticmethod
    def _list_dict_to_dict_list(l_d):
        assert isinstance(l_d, list)
        assert isinstance(l_d[0], dict)

        d_l = dict()
        for l in l_d:
            for k in l.keys():
                if k not in d_l.keys():
                    d_l[k] = []
                d_l[k].append(l[k])
        return d_l

    def content_dict_to_nx_tree(self, content_dict):

        self.G_raw.clear()
        q = list(content_dict.items())
        while q:
            v, d = q.pop()
            if v in self.exclude_fields:
                if v in self.G_raw.nodes:
                    self.G_raw.remove_node(v)
                continue
            for nv, nd in d.items():
                if self.root_to_leaf:
                    self.G_raw.add_edge(v, nv)
                else:
                    self.G_raw.add_edge(nv, v)

                self.G_raw.nodes[v]['raw'] = ''

                if isinstance(nd, dict):
                    q.append((nv, nd))
                if isinstance(nv, str) and isinstance(nd, int):
                    self.G_raw.nodes[nv]['raw'] = str(nd)
                if isinstance(nd, str):
                    self.G_raw.nodes[nv]['raw'] = str(nd)
                if isinstance(nd, list) and isinstance(nd[0], list):
                    self.G_raw.nodes[nv]['raw'] = str(nd)

                if isinstance(nd, list):
                    if isinstance(nd[0], dict):
                        nd = self._list_dict_to_dict_list(nd)
                        q.append((nv, nd))

                if isinstance(nd, list):
                    if isinstance(nd[0], str):
                        self.G_raw.nodes[nv]['raw'] = str(nd)

        self.assert_has_raw()

    def assert_has_raw(self):
        for n in self.G_raw.nodes():
            if 'raw' not in self.G_raw.nodes[n].keys():
                self.G_raw.nodes[n]['raw'] = ''

    def raw_to_feature(self):
        G_feat = nx.DiGraph()
        G_feat.add_nodes_from(self.G_raw)
        G_feat.add_edges_from(self.G_raw.edges)

        for n in G_feat.nodes:
            if self.featurizer_obj:
                feature = self.featurizer_obj(self.G_raw.nodes[n]['raw'])
            else:
                feature = torch.tensor([float('nan')])
            G_feat.nodes[n][self.feature_key] = feature

        return G_feat

    def __call__(self, content_dict):
        self.content_dict_to_nx_tree(content_dict)
        G_feat = self.raw_to_feature()
        G_pyG = from_networkx(G_feat)
        G_pyG['nodes'] = list(self.G_raw.nodes)

        return G_pyG


class DatasetCTGraph(Dataset_geometric):
    def __init__(self, featurizer_obj, source_dir, split_label_path, feature_key='x', root_to_leaf=False,
                 exclude_fields=('StatusModule', 'ResultsSection', 'DerivedSection')):
        self.featurizer_obj = featurizer_obj
        self.source_dir = source_dir
        with open(split_label_path, 'r') as f:
            lines = f.readlines()
        self.docids_list = [l.strip().split()[0] for l in lines]
        try:
            self.labels_list = [l.strip().split()[1] for l in lines]
        except:
            self.labels_list = [None for l in lines]

        self.pyg_obj = CTDictToPyGGraph(self.featurizer_obj, feature_key=feature_key, root_to_leaf=root_to_leaf,
                           exclude_fields=exclude_fields)

    def __len__(self):
        return len(self.docids_list)

    def __getitem__(self, item):
        label = self.labels_list[item]
        docid = self.docids_list[item]
        doc_path = os.path.join(self.source_dir, docid + '.json')
        with open(doc_path, 'r') as f:
            doc = json.load(f)

        data = self.pyg_obj(doc)
        data['docids'] = docid
        data['labels'] = label

        return data


class DatasetCTGraphInMemory(InMemoryDataset):
    def __init__(self, root, dataset_base_obj, transform=None, pre_transform=None):
        self.dataset_base_obj = dataset_base_obj

        super(DatasetCTGraphInMemory, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def process(self):
        data_list = []
        for idx in range(len(self.dataset_base_obj)):
            if idx % 100 == 0:
                print(idx)

            data = self.dataset_base_obj[idx]
            data_list.append(data)

        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])

