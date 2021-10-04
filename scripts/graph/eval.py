import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from torch_geometric.data import DataLoader

import pickle
import sys, os, json, shutil
sys.path.insert(0, '../../')

from ct.data import ROOT_DIR, BagOfWordsBasic, FeaturizerChannelHuggingFace
from ct.data.graph import CTDictToPyGGraph, DatasetCTGraph, DatasetCTGraphInMemory
from ct.networks.graph import GCNGlobal, GCNSelective, get_indices_batch, GATGlobal, GATSelective, BOWFeaturizer
from ct.utils import AverageMeter, loss_function, get_metrics, metrics_list

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', default='./config_eval.json')
config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

generate_run = True

with open(os.path.join('./runs', config["train_run"], 'used_config.json'), 'r') as _json_file:
    _config = json.load(_json_file)
    config["features"] = _config["features"]
    config["network"] = _config["network"]
    config["pooling"] = _config["pooling"]
    config["data"]["exclude_fields"] = _config["data"]["exclude_fields"]
    config["data"]["root_to_leaf"] = _config["data"]["root_to_leaf"]


device = torch.device(config["global"]["device"])

features_type = config["features"]["type"]

if 'bow' in features_type.lower():
    tag = 'BOW'
    with open(config["features"]["init"]["path"], 'rb') as file:
        bow_obj = pickle.load(file)

    class Dummy:
        def __init__(self, bow_obj):
            self.bow_obj = bow_obj

        def __call__(self, raw):
            feat = self.bow_obj.transform_project([raw]).reshape(-1)
            return feat

    featurizer_obj = Dummy(bow_obj)

elif 'lm' in features_type.lower():
    tag = 'LM'
    model_name = config["features"]["init"]["model_name"]
    tokenizer_name = config["features"]["init"]["tokenizer_name"]
    max_len = config["features"]["init"]["max_len"]

    featurizer_obj = FeaturizerChannelHuggingFace(model_name=model_name, tokenizer_name=tokenizer_name, device=device,
                                                  truncation=True, max_length=512)

    class Dummy:
        def __init__(self, f_o):
            self.f_o = f_o

        def __call__(self, raw):
            return self.f_o([raw]).reshape(-1).cpu().numpy()

    featurizer_obj = Dummy(featurizer_obj)

if config["pooling"]["method"].lower() == 'global':
    tag += '_GLOBAL'

elif config["pooling"]["method"].lower() == 'selective':
    tag += '_SELECTIVE'


source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
split_test_path = os.path.join(ROOT_DIR, config["data"]["split_test_path"])

graph_set_path_test = os.path.join(ROOT_DIR, config["data"]["target_test_dir"])
exclude_fields = config["data"]["exclude_fields"]
root_to_leaf = config["data"]["root_to_leaf"]

dataset_base_test = DatasetCTGraph(featurizer_obj, source_dir, split_test_path, feature_key='x', root_to_leaf=root_to_leaf,
                                    exclude_fields=tuple(exclude_fields))


dataset_test = DatasetCTGraphInMemory(graph_set_path_test, dataset_base_test)


dataloader_test = DataLoader(dataset_test, batch_size=config["global"]["batch"],
                              shuffle=False, num_workers=config["global"]["workers"])

if generate_run:
    writer = SummaryWriter(comment='_eval' + '_' + tag)
    shutil.copy(config_path, os.path.join(writer.log_dir, 'used_config.json'))


#  Initialize the model:
state_dict_path = os.path.join('./runs', config["train_run"], 'net.pth')
if config["pooling"]["method"].lower() == 'global':
    net = GCNGlobal(config_network=config["network"]).to(device)
    # net = GATGlobal(config_network=config["network"]).to(device)
elif config["pooling"]["method"].lower() == 'selective':
    net = GCNSelective(config_network=config["network"],
                       len_base_nodes_list=len(config["pooling"]["base_nodes_list"])).to(device)

    # net = GATSelective(config_network=config["network"],
    #                    len_base_nodes_list=len(config["pooling"]["base_nodes_list"])).to(device)

net.load_state_dict(torch.load(state_dict_path, map_location=device))
net.to(device)
net.eval()

print(net)

metrics_test = {k: AverageMeter(k) for k in metrics_list}

loss_test = AverageMeter('loss_test')

out_list = []
lbl_list = []
docids_list = []
loss_batch_list_test = []
batch_list = []

if config["pooling"]["method"].lower() == 'selective':
    latent_base_array = np.array([]).reshape(0, 1+len(config["pooling"]["base_nodes_list"]), net.d_latent)
    latent_global_array = np.array([]).reshape(0, net.d_h)

with torch.no_grad():
    for i_batch, _batch in enumerate(dataloader_test):
        print('.. Evaluating batch {} from the test set..'.format(i_batch))

        inp = _batch.to(device)
        lbl = torch.tensor([int(_l) for _l in _batch['labels']]).float().to(device)

        if config["pooling"]["method"].lower() == 'global':
            out = net(inp.x.float(), inp.edge_index, inp.batch).view(-1)

        elif config["pooling"]["method"].lower() == 'selective':
            base = get_indices_batch(_batch.nodes, config["pooling"]["base_nodes_list"])
            out = net(inp.x.float(), inp.edge_index, inp.batch, base).view(-1)
            latent_base = net.latent_base.cpu().numpy()
            latent_base_array = np.concatenate((latent_base_array, latent_base), axis=0)

            latent_global = net.latent_global.cpu().numpy()
            latent_global_array = np.concatenate((latent_global_array, latent_global), axis=0)

        docids_list.extend(_batch['docids'])
        loss = loss_function(out, lbl)
        print('loss of this batch: {}'.format(loss.item()))
        loss_batch_list_test.append(loss.item())
        batch_list.append(out.shape[0])

        out_list.extend(out.tolist())
        lbl_list.extend(lbl.tolist())


avg_loss_test = sum([loss_batch_list_test[_i] * batch_list[_i]
                     for _i in range(len(batch_list))]) / sum(batch_list)
loss_test.update(avg_loss_test)

if generate_run:
    writer.add_scalar('loss/test', loss_test.val, loss_test.count)
    writer.add_pr_curve('pr_curve/test', labels=torch.tensor(lbl_list), predictions=torch.tensor(out_list), num_thresholds=127)

metrics_dict = get_metrics(out.cpu().numpy(), lbl.cpu().numpy(), metrics_list)
for k in metrics_list:
    metrics_test[k].update(metrics_dict[k])
    if generate_run:
        writer.add_scalar(k + '/test', metrics_test[k].val, metrics_test[k].count)
print(metrics_dict)

if generate_run:
    writer.close()

if generate_run:
    with open(os.path.join(writer.log_dir, 'scores.txt'), 'w') as f:
        line = ' '.join(['NCTId', 'label', 'score']) + '\n'
        f.write(line)
        for i_docid, docid in enumerate(docids_list):
            line = ' '.join([docid, str(lbl_list[i_docid]), str(out_list[i_docid])]) + '\n'
            f.write(line)


with open(os.path.join(writer.log_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics_dict, f, indent=2)


if config["pooling"]["method"].lower() == 'selective':

    sel_idx = np.random.choice(len(lbl_list), size=1000, replace=False)
    lbl_list = np.array(lbl_list)[sel_idx]
    latent_base_array = latent_base_array[sel_idx, :, :]
    latent_global_array = latent_global_array[sel_idx, :]



    base_nodes_list = config["pooling"]["base_nodes_list"]

    plt.style.use('ggplot')
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))
    fig.tight_layout()

    for i_base, base in enumerate(base_nodes_list):
        base = base.replace('Module', '')
        x_tsne = TSNE(n_components=2, perplexity=50).fit_transform(latent_base_array[:, i_base, :])

        axs[i_base // 3, i_base % 3].scatter(x_tsne[lbl_list == 1, 0], x_tsne[lbl_list == 1, 1], alpha=0.4)
        axs[i_base // 3, i_base % 3].scatter(x_tsne[lbl_list == 0, 0], x_tsne[lbl_list == 0, 1], alpha=0.4)
        axs[i_base // 3, i_base % 3].set_title(base, fontsize=25)

    plt.legend(['not completed', 'completed'], fontsize=35, loc="upper left")
    plt.savefig(os.path.join(writer.log_dir,'./latent_base.pdf'), bbox_inches='tight', format='pdf')

    x_tsne = TSNE(n_components=2, perplexity=50).fit_transform(latent_global_array)
    fig = plt.figure(figsize=(18, 18))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(x_tsne[lbl_list == 1, 0], x_tsne[lbl_list == 1, 1], alpha=0.4)
    ax.scatter(x_tsne[lbl_list == 0, 0], x_tsne[lbl_list == 0, 1], alpha=0.4)

    plt.legend(['not completed', 'completed'], fontsize=35)
    plt.savefig(os.path.join(writer.log_dir,'./latent_global.pdf'), bbox_inches='tight', format='pdf')