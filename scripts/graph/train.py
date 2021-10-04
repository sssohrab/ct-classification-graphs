import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('-c', default='./config_train.json')
config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

device = torch.device(config["global"]["device"])
features_type = config["features"]["type"]

if 'bow' in features_type.lower():
    tag = 'BOW'
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
    tag = 'LM'
    model_name = config["features"]["init"]["model_name"]
    tokenizer_name = config["features"]["init"]["tokenizer_name"]
    max_len = config["features"]["init"]["max_len"]

    featurizer_obj = FeaturizerChannelHuggingFace(model_name=model_name, tokenizer_name=tokenizer_name, device=device,
                                                  truncation=True, max_length=512)

    class Featurizer:
        def __init__(self, f_o):
            self.f_o = f_o

        def __call__(self, raw):
            return self.f_o([raw]).reshape(-1).cpu().numpy()

    featurizer_obj = Featurizer(featurizer_obj)

if config["pooling"]["method"].lower() == 'global':
    tag += '_GLOBAL'

elif config["pooling"]["method"].lower() == 'selective':
    tag += '_SELECTIVE'


source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
split_train_path = os.path.join(ROOT_DIR, config["data"]["split_train_path"])
split_valid_path = os.path.join(ROOT_DIR, config["data"]["split_valid_path"])

graph_set_path_train = os.path.join(ROOT_DIR, config["data"]["target_train_dir"])
graph_set_path_valid = os.path.join(ROOT_DIR, config["data"]["target_valid_dir"])
exclude_fields = config["data"]["exclude_fields"]
root_to_leaf = config["data"]["root_to_leaf"]

dataset_base_train = DatasetCTGraph(featurizer_obj, source_dir, split_train_path, feature_key='x',
                                    root_to_leaf=root_to_leaf, exclude_fields=tuple(exclude_fields))

dataset_base_valid = DatasetCTGraph(featurizer_obj, source_dir, split_valid_path, feature_key='x',
                                    root_to_leaf=root_to_leaf, exclude_fields=tuple(exclude_fields))

dataset_train = DatasetCTGraphInMemory(graph_set_path_train, dataset_base_train)
dataset_valid = DatasetCTGraphInMemory(graph_set_path_valid, dataset_base_valid)


dataloader_train = DataLoader(dataset_train, batch_size=config["global"]["batch"],
                              shuffle=True, num_workers=config["global"]["workers"])
dataloader_valid = DataLoader(dataset_valid, batch_size=config["global"]["batch"],
                              shuffle=False, num_workers=config["global"]["workers"])

writer = SummaryWriter(comment='_train' + '_' + tag)
shutil.copy(config_path, os.path.join(writer.log_dir, 'used_config.json'))


#  Initialize the model:
if config["pooling"]["method"].lower() == 'global':
    net = GCNGlobal(config_network=config["network"]).to(device)
    # net = GATGlobal(config_network=config["network"]).to(device)
elif config["pooling"]["method"].lower() == 'selective':
    net = GCNSelective(config_network=config["network"],
                       len_base_nodes_list=len(config["pooling"]["base_nodes_list"])).to(device)

    # net = GATSelective(config_network=config["network"],
    #                    len_base_nodes_list=len(config["pooling"]["base_nodes_list"])).to(device)

print(net)

# The optimizer
optimizer = optim.Adam(net.parameters(), lr=config["global"]["lr"], weight_decay=config["global"]["weight_decay"])

metrics_dict_list_valid = {m: [] for m in metrics_list}  # Only for valid set.
metrics_dict_list_valid['loss'] = []

metrics_train = {k: AverageMeter(k) for k in metrics_list}
metrics_valid = {k: AverageMeter(k) for k in metrics_list}

loss_train = AverageMeter('loss_train')
loss_valid = AverageMeter('loss_valid')


best_epoch_loss_valid = 100  # Just some large number
best_arg_epoch = -1

for i_epoch in range(config["global"]["epochs"]):
    print(' **************** epoch number', i_epoch + 1, ' **********************')

    net.train()
    out_list = []
    lbl_list = []
    for i_batch, _batch in enumerate(dataloader_train):
        inp = _batch.to(device)
        lbl = torch.tensor([int(_l) for _l in _batch['labels']]).float().to(device)

        if config["pooling"]["method"].lower() == 'global':
            out = net(inp.x.float(), inp.edge_index, inp.batch).view(-1)

        elif config["pooling"]["method"].lower() == 'selective':
            base = get_indices_batch(_batch.nodes, config["pooling"]["base_nodes_list"])
            out = net(inp.x.float(), inp.edge_index, inp.batch, base).view(-1)

        optimizer.zero_grad()
        loss = loss_function(out, lbl)
        loss_train.update(loss.item())
        if i_batch % 1 == 0:
            print('train iter = ', i_batch + 1, ': ', loss_train.val)
        writer.add_scalar('loss/train', loss_train.val, loss_train.count)

        with torch.no_grad():
            metrics_dict = get_metrics(out.cpu().numpy(), lbl.cpu().numpy(), metrics_list)
            for k in metrics_list:
                metrics_train[k].update(metrics_dict[k])
                writer.add_scalar(k + '/train', metrics_train[k].val, metrics_train[k].count)

        loss.backward()
        optimizer.step()

        out_list.extend(out.tolist())
        lbl_list.extend(lbl.tolist())

    writer.add_pr_curve('pr_curve/train', labels=torch.tensor(lbl_list), predictions=torch.tensor(out_list),
                        global_step=i_epoch)

    print(' **************** validation at epoch ', i_epoch + 1, ' **********************')
    net.eval()
    out_list = []
    lbl_list = []
    loss_batch_list_valid = []
    batch_list = []

    with torch.no_grad():
        for i_batch, _batch in enumerate(dataloader_valid):
            inp = _batch.to(device)
            lbl = torch.tensor([int(_l) for _l in _batch['labels']]).float().to(device)

            if config["pooling"]["method"].lower() == 'global':
                out = net(inp.x.float(), inp.edge_index, inp.batch).view(-1)

            elif config["pooling"]["method"].lower() == 'selective':
                base = get_indices_batch(_batch.nodes, config["pooling"]["base_nodes_list"])
                out = net(inp.x.float(), inp.edge_index, inp.batch, base).view(-1)

            loss = loss_function(out, lbl)
            loss_batch_list_valid.append(loss.item())
            batch_list.append(out.shape[0])

            out_list.extend(out.tolist())
            lbl_list.extend(lbl.tolist())

    this_epoch_loss_valid = sum([loss_batch_list_valid[_i] * batch_list[_i]
                                 for _i in range(len(batch_list))]) / sum(batch_list)
    print('validation loss for this epoch: {}'.format(this_epoch_loss_valid))
    loss_valid.update(this_epoch_loss_valid)
    writer.add_scalar('loss/valid', loss_valid.val, loss_valid.count)

    writer.add_pr_curve('pr_curve/valid', labels=torch.tensor(lbl_list), predictions=torch.tensor(out_list),
                        global_step=i_epoch)

    metrics_dict = get_metrics(torch.tensor(out_list).cpu().numpy(), torch.tensor(lbl_list).cpu().numpy(), metrics_list)
    for k in metrics_list:
        metrics_valid[k].update(metrics_dict[k])
        writer.add_scalar(k + '/valid', metrics_valid[k].val, metrics_valid[k].count)
        metrics_dict_list_valid[k].append(metrics_dict[k])
    metrics_dict_list_valid['loss'].append(this_epoch_loss_valid)

    if best_epoch_loss_valid > this_epoch_loss_valid:
        best_epoch_loss_valid = this_epoch_loss_valid
        best_arg_epoch = i_epoch
        print('Saving the model for this epoch, as the best available..')
        torch.save(net.state_dict(), os.path.join(writer.log_dir, 'net.pth'))

    if i_epoch - best_arg_epoch >= 20:  # patience
        break

hparams_dict = {
    'lr': config["global"]["lr"],
    'batch': config["global"]["batch"],
    'weight_decay': config["global"]["weight_decay"],
    'dropout': config["network"]["dropout"],
}

writer.add_hparams(hparam_dict=hparams_dict,
                   metric_dict={
                       'best_loss_valid': min(metrics_dict_list_valid['loss']),
                       'best_loss_valid_epoch': int(torch.tensor(metrics_dict_list_valid['loss']).argmin().item()),
                       'best_f1_valid': max(metrics_dict_list_valid['f1_macro']),
                       'best_f1_valid_epoch': int(torch.tensor(metrics_dict_list_valid['f1_macro']).argmax().item()),
                   })

writer.close()
