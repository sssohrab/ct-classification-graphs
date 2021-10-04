import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse


import sys, os, json, pickle, shutil
sys.path.insert(0, '../../')


from ct.data import ROOT_DIR, FeaturizerChannelHuggingFace
from ct.data.flat import DatasetFeatureText, collate_fn

from ct.networks.flat import FMLP
from ct.utils import AverageMeter, loss_function, get_metrics, metrics_list


parser = argparse.ArgumentParser()

parser.add_argument('-c', default='./config_eval.json')

config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

with open(os.path.join('./runs', config["train_run"], 'used_config.json'), 'r') as _json_file:
    _config = json.load(_json_file)
    config["features"] = _config["features"]
    config["network"] = _config["network"]


device = torch.device(config["global"]["device"])

source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
split_path_test = os.path.join(ROOT_DIR, config["data"]["test_path"])

features_key = config["features"]["key"]
features_type = config["features"]["type"]

if "bow" in features_type.lower():
    feature_tag = 'BOW'
    dim = config["features"]["init"]["proj_dim"]
    nnz_proj = config["features"]["init"]["nnz_proj"]
    ngram_low = config["features"]["init"]["ngram_low"]
    ngram_high = config["features"]["init"]["ngram_high"]
    max_features = config["features"]["init"]["max_features"]

    featurizer_obj_path = os.path.join('../../storage',
                                       'bow_obj_dim{}_nnzp{}_l{}_h{}_max{}.pkl'.format(dim, nnz_proj, ngram_low,
                                                                                       ngram_high, max_features))

    with open(featurizer_obj_path, 'rb') as file:
        featurizer_obj = pickle.load(file)

elif "lm" in features_type.lower():
    feature_tag = 'LM'
    model_name = config["features"]["init"]["model_name"]
    tokenizer_name = config["features"]["init"]["tokenizer_name"]
    max_len = config["features"]["init"]["max_len"]

    featurizer_obj = FeaturizerChannelHuggingFace(model_name=model_name, tokenizer_name=tokenizer_name, device=device,
                                                  truncation=True, max_length=max_len)


writer = SummaryWriter(comment='_{}_eval'.format(feature_tag))
shutil.copy(config_path, os.path.join(writer.log_dir, 'used_config.json'))

print('Loading the data from text with online featurizing..')


dataset_test = DatasetFeatureText(featurizer_obj, source_dir, split_label_path=split_path_test, ext='.json',
                                  features_key=features_key)


dataloader_test = DataLoader(dataset_test, batch_size=config["global"]["batch"],
                             shuffle=False, num_workers=config["global"]["workers"], collate_fn=collate_fn)


#  Initialize the model:
state_dict_path = os.path.join('./runs', config["train_run"], 'net.pth')
net = FMLP(d_in=config["network"]["d_in"], c_in=config["network"]["c_in"], dropout=config["network"]["dropout"])

net.load_state_dict(torch.load(state_dict_path, map_location=device))
net.to(device)
net.eval()


metrics_test = {k: AverageMeter(k) for k in metrics_list}

loss_test = AverageMeter('loss_test')

out_list = []
lbl_list = []
docids_list = []
loss_batch_list_test = []
batch_list = []

with torch.no_grad():
    for i_batch, _batch in enumerate(dataloader_test):
        print('.. Evaluating batch {} from the test set..'.format(i_batch))
        inp = _batch['features'].float().to(device)
        lbl = _batch['labels'].float().to(device)
        docids_list.extend(_batch['docids'])
        out = net(inp)
        loss = loss_function(out, lbl)
        print('loss of this batch: {}'.format(loss.item()))
        loss_batch_list_test.append(loss.item())
        batch_list.append(inp.shape[0])

        out_list.extend(out.tolist())
        lbl_list.extend(lbl.tolist())

avg_loss_test = sum([loss_batch_list_test[_i] * batch_list[_i]
                                 for _i in range(len(batch_list))]) / sum(batch_list)
loss_test.update(avg_loss_test)

writer.add_scalar('loss/test', loss_test.val, loss_test.count)

writer.add_pr_curve('pr_curve/test', labels=torch.tensor(lbl_list), predictions=torch.tensor(out_list), num_thresholds=127)

metrics_dict = get_metrics(out.cpu().numpy(), lbl.cpu().numpy(), metrics_list)
for k in metrics_list:
    metrics_test[k].update(metrics_dict[k])
    writer.add_scalar(k + '/test', metrics_test[k].val, metrics_test[k].count)


writer.close()

with open(os.path.join(writer.log_dir, 'scores.txt'), 'w') as f:
    line = ' '.join(['NCTId', 'label', 'score']) + '\n'
    f.write(line)
    for i_docid, docid in enumerate(docids_list):
        line = ' '.join([docid, str(lbl_list[i_docid]), str(out_list[i_docid])]) + '\n'
        f.write(line)




