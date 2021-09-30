import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse


import sys, os, json, pickle, shutil
sys.path.insert(0, '../../')

from ct.data import ROOT_DIR
from ct.data.flat import DatasetFeatureArray, collate_fn

from ct.networks.flat import FMLP
from ct.utils import AverageMeter, loss_function, get_metrics, metrics_list


parser = argparse.ArgumentParser()
parser.add_argument('-c', default='./config_train.json')
config_path = parser.parse_args().c

with open(config_path, 'r') as json_file:
    config = json.load(json_file)

source_dir = os.path.join(ROOT_DIR, config["data"]["source_dir"])
split_path_train = os.path.join(ROOT_DIR, config["data"]["train_path"])
split_path_valid = os.path.join(ROOT_DIR, config["data"]["valid_path"])


features_key = config["features"]["key"]    # whether a blob of text (BasicToText), or with 9 fields separated
features_type = config["features"]["type"]  # whether BOW-based or LM-based


if "bow" in features_type.lower():
    feature_tag = 'BOW'
    dim = config["features"]["init"]["proj_dim"]
    nnz_proj = config["features"]["init"]["nnz_proj"]
    ngram_low = config["features"]["init"]["ngram_low"]
    ngram_high = config["features"]["init"]["ngram_high"]
    max_features = config["features"]["init"]["max_features"]

    h5_path_train = os.path.join('../../storage', 'train_BOW_dim{}_nnzp{}_l{}_h{}_max{}.h5'.format(
        dim, nnz_proj, ngram_low, ngram_high, max_features))

    h5_path_valid = os.path.join('../../storage', 'valid_BOW_dim{}_nnzp{}_l{}_h{}_max{}.h5'.format(
        dim, nnz_proj, ngram_low, ngram_high, max_features))


elif "lm" in features_type.lower():
    feature_tag = 'LM'
    model_name = config["features"]["init"]["model_name"]
    tokenizer_name = config["features"]["init"]["tokenizer_name"]
    max_len = config["features"]["init"]["max_len"]

    h5_path_train = os.path.join('./storage', 'train_LM_max-length{}.h5'.format(512))
    h5_path_valid = os.path.join('./storage', 'valid_LM_max-length{}.h5'.format(512))


writer = SummaryWriter(comment='_{}_train'.format(feature_tag))
shutil.copy(config_path, os.path.join(writer.log_dir, 'used_config.json'))

print('Loading the data from arrayed source in h5 format from {}'.format(h5_path_train))


dataset_train = DatasetFeatureArray(source_h5_path=h5_path_train, split_label_path=split_path_train)
dataset_valid = DatasetFeatureArray(source_h5_path=h5_path_valid, split_label_path=split_path_valid)


dataloader_train = DataLoader(dataset_train, batch_size=config["global"]["batch"],
                              shuffle=True, num_workers=config["global"]["workers"], collate_fn=collate_fn)
dataloader_valid = DataLoader(dataset_valid, batch_size=config["global"]["batch"],
                              shuffle=False, num_workers=config["global"]["workers"], collate_fn=collate_fn)


#  Initialize the model:
net = FMLP(d_in=config["network"]["d_in"], c_in=config["network"]["c_in"], dropout=config["network"]["dropout"])
device = torch.device(config["global"]["device"])
net.to(device)


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
    print(' **************** training at epoch ', i_epoch + 1, ' **********************')

    net.train()
    out_list = []
    lbl_list = []
    for i_batch, _batch in enumerate(dataloader_train):
        inp = _batch['features'].float().to(device)
        lbl = _batch['labels'].float().to(device)
        out = net(inp)
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
            inp = _batch['features'].float().to(device)
            lbl = _batch['labels'].float().to(device)
            out = net(inp)
            loss = loss_function(out, lbl)
            loss_batch_list_valid.append(loss.item())
            batch_list.append(inp.shape[0])

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
