import torch
import torch.nn.functional as F
from sklearn import metrics


metrics_list = [
    'precision_micro',
    'precision_macro',
    'recall_micro',
    'recall_macro',
    'f1_micro',
    'f1_macro',
    'roc_auc',
    'pr_auc'
]


def get_metrics(output, label, metrics_list):
    metrics_dict = {m: None for m in metrics_list}
    label = label.round()

    if 'precision_micro' in metrics_list:
        metrics_dict['precision_micro'] = metrics.precision_score(y_true=label, y_pred=output.round(), average='micro')
    if 'precision_macro' in metrics_list:
        metrics_dict['precision_macro'] = metrics.precision_score(y_true=label, y_pred=output.round(), average='macro')
    if 'recall_micro' in metrics_list:
        metrics_dict['recall_micro'] = metrics.recall_score(y_true=label, y_pred=output.round(), average='micro')
    if 'recall_macro' in metrics_list:
        metrics_dict['recall_macro'] = metrics.recall_score(y_true=label, y_pred=output.round(), average='macro')
    if 'f1_micro' in metrics_list:
        metrics_dict['f1_micro'] = metrics.f1_score(y_true=label, y_pred=output.round(), average='micro')
    if 'f1_macro' in metrics_list:
        metrics_dict['f1_macro'] = metrics.f1_score(y_true=label, y_pred=output.round(), average='macro')

    if 'roc_auc' in metrics_list:
        metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true=label, y_score=output)

    if 'pr_auc' in metrics_list:
        metrics_dict['pr_auc'] = metrics.average_precision_score(y_true=label, y_score=output)

    return metrics_dict


def loss_function(output, label):

    b, n_1 = label.shape[0], label.sum()
    weight = torch.ones_like(label) * (b - n_1) * b / (2 * n_1 * (b - n_1))
    weight[label == 0] = n_1* b / (2 * n_1 * (b - n_1))
    return F.binary_cross_entropy(output, label, weight)


class AverageMeter(object):
    """
    Adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py

    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # TODO: What is this line below? Looks really good.
        return fmtstr.format(**self.__dict__)

