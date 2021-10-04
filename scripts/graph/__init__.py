import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_sort_pool


class BOWFeaturizer:
    def __init__(self, bow_obj):
        self.bow_obj = bow_obj

    def __call__(self, raw):
        feat = self.bow_obj.transform_project([raw]).reshape(-1)
        return feat


class GCNGlobal(nn.Module):
    def __init__(self, config_network):
        super(GCNGlobal, self).__init__()
        # torch.manual_seed(12345)

        d_in = config_network["d_in"]
        d_h = config_network["d_h"]
        self.dropout = config_network["dropout"]
        self.sigm = config_network["sigm"]

        self.conv1 = GCNConv(d_in, d_h)
        self.conv2 = GCNConv(d_h, d_h)
        self.conv3 = GCNConv(d_h, d_h)
        self.lin = nn.Linear(d_h, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        # x = global_sort_pool(x, batch, k=5)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        if self.sigm:
            x = torch.sigmoid(x)

        return x


class GATGlobal(nn.Module):
    def __init__(self, config_network):
        super(GATGlobal, self).__init__()
        # torch.manual_seed(12345)

        d_in = config_network["d_in"]
        d_h = config_network["d_h"]
        self.dropout = config_network["dropout"]
        self.sigm = config_network["sigm"]

        self.conv1 = GATConv(d_in, d_h)
        self.conv2 = GATConv(d_h, d_h)
        self.conv3 = GATConv(d_h, d_h)
        self.lin = nn.Linear(d_h, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        # x = global_sort_pool(x, batch, k=5)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin(x)
        if self.sigm:
            x = torch.sigmoid(x)

        return x



class GCNSelective(nn.Module):
    def __init__(self, config_network, len_base_nodes_list):
        super(GCNSelective, self).__init__()
        # torch.manual_seed(12345)

        self.d_latent = 20
        self.len_base_nodes_list = len_base_nodes_list

        d_in = config_network["d_in"]
        self.d_h = config_network["d_h"]
        self.dropout = config_network["dropout"]
        self.sigm = config_network["sigm"]

        self.conv1 = GCNConv(d_in, self.d_h)
        self.conv2 = GCNConv(self.d_h, self.d_h)
        self.conv3 = GCNConv(self.d_h, self.d_h)
        self.conv4 = GCNConv(self.d_h, self.d_h)
        self.conv5 = GCNConv(self.d_h, self.d_latent)
        self.lin1 = nn.Linear(self.d_latent * (1+self.len_base_nodes_list), self.d_h)
        self.lin2 = nn.Linear(self.d_h, 1)

        self.register_buffer('latent_base', None, persistent=False)
        self.register_buffer('latent_global', None, persistent=False)

    def forward(self, x, edge_index, batch, base):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()

        x = self.conv3(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()

        x = self.conv4(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()

        x = self.conv5(x, edge_index)

        x_g = global_mean_pool(x, batch)
        x_s = selective_pool(x, batch, base)
        x = torch.cat((x_s, x_g), dim=1)
        self.register_buffer('latent_base', x.view(x.shape[0], 1+self.len_base_nodes_list, self.d_latent),
                             persistent=False)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin1(x)

        self.register_buffer('latent_global', x, persistent=False)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.lin2(x)

        if self.sigm:
            x = torch.sigmoid(x)

        return x



class GATSelective(nn.Module):
    def __init__(self, config_network, len_base_nodes_list):
        super(GATSelective, self).__init__()
        # torch.manual_seed(12345)

        self.d_latent = 20
        self.len_base_nodes_list = len_base_nodes_list

        d_in = config_network["d_in"]
        self.d_h = config_network["d_h"]
        self.dropout = config_network["dropout"]
        self.sigm = config_network["sigm"]

        self.conv1 = GATConv(d_in, self.d_h)
        self.conv2 = GATConv(self.d_h, self.d_h)
        self.conv3 = GATConv(self.d_h, self.d_latent)
        self.lin1 = nn.Linear(self.d_latent * (1+self.len_base_nodes_list), self.d_h)
        self.lin2 = nn.Linear(self.d_h, 1)

        self.register_buffer('latent_base', None, persistent=False)
        self.register_buffer('latent_global', None, persistent=False)

    def forward(self, x, edge_index, batch, base):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.conv3(x, edge_index)

        x_g = global_mean_pool(x, batch)
        x_s = selective_pool(x, batch, base)
        x = torch.cat((x_s, x_g), dim=1)
        self.register_buffer('latent_base', x.view(x.shape[0], 1+self.len_base_nodes_list, self.d_latent),
                             persistent=False)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin1(x)

        self.register_buffer('latent_global', x, persistent=False)

        x = F.dropout(x, self.dropout, training=self.training)
        x = x.relu()
        x = self.lin2(x)

        if self.sigm:
            x = torch.sigmoid(x)

        return x


def get_indices(nodes, base_list):
    return list(map(lambda n: nodes.index(n) if n in nodes else None, base_list.copy()))


def get_indices_batch(nodes_batch, base_list):
    return list(map(lambda nodes: get_indices(nodes, base_list), nodes_batch))


def selective_pool(inp_x, inp_batch, base):
    b, d = len(base), inp_x.shape[1]
    x = torch.zeros(b, len(base[0]), d, device=inp_x.device, dtype=inp_x.dtype)
    for i_b in range(b):
        _ind = [base[i_b].index(_i) for _i in base[i_b] if _i]
        x[i_b, _ind, :] = inp_x[inp_batch == i_b][[base[i_b][_i] for _i in _ind], :]

    return x.view(b, -1)
