import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

class PPDOConv2D(MessagePassing):
    def __init__(self, in_channels, out_channels, norm_factor=None):
        # mean propagation per the original paper
        super().__init__(aggr='mean')
        self.lin = Linear(, out_channels)


    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    def message(self, x_i, x_j, pos_i, pos_j):
        # vector subtraction to find the partial feature differences
        feat_diff = x_i - x_j
        # vector subtraction of the node positions
        pos_diff = pos_i - pos_j

        # no need to take root since we square it when we use it anyways
        dist = torch.sum(pos_diff, dim=-1)

        msg = torch.squeeze((pos_diff * feat_diff ) / dist), -2)

        msg = torch.cat([msg, x_i], dim=1)

        return msg

    def update(self, aggr_out):
        return aggr_out
