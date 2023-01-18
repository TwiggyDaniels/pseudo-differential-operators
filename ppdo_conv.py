import torch
from torch.nn import Linear
from torch_geometric.nn import MessagePassing

class PPDOConv2D(MessagePassing):
    def __init__(self, in_channels, out_channels, norm_factor=None):
        # mean propagation per the original paper
        super().__init__(aggr='mean')
        self.lin = Linear(in_channels*3, out_channels)

    def forward(self, x, edge_index, pos):
        x = self.propagate(edge_index, x=x, pos=pos)
        x = self.lin(x)
        return x

    def message(self, x_i, x_j, pos_i, pos_j):
        # vector subtraction to find the partial feature differences
        feat_diff = x_i - x_j
        # vector subtraction of the node positions
        pos_diff = pos_i - pos_j

        # no need to take root since we square it when we use it anyways
        dist = torch.sum(pos_diff, dim=-1)

        # I want to multiply each row of the feature matrix by the 
        # corresponding x & y row of the positional differences...
        # There MUST be an easier way to do this...
        # I could also unsqueeze but that basically does the same
        # thing as what I am already doing... Damn there must be
        # a one-line way to do this...
        part_x = pos_diff[:,0][:,None] * feat_diff
        part_y = pos_diff[:,1][:,None] * feat_diff

        # Same as the last bigass comment
        part_x /= dist[:, None]
        part_y /= dist[:, None]

        return torch.cat([part_x, part_y, x_i], dim=-1)

    def update(self, aggr_out):
        return aggr_out
