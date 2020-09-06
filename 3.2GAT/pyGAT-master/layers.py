import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha # 学习因子
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Xavier初始化参数
        # 这里的self.a,对应的是论文里的向量a，故其维度大小应该为(2*out_features, 1)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)  # h:torch.Size([2708, 8]) 8是label的个数
        N = h.size()[0] # N:2708

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # print(a_input.shape) # torch.Size([2708, 2708, 16])
        # a_input是由Whi和Whj concat得到，对应论文里的Whi||Whj

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2)) # e_ij
        # squeeze除去维数为1的维度
        # [2708, 2708, 16]与[16, 1]相乘再除去维数为1的维度，故其维度为[2708,2708],与领接矩阵adj的维度一样

        zero_vec = -9e15*torch.ones_like(e)
        # 维度大小与e相同，所有元素都是-9*10的15次方


        attention = torch.where(adj > 0, e, zero_vec) # 如果邻接矩阵的元素值》0，设为e，否则zero_vec
        '''这里我们回想一下在utils.py里adj怎么建成的：两个节点有边，则为1，否则为0。
                故adj的领接矩阵的大小为[2708,2708]。(不熟的自己去复习一下图结构中的领接矩阵)。
                print(adj）这里我们看其中一个adj
                tensor([[0.1667, 0.0000, 0.0000,  ..., 0.0000, 0.0000,   0.0000],
                [0.0000, 0.5000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.2000,  ..., 0.0000, 0.0000, 0.0000],
                ...,
                [0.0000, 0.0000, 0.0000,  ..., 0.2000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.2000, 0.0000],
                [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.2500]])
                不是1而是小数是因为进行了归一化处理
                故当adj>0，即两结点有边，则用gat构建的矩阵e，若adj=0,则另其为一个很大的负数，这么做的原因是进行softmax时，这些数就会接近于0了。

                '''

        attention = F.softmax(attention, dim=1) # 对应论文公式3，attention就是公式里的αij
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

    
class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
