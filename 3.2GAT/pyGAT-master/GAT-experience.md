# GAT-PyTorch实验解析



### 一、实验预处理

```python
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

```

`load_data()`的参数是用来拼接数据集的字符串。

**第5-6行：**`idx_features_labels`用来存储读取`data/cora/cora.content`文件中的内容。这里涉及`numpy`中的方法`genfromtxt()`，关于该方法如何使用可参考[1,2]，可以快速将txt中保存的文本内容转换成`numpy`中的数组。文件数据的格式为`id features labels`，因此`idx_features_labels[:, 0]`、`idx_features_labels[:, 1:-1]`、`idx_features_labels[:, -1]`分别代表以上三部分。

**第7行：**`idx_features_labels[:, 1:-1]`表示节点本身附带的特征，由于这些特征构成的矩阵是稀疏矩阵，使用`csr_matrix()`方法将其存储为稀疏矩阵。关于该方法的使用参考[3,4]。

**第8行：**`labels`在文件中是用字符串表示的某个论文所属的类别，如“reinforce learning”，要将其表示成one-hot向量的形式，用到了`encode_onehot()`的方法。具体来看：

```python
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
```

先将所有由字符串表示的标签数组用`set`保存，`set`的重要特征就是元素没有重复，因此表示成`set`后可以直接得到所有标签的总数，随后为每个标签分配一个编号，创建一个单位矩阵，单位矩阵的每一行对应一个one-hot向量，也就是`np.identity(len(classes))[i, :]`，再将每个数据对应的标签表示成的one-hot向量，类型为`numpy`数组。

**第11行：**将所有节点的`id`表示成`numpy`数组。

**第12行：**由于文件中节点并非是按顺序排列的，因此建立一个编号为`0-(node_size-1)`的哈希表`idx_map`，哈希表中每一项为`id: number`，即节点`id`对应的编号为`number`。

**第13-14行：**`edges_unordered`为直接从边表文件中直接读取的结果，是一个`(edge_num, 2)`的数组，每一行表示一条边两个端点的`idx`。

**第15-16行：**上边的`edges_unordered`中存储的是端点`id`，要将每一项的`id`换成编号。在`idx_map`中以`idx`作为键查找得到对应节点的编号，`reshape`成与`edges_unordered`形状一样的数组。

**第17-19行：**首先要明确`coo_matrix()`的作用，该方法是构建一个矩阵，根据给出的下标、数据和形状，构建一个矩阵，其中下标位置的值对应数据中的值，使用方法见[5]。所以这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，所以先创建一个长度为`edge_num`的全1数组，每个1的填充位置就是一条边中两个端点的编号，即`edges[:, 0], edges[:, 1]`，矩阵的形状为`(node_size, node_size)`。

**第22行：**对于无向图，邻接矩阵是对称的。上一步得到的`adj`是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵。

**第24-25行：**分别对特征矩阵`features`和邻接矩阵`adj`做标准化，用到了`normalize()`方法：

```python
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
```

首先对每一行求和得到`rowsum`；求倒数得到`r_inv`；如果某一行全为0，则`r_inv`算出来会等于无穷大，将这些行的`r_inv`置为0；构建对角元素为`r_inv`的对角矩阵；用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的`r_inv`相乘。

**第27-37行：**分别构建训练集、验证集、测试集，并创建特征矩阵、标签向量和邻接矩阵的`tensor`，用来做模型的输入。



### 二、搭建模型

```python
model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads,   # 头的数量
                alpha=args.alpha)
```

models.py中定义了模型

```python
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
```

GraphAttentionLayer层定义如下：

```python
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
```

### 三、定义损失函数

自定义优化器

```python
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)
```

`weight_decay`表示正则化的系数

```python
loss_train = F.nll_loss(output[idx_train], labels[idx_train])
```

### 四、训练与测试

```python
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data[0]),
          "accuracy= {:.4f}".format(acc_test.data[0]))
```

训练与测试则比较常规，注意一下PyTorch的顺序，先将model置为训练状态；梯度清零；将输入送到模型得到输出结果；计算损失与准确率；反向传播求梯度更新参数。















参考：

1. [pyGAT](https://github.com/Diego999/pyGAT)
2. [pygcn代码解析](https://zhuanlan.zhihu.com/p/78191258)
3. 