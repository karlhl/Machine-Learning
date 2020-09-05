import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    """
    idx_features_labels用来存储读取data/cora/cora.content文件中的内容。
    这里涉及numpy中的方法genfromtxt()，可以快速将txt中保存的文本内容转换成numpy中的数组。
    文件数据的格式为id features labels，
    因此idx_features_labels[:, 0]、idx_features_labels[:, 1:-1]、idx_features_labels[:, -1]分别代表以上三部分。
    
    idx_features_labels[:, 1:-1]表示节点本身附带的特征，由于这些特征构成的矩阵是稀疏矩阵，使用csr_matrix()方法将其存储为稀疏矩阵。
    
    labels在文件中是用字符串表示的某个论文所属的类别，如“reinforce learning”，要将其表示成one-hot向量的形式，用到了encode_onehot()的方法。
    """
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))  # 读取文件每一行
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)  # 切片左闭右开，除去第一列id，最后一列label
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)  # 将所有id表示为numpy数组
    idx_map = {j: i for i, j in enumerate(idx)} # 因为节点并不是按照顺序排列的，所以建立一个嘻哈表{文章id：排列的number}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32) # 读取的边文件的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx。
    # 将边信息的id，转为上边的序号。然后reshape成原来一样的形状
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 构建邻接矩阵adj
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵。
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)  # 同gcn的normalize
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140) #训练集
    idx_val = range(200, 500) # 验证集
    idx_test = range(500, 1500) # 测试集

    adj = torch.FloatTensor(np.array(adj.todense())) # 邻接矩阵Tensor
    features = torch.FloatTensor(np.array(features.todense())) # 特征矩阵tensor
    labels = torch.LongTensor(np.where(labels)[1]) # 标签向量tensor

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

"""
首先对每一行求和得到rowsum；求倒数得到r_inv；如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0；构建对角元素为r_inv的对角矩阵；用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘。
"""
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


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

