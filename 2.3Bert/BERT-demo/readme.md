# BERT-PyTorch的简单实现

### 准备数据集

```python
import re
import math
import torch
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

text = (
    'Hello, how are you? I am Romeo.\n' # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n' # J
    'Nice meet you too. How are you today?\n' # R
    'Great. My baseball team won the competition.\n' # J
    'Oh Congratulations, Juliet\n' # R
    'Thank you Romeo\n' # J
    'Where are you going today?\n' # R
    'I am going shopping. What about you?\n' # J
    'I am going to visit my grandmother. she is not very well' # R
)
sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n') # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split())) # ['hello', 'how', 'are', 'you',...]
word2idx = {'[PAD]' : 0, '[CLS]' : 1, '[SEP]' : 2, '[MASK]' : 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx) # 总长度

token_list = list()
for sentence in sentences:
    arr = [word2idx[s] for s in sentence.split()]
    token_list.append(arr)
```

数据集采用了两个人的对话

首先通过正则表达式，将text的标点符号去掉，全都换成小写，每个句子之间分开。

需要构建一个索引和单词转换表，其中`[PAD]`是补齐填充，`[CLS]`是句子开始位，`[SEP]`是两个句子之间分隔位，`[MASK]`是遮挡的位置。将这四个特殊的键值单独插在转换表中。

```
index2word:{0: '[PAD]', 1: '[CLS]', 2: '[SEP]', 3: '[MASK]', 4: 'nice', 5: 'going', 6: 'to', 7: 'about', 8: 'meet', 9: 'she', 10: 'competition', 11: 'i', 12: 'shopping', 13: 'oh', 14: 'am', 15: 'are', 16: 'romeo', 17: 'team', 18: 'very', 19: 'what', 20: 'where', 21: 'visit', 22: 'grandmother', 23: 'you', 24: 'how', 25: 'is', 26: 'too', 27: 'well', 28: 'not', 29: 'great', 30: 'today', 31: 'baseball', 32: 'the', 33: 'thank', 34: 'my', 35: 'name', 36: 'juliet', 37: 'won', 38: 'congratulations', 39: 'hello'}
token_list:[[39, 24, 15, 23, 11, 14, 16], [39, 16, 34, 35, 25, 36, 4, 6, 8, 23], [4, 8, 23, 26, 24, 15, 23, 30], [29, 34, 31, 17, 37, 32, 10], [13, 38, 36], [33, 23, 16], [20, 15, 23, 5, 30], [11, 14, 5, 12, 19, 7, 23], [11, 14, 5, 6, 21, 34, 22, 9, 25, 28, 18, 27]]
```

原文中的九句话，被按照单词表转化了。

### 模型参数

```
# BERT Parameters
maxlen = 30 # 每句话最长的长度，不足则补0
batch_size = 6 
max_pred = 5 # 每次被masked的最多的数量
n_layers = 6 # 表示 Encoder Layer 的数量
n_heads = 12 # self-attention多头的数量
d_model = 768 # 
d_ff = 768*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V，表示 Token Embeddings、Segment Embeddings、Position Embeddings 的维度
n_segments = 2 # Decoder input 由几句话组成
```

### 数据集预处理

BERT随机mask 15%的词，其中80%替换为`[MASK]`,10%替换为随机任意一个词，10%不进行替换

```python
# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = [] # 目的是返回batch
    positive = negative = 0 
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences)) # 随机从句子中抽取两个句子索引
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']] # 加上特殊字符和分隔符
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1) 
		
        # MASK LM
        n_pred =  min(max_pred, max(1, int(len(input_ids) * 0.15))) # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']] # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]'] # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1) # random index in vocabulary
                while index < 4: # can't involve 'CLS', 'SEP', 'PAD'
                  index = randint(0, vocab_size - 1)
                input_ids[pos] = index # replace

        # Zero Paddings
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # Zero Padding，如果一个句子没有超过最长mask的数量，需要将补齐，否则每个句子mask的数量不一样
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
            negative += 1
    return batch
```

`positive`变量代表两句话是连续的个数，`negative`代表两句话不是连续的个数，我们需要做到在一个 batch 中，这两个样本的比例为 1:1。随机选取的两句话是否连续，只要通过判断`tokens_a_index + 1 == tokens_b_index`即可。

然后是随机 mask 一些 token，`n_pred`变量代表的是即将 mask 的 token 数量，`cand_maked_pos`代表的是有哪些位置是候选的、可以 mask 的（因为像 [SEP]，[CLS] 这些不能做 mask，没有意义），最后`shuffle()`一下，然后根据`random()`的值选择是替换为`[MASK]`还是替换为其它的 token

接下来会做两个 Zero Padding，第一个是为了补齐句子的长度，使得一个 batch 中的句子都是相同长度。第二个是为了补齐 mask 的数量，因为不同句子长度，会导致不同数量的单词进行 mask，我们需要保证同一个 batch 中，mask 的数量（必须）是相同的，所以也需要在后面补一些没有意义的东西，比方说`[0]`

### 构建模型

采用了 Transformer 的 Encoder

```python
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_len.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual) # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0]) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2] predict isNext

        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model) # [batch_size, maxlen, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, maxlen, d_model]
        h_masked = self.activ2(self.linear(h_masked)) # [batch_size, maxlen, d_model]
        logits_lm = self.fc2(h_masked) # [batch_size, maxlen, vocab_size]
        return logits_lm, logits_clsf
model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters(), lr=0.001)
```

### 训练

```python
for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
      logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
      loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1)) # for masked LM
      loss_lm = (loss_lm.float()).mean()
      loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
      loss = loss_lm + loss_clsf
      if (epoch + 1) % 10 == 0:
          print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
```

输出：

```
Epoch: 0010 loss = 1.455598
Epoch: 0020 loss = 1.071564
Epoch: 0030 loss = 0.895903
Epoch: 0040 loss = 0.842477
Epoch: 0050 loss = 0.862473
```



### 测试

```python
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
print(text)
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), \
                 torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)
```

输出

```
Hello, how are you? I am Romeo.
Hello, Romeo My name is Juliet. Nice to meet you.
Nice meet you too. How are you today?
Great. My baseball team won the competition.
Oh Congratulations, Juliet
Thank you Romeo
Where are you going today?
I am going shopping. What about you?
I am going to visit my grandmother. she is not very well
================================
['[CLS]', 'thank', 'you', 'romeo', '[SEP]', 'where', 'are', '[MASK]', 'going', 'today', '[SEP]']
masked tokens list :  [23]
predict masked tokens list :  [23]
isNext :  True
predict isNext :  True
```

取batch[0]中的两个句子，是分隔符的第一行。

然后mask的词是转换表的第23个词，预测出来的结果一致。

两个句子是相邻的关系，预测的结果也一直。











