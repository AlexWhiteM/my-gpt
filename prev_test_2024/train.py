import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

# 创建一个ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser(description='This is a demonstration program')

# 添加一个参数解析，指定期望的类型和帮助信息等
parser.add_argument('-batch_size', type=str, required=True, help='Please provide a batch_size')

args = parser.parse_args()

# 现在我们可以在程序中使用解析的参数值
print(f'批处理大小: {args.batch_size}')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = int(args.batch_size)
block_size = 128
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_head = 1
n_layer = 1
dropout = 0.2

print(device)

# 读取vocab.txt文件，获取所有字符并创建字符到索引和索引到字符的映射
chars = ""
with open("openwebtext/vocab.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    
vocab_size = len(chars)

string_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_string = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# 使用内存映射从单个文件中获取小块文本数据
def get_random_chunk(split):
    filename = "openwebtext/train_split.txt" if split == 'train' else "openwebtext/val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # 确定文件大小并随机选择一个开始读取的位置
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # 移动到随机选择的位置并读取一块文本数据
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # 将读取的数据解码为字符串，并忽略任何无效的字节序列
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r', '')
            
            # 训练和测试数据
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
            
    return data

# 获取批处理数据
def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ 自注意力机制中的一个头 """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 输入尺寸 (batch, time-step, channels)
        # 输出尺寸 (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        # 计算注意力得分
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # 执行加权聚合
        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ 并行的多个自注意力头 """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, hs) * num_heads -> (B, T, n_embd)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ 一个简单的前馈神经网络层 """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: 包含自注意力机制和前馈神经网络 """

    def __init__(self, n_embd, n_head):
        # n_embd: 嵌入维度, n_head: 头的数量
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # 最后一层的LayerNorm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        print(index.shape)
        B, T = index.shape
        
        # index 和 targets 都是 (B, T) 的整数张量
        tok_emb = self.token_embedding_table(index) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, index, max_new_tokens):
        # index 是当前上下文中 (B, T) 的索引数组
        for _ in range(max_new_tokens):
            # 获取预测结果
            logits, loss = self.forward(index)
            # 只关注最后一个时间步
            logits = logits[:, -1, :] # 变成 (B, C)
            # 应用 softmax 获取概率
            probs = F.softmax(logits, dim=-1) # (B, C)
            # 从分布中采样
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将采样到的索引添加到运行序列中
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
# print('加载模型参数...')
# with open('model-01.pkl', 'rb') as f:
#     model = pickle.load(f)
# print('加载成功!')
m = model.to(device)

# 创建一个PyTorch优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        print(f"步数: {iter}, 训练损失: {losses['train']:.3f}, 验证损失: {losses['val']:.3f}")

    # 采样一个数据批次
    xb, yb = get_batch('train')

    # 评估损失
    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss.item())

# 保存模型
with open('model-01.pkl', 'wb') as f:
    pickle.dump(model, f)
print('模型已保存')
