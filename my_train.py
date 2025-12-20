import time, math
from contextlib import nullcontext

import torch
from my_model import GPT, GPTConfig

# ----------------------------
# A) 超参（先固定写死，后续再做 argparse）
batch_size = 16
block_size = 256
max_iters = 2000
eval_interval = 500
eval_iters = 200

learning_rate = 3e-4
min_lr = 3e-5
warmup_iters = 100
lr_decay_iters = max_iters

weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

gradient_accumulation_steps = 4  # 想模拟更大 batch 就改成 4/8，并相应调小 batch_size

# ----------------------------
# B) 设备与混合精度
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device == "cuda" else "cpu"
torch.manual_seed(1337)
if device_type == "cuda":
    torch.cuda.manual_seed(1337)
print("Using device:", device)

if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


use_amp = (device_type == "cuda")
use_bf16 = use_amp and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

# nanoGPT 风格：CPU 就是空上下文；GPU 才 autocast
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=amp_dtype)
# fp16 需要 GradScaler；bf16 不需要（更稳定）
scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and (not use_bf16)))

# ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type=device_type, dtype=amp_dtype)
# scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (not use_bf16)))


# ----------------------------
# C) 数据：TinyShakespeare char-level
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda idxs: "".join(itos[i] for i in idxs)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split: str):
    data_ = train_data if split == "train" else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i+block_size] for i in ix])
    y = torch.stack([data_[i+1:i+1+block_size] for i in ix])

    if device_type == "cuda":
        # 更快的数据搬运
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ----------------------------
# D) 模型
gptconf = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
    bias=True,
)
model = GPT(gptconf).to(device)

# optimizer（用你在 my_model.py 里实现的 nanoGPT 风格分组）
optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
)

# ----------------------------
# E) lr schedule（warmup + cosine）
def get_lr(it: int):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ----------------------------
# F) eval
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ----------------------------
# G) 训练循环（带梯度累积 + AMP + clip）
X, Y = get_batch("train")  # 先取第一个 batch（nanoGPT 会这么干）
t0 = time.time()


best_val = 1e9

for it in range(max_iters):

    # set lr
    lr = get_lr(it)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # eval
    if it % eval_interval == 0:
        losses = estimate_loss()
        if it > 0 and losses["val"] < best_val:
            best_val = losses["val"]
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": it,
                "best_val": best_val,
                "config": gptconf.__dict__,
            }
            torch.save(ckpt, "ckpt.pt")
            print(f"saved ckpt.pt (best_val={best_val:.4f})")

        dt = time.time() - t0
        t0 = time.time()
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}, dt {dt:.2f}s")

    # 梯度清零
    optimizer.zero_grad(set_to_none=True)

    # 梯度累积：一个 iter 里做多个 micro step
    for micro in range(gradient_accumulation_steps):
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps

        # 预取下一个 batch（让 GPU 更忙，吞吐更高）
        X, Y = get_batch("train")

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

    # clip grad（fp16 先 unscale 再 clip）
    if grad_clip != 0.0:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # grad_norm = None
    # if grad_clip != 0.0:
    #     if scaler.is_enabled():
    #         scaler.unscale_(optimizer)

    #     # 返回的是 “裁剪前的总梯度范数”
    #     grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    #     # 每隔 N 步打印一次（你可以改成 50/100/200）
    #     if it % 100 == 0:
    #         gn = float(grad_norm.item())
    #         clipped = (gn > grad_clip)
    #         print(f"iter {it}: grad_norm={gn:.3f} clipped={clipped}")

    # optimizer step
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

# ----------------------------
# 生成看看效果
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(out))
