import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import math

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool = True

class LayerNorm(nn.Module):
    """LayerNorm but with optional bias (nanoGPT style)"""
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )


    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2) #(B,T,3C) ->(B,T,C)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )

        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
    
        




class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x



class Block(nn.Module):
    """Pre-LN Transformer Block"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """你的 GPT（阶段A：结构对齐 nanoGPT，但内部Attention还没换成QKV合并版）"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 1) 先初始化
        self.apply(self._init_weights)

        # 2) 再对所有 residual projection 做缩放（attn.c_proj + mlp.c_proj 都会命中）
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # 3) 最后再 weight tying（避免同一参数被 init 两次）
        self.lm_head.weight = self.transformer.wte.weight


        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas: tuple, device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        use_fused = False
        if device_type == "cuda":
            try:
                import inspect
                use_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
            except Exception:
                use_fused = False
        
        extra_args = {"fused": True} if use_fused else {}
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"

        device = idx.device
        # tok_emb = self.wte(idx)  # (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        # pos_emb = self.wpe(pos)  # (T, n_embd)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)  # (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x)
        # x = self.blocks(x)
        x = self.transformer.ln_f(x)
        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None






        # 如果没有提供 targets（比如推理 / 生成阶段），就不计算 loss
        else:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                # logits 原始形状: [B, T, C]
                # B = batch size
                # T = 序列长度 (token 数)
                # C = vocab size / 类别数
                #
                # view(-1, logits.size(-1)) 的作用是：
                # 把前两维 (B, T) 展平成一维 (B*T)
                # 每一行对应“一个 token 的分类结果”
                #
                # 变换后 logits 形状: [B*T, C]
                logits.view(-1, logits.size(-1)),

                # targets 原始形状: [B, T]
                # 每个 token 对应一个正确的词 id
                #
                # 同样展平成一维，和 logits 一一对齐
                #
                # 变换后 targets 形状: [B*T]
                targets.view(-1),
            )

        # 返回：
        # logits: 仍然是 [B, T, C]（用于生成或后续计算）
        # loss:   一个标量（训练时用；推理时为 None）
        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]              # (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == "__main__":
    cfg = GPTConfig(block_size=16, vocab_size=65, n_layer=2, n_head=4, n_embd=32, dropout=0.1, bias=True)
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, 32)
    y = attn(x)
    print("y:", y.shape)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    print("logits:", logits.shape, "loss:", loss.item())
    opt = model.configure_optimizers(0.1, 3e-4, (0.9, 0.95), "cpu")
    print("optimizer ok:", type(opt) )