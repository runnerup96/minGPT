import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN
from mingpt.model import SelfAttention, NewGELU

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for encoder-decoder models.
    Queries come from the decoder hidden state; keys and values come from
    the encoder output. No causal masking — the decoder attends to all
    encoder positions freely.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.q_proj  = nn.Linear(config.n_embd, config.n_embd)
        self.kv_proj = nn.Linear(config.n_embd, 2 * config.n_embd)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout  = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, enc_out):
        # x:       decoder hidden state  (B, T_dec, C)
        # enc_out: encoder output        (B, T_enc, C)
        B, T, C = x.size()
        T_enc = enc_out.size(1)

        q    = self.q_proj(x)
        k, v = self.kv_proj(enc_out).split(self.n_embd, dim=2)
        q = q.view(B, T,     self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_dec, hs)
        k = k.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_enc, hs)
        v = v.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T_enc, hs)

        att = None # CODE HERE
        att = None # CODE HERE
        att = self.attn_dropout(att)
        y = None # CODE HERE                                            # (B, nh, T_dec, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class EncoderBlock(nn.Module):
    """Transformer encoder block with bidirectional (non-causal) self-attention."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = None # CODE HERE
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            fully_connected = nn.Linear(config.n_embd, 4 * config.n_embd),
            projection      = nn.Linear(4 * config.n_embd, config.n_embd),
            activation      = NewGELU(),
            dropout         = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.projection(m.activation(m.fully_connected(x))))

    def forward(self, x):
        # CODE HERE
        pass


class DecoderBlock(nn.Module):
    """Transformer decoder block: causal self-attention -> cross-attention -> MLP."""

    def __init__(self, config):
        super().__init__()
        self.ln_1       = nn.LayerNorm(config.n_embd)
        self.self_attn  = None # CODE HERE
        self.ln_2       = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3       = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            fully_connected = nn.Linear(config.n_embd, 4 * config.n_embd),
            projection      = nn.Linear(4 * config.n_embd, config.n_embd),
            activation      = NewGELU(),
            dropout         = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.projection(m.activation(m.fully_connected(x))))

    def forward(self, x, enc_out):
        # CODE HERE
        pass


class EncoderDecoderGPT(nn.Module):
    """
    Encoder-Decoder GPT.

    Encoder: stack of bidirectional self-attention blocks that reads the full
             source sequence and produces context vectors.
    Decoder: stack of causal self-attention + cross-attention blocks that
             generates the target sequence with teacher forcing during training.

    forward(src_idx, tgt_idx, targets=None)
        src_idx : (B, T_src)  encoder input token indices
        tgt_idx : (B, T_tgt)  decoder input token indices (teacher-forced, shifted right)
        targets : (B, T_tgt)  ground-truth labels for loss (use -1 to ignore a position)
    """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given
        C.model_type = 'gpt-nano'
        C.n_layer    = None
        C.n_head     = None
        C.n_embd     = None
        # must be set externally
        C.vocab_size         = None
        C.encoder_block_size = None  # max source sequence length
        C.decoder_block_size = None  # max target sequence length
        # dropout
        C.embd_pdrop  = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop  = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size         is not None
        assert config.encoder_block_size is not None
        assert config.decoder_block_size is not None
        self.encoder_block_size = config.encoder_block_size
        self.decoder_block_size = config.decoder_block_size

        type_given   = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given
        if type_given:
            config.merge_from_dict({
                'openai-gpt':  dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
                'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
                'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
                'gopher-44m':  dict(n_layer=8,  n_head=16, n_embd=512),
                'gpt-mini':    dict(n_layer=6,  n_head=6,  n_embd=192),
                'gpt-micro':   dict(n_layer=4,  n_head=4,  n_embd=128),
                'gpt-nano':    dict(n_layer=3,  n_head=3,  n_embd=48),
            }[config.model_type])

        # Encoder blocks use non-causal attention (no mask), block_size only
        # matters for position embeddings.
        enc_cfg = CN()
        enc_cfg.n_embd      = config.n_embd
        enc_cfg.n_head      = config.n_head
        enc_cfg.attn_pdrop  = config.attn_pdrop
        enc_cfg.resid_pdrop = config.resid_pdrop
        enc_cfg.block_size  = config.encoder_block_size

        # Decoder blocks use causal attention; block_size is needed for the mask.
        dec_cfg = CN()
        dec_cfg.n_embd      = config.n_embd
        dec_cfg.n_head      = config.n_head
        dec_cfg.attn_pdrop  = config.attn_pdrop
        dec_cfg.resid_pdrop = config.resid_pdrop
        dec_cfg.block_size  = config.decoder_block_size

        self.encoder = None # CODE HERE
        self.decoder = None # CODE HERE
        self.lm_head = None # CODE HERE

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def encode(self, src_idx):
        device = src_idx.device
        b, t = src_idx.size()
        assert t <= self.encoder_block_size, \
            f"Encoder input length {t} exceeds encoder_block_size {self.encoder_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        x = None # CODE HERE
        for block in self.encoder.h:
            # CODE HERE
            pass
        return self.encoder.ln_f(x)  # (B, T_src, n_embd)

    def decode(self, tgt_idx, enc_out):
        device = tgt_idx.device
        b, t = tgt_idx.size()
        assert t <= self.decoder_block_size, \
            f"Decoder input length {t} exceeds decoder_block_size {self.decoder_block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        x = None # CODE HERE
        for block in self.decoder.h:
            # CODE HERE
            pass
        x = self.decoder.ln_f(x)
        return self.lm_head(x)  # (B, T_tgt, vocab_size)

    def forward(self, src_idx, tgt_idx, targets=None):
        # CODE HERE
        pass

    def configure_optimizers(self, train_config):
        decay, no_decay = set(), set()
        whitelist = (nn.Linear,)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        assert not (decay & no_decay)
        assert not (param_dict.keys() - (decay | no_decay))
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)],    "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

    @torch.no_grad()
    def generate(self, src_idx, max_new_tokens, bos_token=0, temperature=1.0, do_sample=False, top_k=None):
        """
        Autoregressively decode target tokens given a source sequence.
        Encodes src_idx once, then appends one predicted token at a time starting
        from bos_token. Returns the generated tokens (bos_token stripped).
        """
        enc_out = None # CODE HERE
        tgt = torch.full((src_idx.size(0), 1), bos_token, dtype=torch.long, device=src_idx.device)
        for _ in range(max_new_tokens):
            tgt_cond = tgt[:, -self.decoder_block_size:]
            logits   = self.decode(tgt_cond, enc_out)
            logits   = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) if do_sample \
                       else torch.topk(probs, k=1, dim=-1)[1]
            tgt = None # CODE HERE
        return tgt[:, 1:]  # strip the leading bos_token
