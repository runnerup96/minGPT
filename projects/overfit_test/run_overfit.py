"""
Single-batch overfit test for both model types.

Task: sequence reversal
  source : [a, b, c, d, e]
  target : [e, d, c, b, a]

Switch between models by setting MODEL at the top of the file:
  'enc_dec'  — EncoderDecoderGPT (encoder + decoder with cross-attention)
  'gpt'      — decoder-only GPT  (src and tgt concatenated into one sequence)

Run from the repo root:
    python projects/enc_dec_test/overfit_test.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from mingpt.utils import CfgNode as CN

# ---------------------------------------------------------------------------
# Toggle here
# ---------------------------------------------------------------------------
MODEL = 'enc_dec'   # 'enc_dec' | 'gpt'

# ---------------------------------------------------------------------------
# Fixed batch  (token 0 = BOS/pad, tokens 1-9 = digits)
# ---------------------------------------------------------------------------
BOS     = 0
SEQ_LEN = 5

srcs = torch.tensor([
    [3, 1, 4, 1, 5],
    [2, 7, 1, 8, 2],
    [6, 6, 2, 8, 3],
    [1, 4, 1, 4, 2],
], dtype=torch.long)

tgts = srcs.flip(dims=[1])   # reversed source = target

# ---------------------------------------------------------------------------
# Model + batch setup
# ---------------------------------------------------------------------------
torch.manual_seed(42)

if MODEL == 'enc_dec':
    from mingpt.enc_dec_model import EncoderDecoderGPT

    # Decoder input: BOS prepended, last target token dropped (teacher forcing)
    dec_inputs  = torch.cat([torch.full((len(srcs), 1), BOS, dtype=torch.long), tgts[:, :-1]], dim=1)
    dec_targets = tgts

    model_cfg = EncoderDecoderGPT.get_default_config()
    model_cfg.model_type         = 'gpt-nano'
    model_cfg.vocab_size         = 10
    model_cfg.encoder_block_size = SEQ_LEN
    model_cfg.decoder_block_size = SEQ_LEN
    model_cfg.embd_pdrop  = 0.0
    model_cfg.resid_pdrop = 0.0
    model_cfg.attn_pdrop  = 0.0

    model = EncoderDecoderGPT(model_cfg)

else:  # 'gpt'
    from mingpt.model import GPT

    # Concatenate [src | tgt] into one sequence of length 2*SEQ_LEN.
    # GPT sees full_seq[:-1] as input and predicts full_seq[1:].
    # Mask the source-prediction positions so loss is only on the target tokens.
    #
    #   full : [s0 s1 s2 s3 s4 | t0 t1 t2 t3 t4]
    #   x    : [s0 s1 s2 s3 s4 | t0 t1 t2 t3]      (drop last)
    #   y    : [-1 -1 -1 -1 t0 | t1 t2 t3 t4]      (mask first SEQ_LEN-1)
    #
    full = torch.cat([srcs, tgts], dim=1)           # (B, 2*SEQ_LEN)
    gpt_x = full[:, :-1]                            # (B, 2*SEQ_LEN - 1)
    gpt_y = full[:, 1:].clone()
    gpt_y[:, :SEQ_LEN - 1] = -1                    # ignore source predictions

    model_cfg = GPT.get_default_config()
    model_cfg.model_type = 'gpt-nano'
    model_cfg.vocab_size  = 10
    model_cfg.block_size  = 2 * SEQ_LEN - 1
    model_cfg.embd_pdrop  = 0.0
    model_cfg.resid_pdrop = 0.0
    model_cfg.attn_pdrop  = 0.0

    model = GPT(model_cfg)

model.train()

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
train_cfg = CN()
train_cfg.learning_rate = 1e-3
train_cfg.weight_decay  = 0.0
train_cfg.betas         = (0.9, 0.95)

optimizer = model.configure_optimizers(train_cfg)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
NUM_STEPS   = 500
PRINT_EVERY = 50

print(f"\nModel: {MODEL}")
print(f"\n{'Step':>6}  {'Loss':>10}")
print("-" * 20)

for step in range(1, NUM_STEPS + 1):
    if MODEL == 'enc_dec':
        logits, loss = model(srcs, dec_inputs, targets=dec_targets)
    else:
        logits, loss = model(gpt_x, gpt_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % PRINT_EVERY == 0 or step == 1:
        print(f"{step:>6}  {loss.item():>10.6f}")

# ---------------------------------------------------------------------------
# Evaluation — greedy decoding
# ---------------------------------------------------------------------------
model.eval()

if MODEL == 'enc_dec':
    preds = model.generate(srcs, max_new_tokens=SEQ_LEN, bos_token=BOS)
else:
    # Seed generation with the source, then take the SEQ_LEN generated tokens
    generated = model.generate(srcs, max_new_tokens=SEQ_LEN)  # (B, 2*SEQ_LEN)
    preds = generated[:, SEQ_LEN:]                            # strip the source prefix

print("\n--- Greedy decoding results ---")
print(f"{'Source':<22}  {'Expected':<22}  {'Predicted':<22}  OK?")
print("-" * 76)

all_correct = True
for i in range(len(srcs)):
    src_lst  = srcs[i].tolist()
    exp_lst  = tgts[i].tolist()
    pred_lst = preds[i].tolist()
    ok = exp_lst == pred_lst
    all_correct = all_correct and ok
    print(f"{str(src_lst):<22}  {str(exp_lst):<22}  {str(pred_lst):<22}  {'PASS' if ok else 'FAIL'}")

print()
if all_correct:
    print("All examples correct — overfit successful.")
else:
    print("Some predictions wrong — try increasing NUM_STEPS.")
