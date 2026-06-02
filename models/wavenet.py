"""
Hierarchical WaveNet for character-level language modeling.

This is the final model in the progression. Architecturally it is a stack of
three `(FlattenConsecutive(2), Linear, BatchNorm, Tanh)` blocks followed by
an output linear projection. Each block fuses adjacent positions in the
context window and projects through a nonlinearity, producing a hierarchical
receptive field:

    Input    (B, 8)                  raw character indices
    Embed    (B, 8, 24)              learned 24-d character vectors
    Block 1  (B, 4, 128)             bigram-level features
    Block 2  (B, 2, 128)             4-gram-level features
    Block 3  (B, 128)                full 8-character representation (squeezed)
    Output   (B, 27)                 logits

Compared to a flat MLP with the same total parameter count, this hierarchical
factorization (a) builds in the inductive bias that locality matters, and
(b) lets the model learn progressively more abstract features at each level.
The probing analysis in `notebooks/06_probing.ipynb` characterizes what
these features actually become after training.

Hyperparameters (matching Karpathy's reference implementation):
    block_size = 8     — context window. 8-character history is enough to
                         capture name-internal phonological dependencies
                         (cf. block_size=3 in the MLP, which is just 3rd-
                         order Markov).
    n_embd     = 24    — embedding dimensionality. Larger than the MLP's 10
                         to give the model room to encode the richer
                         distinctions it needs at 8 positions.
    n_hidden   = 128   — width of each hierarchical level. Constant across
                         levels — the depth, not the width, is what gives
                         the hierarchy its expressive power.

These values are intentionally kept at Karpathy's reference settings (not
tuned upward to get better numbers) so the loss table in the README is
directly comparable to the published reference.
"""

import torch
import torch.nn.functional as F

from makemore.layers import (
    Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential,
)


def build_wavenet(vocab_size, block_size=8, n_embd=24, n_hidden=128,
                  seed=2147483647):
    """Construct the hierarchical WaveNet as a Sequential pipeline.

    The architectural pattern at each level is:
        FlattenConsecutive(2) — fold adjacent pairs into the channel dim
        Linear(bias=False)    — project; bias dropped because BN follows
        BatchNorm1d           — normalize to stable activation distribution
        Tanh                  — nonlinearity (saturation diagnosed in probing)

    Replicated three times, this builds a receptive field of 8 positions
    across 3 nonlinear layers. The output linear projection at the end maps
    the final 128-d representation into the 27-d logit space.

    Initialization caveats:
    - Standard Kaiming for the hidden layers (the Linear class handles the
      `1/sqrt(fan_in)` baseline; the 5/3 gain for tanh is omitted here
      because BatchNorm corrects variance directly, making the gain less
      critical than it was in the bias-free pre-BN MLP).
    - Final layer scaled down by 0.1 to start training with near-uniform
      logits — same rationale as the MLP+BN model, applied at the WaveNet
      scale (0.1 rather than 0.01 because the final layer has fewer
      effective inputs than the MLP's hidden layer).
    """
    torch.manual_seed(seed)

    model = Sequential([
        Embedding(vocab_size, n_embd),
        # Level 1: bigram features. (B, 8, n_embd) → (B, 4, n_embd*2)
        FlattenConsecutive(2),
        Linear(n_embd * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        # Level 2: 4-gram features. (B, 4, n_hidden) → (B, 2, n_hidden*2)
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        # Level 3: 8-gram features. (B, 2, n_hidden) → (B, n_hidden*2) (squeezed)
        FlattenConsecutive(2),
        Linear(n_hidden * 2, n_hidden, bias=False),
        BatchNorm1d(n_hidden),
        Tanh(),
        # Output projection. (B, n_hidden) → (B, vocab_size)
        Linear(n_hidden, vocab_size),
    ])

    # Scale down the final logit projection so initial logits are near zero.
    with torch.no_grad():
        model.layers[-1].weight *= 0.1

    for p in model.parameters():
        p.requires_grad = True

    return model


def save_checkpoint(model, path, vocab_size, block_size, n_embd, n_hidden,
                    itos=None, stoi=None):
    """Persist model state and architecture metadata to a `.pth` file.

    We save both the parameter tensors *and* the architecture hyperparameters
    so that loading is fully self-describing — `load_checkpoint` reconstructs
    the model topology from the file alone, without the caller having to
    remember (or document) what hyperparameters the checkpoint was trained
    with. This is the difference between a checkpoint that's reproducible
    six months from now and one that requires archaeology to load.
    """
    state = {
        'parameters': [p.detach().clone() for p in model.parameters()],
        # Also save BatchNorm running stats — without these, the model is
        # untrainable for inference because BN switches to running stats
        # in eval mode and would otherwise have its zero-init buffers.
        'bn_buffers': [
            (layer.running_mean.clone(), layer.running_var.clone())
            for layer in model.layers if isinstance(layer, BatchNorm1d)
        ],
        'vocab_size': vocab_size,
        'block_size': block_size,
        'n_embd': n_embd,
        'n_hidden': n_hidden,
        'itos': itos,
        'stoi': stoi,
    }
    torch.save(state, path)


def load_checkpoint(path):
    """Restore a WaveNet from a checkpoint produced by `save_checkpoint`.

    Returns (model, metadata) where metadata is a dict containing
    `vocab_size`, `block_size`, `n_embd`, `n_hidden`, and optionally `itos`/`stoi`.
    """
    state = torch.load(path, weights_only=False)
    model = build_wavenet(
        vocab_size=state['vocab_size'],
        block_size=state['block_size'],
        n_embd=state['n_embd'],
        n_hidden=state['n_hidden'],
    )
    with torch.no_grad():
        for p, saved in zip(model.parameters(), state['parameters']):
            p.copy_(saved)
        # BatchNorm running buffers acquire a keepdim shape (e.g. (1,1,128) for
        # 3D inputs, (1,128) for 2D) on the first forward pass — this is the
        # shape they have when checkpointed. Assign by reference rather than
        # copy_ so the buffer shape matches the saved shape regardless of the
        # fresh model's init shape.
        bn_layers = [l for l in model.layers if isinstance(l, BatchNorm1d)]
        for layer, (rm, rv) in zip(bn_layers, state['bn_buffers']):
            layer.running_mean = rm.clone()
            layer.running_var = rv.clone()
    metadata = {k: state[k] for k in ['vocab_size', 'block_size', 'n_embd', 'n_hidden']}
    if state.get('itos') is not None:
        metadata['itos'] = state['itos']
    if state.get('stoi') is not None:
        metadata['stoi'] = state['stoi']
    return model, metadata
