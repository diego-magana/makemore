"""
MLP with Kaiming initialization and BatchNorm — the production-quality
single-layer model and the stepping-stone to WaveNet's modular stack.

This is the same architecture as `mlp.py` (one hidden layer, tanh, linear
output) with two changes that together reduce val NLL from ~2.16 to ~2.10
and stabilize training:

1. *Kaiming initialization* with the tanh gain of 5/3. The weights of the
   first linear layer are scaled by `(5/3) / sqrt(fan_in)`, which keeps the
   pre-activation variance at ~1 across the layer — putting tanh in its
   responsive (high-gradient) regime rather than its saturated regime.

2. *Output layer scaled down* by 0.01. The final-layer weights `W2` are
   multiplied by 0.01 so the initial logits are near zero, which means the
   initial softmax is approximately uniform over the 27 classes, which
   means the initial loss is approximately `log(27) ≈ 3.30`. Without this,
   the initial logits are random N(0,1)-scaled values and produce a
   "confidently wrong" initial softmax that takes the first hundred steps
   of training just to undo. This is sometimes called fixing the "hockey
   stick" loss curve.

3. *Batch normalization* applied after the linear pre-activation. This
   forces the input to tanh to zero mean and unit variance across the
   batch, decoupling layer-wise scale issues from the optimization
   dynamics. See `makemore/layers.py::BatchNorm1d` for full discussion.

A subtle correctness point that appears here for the first time:
*the bias is dropped from the linear layer immediately before BatchNorm*.

Proof that bias is redundant after BatchNorm:

    Let z = Wx + b be the linear pre-activation.
    BN(z) = (z - mean(z)) / std(z) * gamma + beta

    mean(z) = mean(Wx) + b   (b is a constant; mean is linear)
    z - mean(z) = (Wx + b) - (mean(Wx) + b) = Wx - mean(Wx)

    So b is subtracted out exactly. The remaining scale-and-shift is
    governed entirely by gamma, beta — which are BN's own learnable
    parameters. Keeping b would only add `n_hidden` extra parameters
    that contribute nothing the network can't already express via beta.

We keep the bias on `W2` (the output linear) because it is *not* followed
by BatchNorm — the output goes straight into the softmax + cross-entropy.
"""

import torch
import torch.nn.functional as F

from makemore.layers import Linear, BatchNorm1d, Tanh, Embedding, Sequential


class MLPBatchNorm:
    """MLP + Kaiming + BatchNorm, built on the modular layer library.

    The model is a `Sequential` over our own Linear/BatchNorm1d/Tanh classes,
    not torch.nn equivalents. This is deliberate: the probing notebook
    reads `tanh_layer.out` directly to inspect activation distributions,
    which requires the explicit `self.out` caching our layers provide.

    Architecture:
        Embedding(vocab_size, n_embd)
        FLATTEN  : .view to (B, block_size * n_embd)
        Linear(block_size * n_embd, n_hidden, bias=False)   ← bias=False before BN
        BatchNorm1d(n_hidden)
        Tanh()
        Linear(n_hidden, vocab_size)                        ← bias=True (no BN after)

    `Linear` with `bias=False` is the proof from the module docstring made
    operational — we never allocate `b1`.

    The first-layer weight is rescaled by 5/3 after construction. This is
    the Kaiming *gain* for tanh: the standard Kaiming-He derivation assumes
    a ReLU nonlinearity, which has 1/2 the variance-passing fraction of
    a linear unit; tanh, with its zero-mean output, passes more of the
    variance, and the gain of ~5/3 compensates so the post-tanh distribution
    stays near unit variance. The output-layer weight is rescaled by 0.01
    to fix the "confidently wrong" initial loss as described above.
    """

    def __init__(self, vocab_size, block_size=3, n_embd=10, n_hidden=200,
                 seed=2147483647):
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size

        torch.manual_seed(seed)

        # Build the Sequential. We define the layer list flat — the
        # Embedding + FLATTEN + Linear pattern is replicated from Bengio
        # 2003 but expressed in our modular API.
        self.embedding = Embedding(vocab_size, n_embd)
        self.layers = [
            Linear(block_size * n_embd, n_hidden, bias=False),
            BatchNorm1d(n_hidden),
            Tanh(),
            Linear(n_hidden, vocab_size),
        ]
        self.model = Sequential(self.layers)

        # Kaiming gain (5/3) on the first Linear's weight.
        with torch.no_grad():
            self.layers[0].weight *= 5.0 / 3.0
            # Scale down the final layer to start with near-uniform predictions.
            self.layers[-1].weight *= 0.01

        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return self.embedding.parameters() + self.model.parameters()

    def __call__(self, X):
        emb = self.embedding(X)                                       # (B, T, n_embd)
        x = emb.view(emb.shape[0], -1)                                # (B, T*n_embd)
        return self.model(x)                                          # (B, vocab_size)

    def loss(self, X, Y):
        return F.cross_entropy(self(X), Y)

    # Expose training state for the train.py utilities — they look for
    # `model.layers` and set `.training` on BatchNorm sublayers. We delegate
    # to the inner Sequential.
    @property
    def training_layers(self):
        return self.layers
