"""MLP with Kaiming init + BatchNorm — the production-quality single-hidden-layer
model and the stepping-stone to WaveNet's modular stack.

Same architecture as ``mlp.py`` with three changes that drop val NLL ~2.16 -> ~2.10
and stabilize training: Kaiming init with the tanh 5/3 gain (keeps pre-activations
in tanh's responsive range), output weights scaled by 0.01 (near-uniform initial
softmax, so initial loss ~log(27) rather than a confidently-wrong start), and
BatchNorm after the pre-activation. The pre-BN Linear drops its bias — BatchNorm's
mean-subtraction cancels a constant bias exactly, and its ``beta`` plays that role.
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
