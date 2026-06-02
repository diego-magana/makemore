"""
Modular layer library for character-level language modeling.

This module factors the WaveNet architecture into reusable layer classes:
Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, and the Sequential
container that wires them together. The same primitives compose into every
model in `models/` (MLP, MLP+BatchNorm, WaveNet); the only thing that changes
across models is the ordering of the layer list passed to Sequential.

The interface deliberately mirrors PyTorch's `torch.nn` API — every layer
exposes `__call__(x)` for the forward pass and `parameters()` for the
gradient-bearing tensors — but without inheriting from `nn.Module`. This is
pedagogical: it makes the parameter registration and training/eval-mode
machinery visible rather than hidden behind metaclass magic.

Layers store their last forward-pass output as `self.out`. This is not for
backprop (PyTorch's autograd handles that) but for *introspection*: the
probing analysis in `06_probing.ipynb` reads `tanh_layer.out` to characterize
the activation distribution at each hierarchical level, which is impossible
without explicit caching.
"""

import torch


# ---------------------------------------------------------------------------
# Linear projection
# ---------------------------------------------------------------------------

class Linear:
    """Affine transformation: y = x @ W + b (optional bias).

    Kaiming initialization with `1 / sqrt(fan_in)` scaling:
        var(W @ x) = fan_in * var(W) * var(x)
        if var(W) = 1/fan_in and var(x) = 1, then var(W @ x) = 1.

    This is the variance-preservation property that keeps activations from
    exploding or vanishing as they propagate through a deep stack. Without it,
    each Linear layer would multiply the activation variance by `fan_in` —
    in a 4-layer network with 100-wide hidden states, that compounds to a
    `100^4 = 10^8` scaling factor between layer 1 and layer 4, which is
    catastrophic in either direction (saturated tanh on the way up, dead
    gradients on the way down).

    For the layer immediately preceding a tanh nonlinearity, the standard
    Kaiming gain is `5/3` (not 1) — see `models/mlp_batchnorm.py`'s
    `MLPBatchNorm.__init__` for the rationale and the application site.
    That extra scaling is applied externally to `self.weight` after
    construction; the constructor here uses the unscaled `1/sqrt(fan_in)`
    base.

    `bias=False` is used immediately before a BatchNorm1d layer:
        BN(W @ x + b) subtracts the batch mean and divides by std;
        the batch mean of `W @ x + b` is `mean(W @ x) + b`, so the `+b`
        is exactly cancelled by the mean subtraction and contributes nothing
        beyond what BatchNorm's learnable `beta` already provides. Keeping
        the bias would only waste parameters and gradient bandwidth.

    Why this matters for language modeling: every layer in the WaveNet stack
    is one of these. Variance preservation across depth is what lets us train
    a 3-level hierarchical receptive field stably — each level operates on
    activations from the previous level, and any drift compounds.
    """

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out)) / fan_in ** 0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out = self.out + self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])


# ---------------------------------------------------------------------------
# Batch normalization
# ---------------------------------------------------------------------------

class BatchNorm1d:
    """Batch normalization for 2D or 3D inputs.

    The problem ("internal covariate shift"): as earlier weights update during
    training, the distribution of inputs to later layers shifts. Later layers
    have to chase a moving target, which makes training unstable and forces
    small learning rates. BatchNorm fixes this by re-centering and rescaling
    each layer's input to zero mean, unit variance using the current batch's
    statistics.

    Forward pass (training):
        mu = mean(x, batch_dim)
        sigma2 = var(x, batch_dim)
        xhat = (x - mu) / sqrt(sigma2 + eps)
        out = gamma * xhat + beta

    The affine transform (gamma, beta) is necessary because pure normalization
    can remove representational capacity — if the optimal pre-activation
    distribution for a downstream layer is *not* zero-mean unit-variance,
    we'd be hobbling the model by forcing it to be so. Gamma and beta let the
    network learn to "undo" the normalization when a different scale or shift
    is more useful, while keeping the *initial* state of training in the
    stable regime.

    Forward pass (inference):
        Use the running statistics accumulated during training, not the
        current batch. At inference time the batch may be size 1, or may
        contain examples from different distributions than training — in
        either case, batch statistics would be noisy and would make
        predictions non-deterministic (the same input would yield different
        outputs depending on what else was in the batch).

    Running statistics are *buffers*, not parameters: they are updated by EMA
    inside a `torch.no_grad()` block, not via gradient descent. They should
    persist across training steps (so we accumulate good estimates over the
    whole training run) but should not contribute to the computation graph
    (so backprop doesn't try to differentiate through them).

    The `ndim` check handles two input layouts:
        2D: (B, C) — reduce over batch only          → dim=0
        3D: (B, T, C) — reduce over batch and time   → dim=(0, 1)
    The 3D case is what arises inside the WaveNet's hierarchical levels,
    where the layer sees a sequence of bigram embeddings.

    Why this matters for language modeling: BatchNorm is what lets the modular
    MLP stack train at all (see `models/mlp_batchnorm.py` for the loss
    progression). Without it, signal variance drifts across layers and the
    training collapses to learning only the bigram statistics — the deeper
    architecture provides no benefit. With BatchNorm, the same architecture
    drops val NLL from ~2.45 to ~2.10.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Trainable affine parameters (gradient-bearing).
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # EMA buffers (updated under no_grad, NOT in self.parameters()).
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            # Reduce over the batch axis (and time, if a 3D input).
            if x.ndim == 2:
                dim = 0
            elif x.ndim == 3:
                dim = (0, 1)
            else:
                raise ValueError(f"BatchNorm1d expects 2D or 3D input, got {x.ndim}D")
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta

        # Update EMA buffers under no_grad so they don't pollute the graph.
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------

class Tanh:
    """Hyperbolic tangent activation, squashing to (-1, 1).

    No parameters, no internal state — purely an elementwise function. The
    only non-obvious detail is that we *cache* the output as `self.out`. This
    is not used during training (autograd handles the backward pass), but it
    is essential for the probing analysis in `06_probing.ipynb`, which needs
    to inspect the post-tanh activation distribution at each hierarchical
    level to detect saturation.

    Saturation context: tanh's derivative is `1 - tanh^2(x)`, which collapses
    to ~0 when `|x| > ~2`. A "saturated" neuron (output close to ±1) has
    near-zero local gradient, so the upstream learning signal dies at that
    unit. Healthy training requires keeping the fraction of saturated neurons
    low (the conventional threshold is |activation| > 0.97 for "saturated";
    target <15%). This is the entire purpose of Kaiming init and BatchNorm:
    keep the pre-activations in tanh's responsive region.
    """

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


# ---------------------------------------------------------------------------
# Embedding lookup
# ---------------------------------------------------------------------------

class Embedding:
    """Learned lookup table mapping integer indices to continuous vectors.

    Conceptual bridge — the embedding layer is mathematically identical to
    multiplying a one-hot vector by a `(vocab_size, embedding_dim)` weight
    matrix:
        one_hot(i) @ W = W[i]
    The one-hot/matmul formulation is O(vocab_size) per lookup; the embedding
    formulation is O(1). At vocab_size=27 the difference is small, but at
    vocab_size=50,257 (GPT-2's BPE vocab) it is the difference between a
    feasible model and an infeasible one. This is the connection point
    between the bigram neural model (which is literally one-hot @ W) and
    the MLP (which uses an Embedding).

    Why "distributed representation" matters: in the bigram model, each
    character is represented by a sparse one-hot vector — no two characters
    share any structure. In the embedding model, each character is a dense
    vector in continuous space, and characters with similar contextual
    behavior end up in similar regions of that space. This is what allows
    the MLP to generalize: it doesn't need to see every (char_i, char_j)
    bigram to make predictions about them, because similar characters share
    similar embeddings and similar downstream behavior.

    Shape: `IX: (B, T)` → `out: (B, T, embedding_dim)`. Each integer in IX
    is independently mapped through the table, producing one embedding
    vector per input position.

    Why this matters for language modeling: this is the layer where the
    discrete-to-continuous transition happens. Every neural language model,
    from this MLP to GPT-4, has an embedding layer as the first operation.
    The geometry of this embedding space — explored systematically in
    `06_probing.ipynb` — is the first place to look when asking "what does
    the model think the structure of the input is?"
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


# ---------------------------------------------------------------------------
# Consecutive flatten — the WaveNet core
# ---------------------------------------------------------------------------

class FlattenConsecutive:
    """Hierarchical "chunked" flattening — the architectural core of WaveNet.

    Standard flat flattening reshapes `(B, T, C)` to `(B, T*C)` in one step,
    forcing the very first hidden layer to learn relationships across the
    entire context window simultaneously. For an 8-character context this is
    a 64-token receptive field at layer 1, with no notion of locality —
    position 1 and position 8 are treated as equally adjacent to position 4.

    FlattenConsecutive(n) instead groups `n` adjacent positions and folds
    them into the channel dimension:
        (B, T, C) → (B, T // n, C * n)

    Applied iteratively with `n=2` in a stack of three, this produces a
    hierarchical receptive field:
        Layer 1 sees bigrams        (B, 8, C)  → (B, 4, 2C)
        Layer 2 sees 4-grams        (B, 4, C') → (B, 2, 2C')
        Layer 3 sees the full 8     (B, 2, C'')→ (B, 1, 2C'')  → (B, 2C'')

    The structural analogy is to 1D dilated convolution: each level operates
    on the output of the previous level, so by level 3 each unit "sees"
    8 original input positions through 3 layers of nonlinearity — much more
    expressive than 8 positions through 1 layer.

    The trailing `if x.shape[1] == 1` squeeze handles the final level: when
    the time dimension collapses to length 1, we drop it so the output is
    2D (B, features) rather than 3D (B, 1, features) — making the output
    compatible with a standard 2D Linear / BatchNorm1d for the final projection.

    Why this matters for language modeling: language has hierarchical
    structure. Adjacent characters form phoneme-like units; sequences of
    those form syllables; sequences of syllables form words. A flat MLP can
    in principle learn this structure, but has to discover it during
    optimization from a much harder starting point. The hierarchical
    architecture *bakes in* the inductive bias that locality matters —
    which is exactly the bias that holds in natural language.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            # Last level — drop the singleton time dim so downstream layers
            # receive a clean (B, features) tensor.
            x = x.squeeze(1)
        self.out = x
        return self.out

    def parameters(self):
        # Pure shape operation — no learnable weights.
        return []


# ---------------------------------------------------------------------------
# Sequential container
# ---------------------------------------------------------------------------

class Sequential:
    """Wires a list of layers into a single forward pass and parameter list.

    Three responsibilities:

    1. Forward propagation. Each layer's output becomes the next layer's
       input. The container abstracts away the manual `x = layer(x)` loop
       so the model definition can be a flat list — see `models/wavenet.py`
       for how this reads.

    2. Centralized parameter registry. `parameters()` walks every sublayer
       and gathers their `parameters()` into a single flat list. The training
       loop iterates that list for the SGD step; without this aggregation,
       every model would need bespoke parameter-tracking code.

    3. Training/eval mode propagation. Setting `model.training = False`
       on the container needs to propagate to every BatchNorm1d sublayer so
       they switch to using running statistics. This is done explicitly in
       `train.py::evaluate()` rather than baked into Sequential, to keep the
       container minimal — but if extended to nested containers, this would
       need a recursive mode-propagation method (which is what PyTorch's
       `nn.Module.train()` does).
    """

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
