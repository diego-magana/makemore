"""Modular layer library for character-level language modeling.

Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, and the Sequential
container compose into every model in ``models/`` — the only thing that changes
across models is the order of the layer list. The interface mirrors PyTorch's
``torch.nn`` (``__call__`` for forward, ``parameters()`` for the trainable
tensors) but deliberately doesn't inherit from ``nn.Module``: keeping parameter
registration and train/eval switching in plain sight is the point.

Every layer caches its forward output as ``self.out``. That's not for backprop —
autograd handles that — but for introspection: the probing notebook reads
``tanh_layer.out`` to characterize activations at each hierarchical level.
"""

import torch


class Linear:
    """Affine map ``y = x @ W (+ b)``.

    Weights use the Kaiming ``1/sqrt(fan_in)`` scaling that preserves activation
    variance across depth (without it, variance compounds by ``fan_in`` per layer
    and a deep stack saturates or vanishes). The tanh gain of 5/3 is applied
    externally after construction — see ``models/mlp_batchnorm.py``. ``bias=False``
    is used before every BatchNorm, whose mean-subtraction cancels a constant bias
    exactly (its learnable ``beta`` plays that role instead).
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


class BatchNorm1d:
    """Batch normalization for 2D ``(B, D)`` or 3D ``(B, T, D)`` inputs.

    In training it normalizes each feature over the batch (and time) axis to zero
    mean / unit variance, then applies a learnable affine (``gamma``, ``beta``); in
    eval it uses the EMA ``running_mean``/``running_var`` instead, so a single
    example is scored the same way it was trained. The EMA buffers update under
    ``no_grad`` and are not returned by ``parameters()``.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
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

        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    """Elementwise tanh, squashing to (-1, 1). No parameters.

    Caches ``self.out`` so the probing notebook can measure the saturated
    fraction (|activation| > 0.97) per level — saturated units have near-zero
    local gradient ``1 - tanh^2``, so keeping that fraction low is what Kaiming
    init and BatchNorm are for.
    """

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    """Learned lookup table, ``IX (B, T) -> (B, T, embedding_dim)``.

    Equivalent to ``one_hot(i) @ W`` but O(1) per lookup instead of O(vocab) —
    the difference between feasible and not once vocabularies reach real sizes.
    It's also where discrete tokens become continuous vectors, so characters with
    similar behavior can share geometry; that geometry is what the probing
    notebook's embedding analysis inspects.
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]


class FlattenConsecutive:
    """Chunked flatten ``(B, T, C) -> (B, T // n, C * n)`` — the WaveNet core.

    A flat flatten would force layer 1 to relate all 8 context positions at once,
    with no locality. Folding ``n`` adjacent positions into the channel dimension
    and stacking it instead builds a hierarchical receptive field: with ``n=2``
    over three levels, layer 1 sees bigrams, layer 2 sees 4-grams, layer 3 sees
    the full 8 — the 1D-dilated-convolution structure that bakes in the bias that
    nearby characters matter. The singleton-time squeeze at the last level hands
    downstream layers a clean 2D ``(B, features)`` tensor.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.view(B, T // self.n, C * self.n)
        if x.shape[1] == 1:
            x = x.squeeze(1)   # last level: drop the length-1 time dim
        self.out = x
        return self.out

    def parameters(self):
        return []


class Sequential:
    """Chains layers into one forward pass and one flat parameter list.

    Forward threads each layer's output into the next; ``parameters()`` gathers
    every sublayer's tensors for the SGD loop. Train/eval mode is propagated to
    the BatchNorm sublayers explicitly in ``train.py`` rather than here, to keep
    the container minimal.
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
