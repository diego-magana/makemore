"""
Training loop, evaluation utilities, and ancestral sampling for the layer-based models.

The training utilities here are generic over any model exposing a `Sequential`-style
interface (forward via `__call__`, parameters via `.parameters()`, mode flag via
`.training`). The same `train()` and `evaluate()` functions are used for the
modular MLP, MLP+BatchNorm, and WaveNet — the only thing that varies is the
model definition passed in.
"""

import torch
import torch.nn.functional as F


def set_layer_training(model, training):
    """Propagate training/eval mode to every BatchNorm1d sublayer.

    BatchNorm1d behaves fundamentally differently in the two modes: training
    uses current-batch statistics and updates running buffers; evaluation uses
    the stored running buffers and skips updates. Layers that don't care
    about the distinction (Linear, Tanh, Embedding, FlattenConsecutive) do not
    expose a `training` attribute, so we set it only where it matters.

    We avoid using PyTorch's `nn.Module.train()` machinery because this
    library deliberately doesn't inherit from `nn.Module` — keeping the
    parameter registration and mode propagation visible is the pedagogical
    purpose of the layers module.
    """
    from makemore.layers import BatchNorm1d
    for layer in model.layers:
        if isinstance(layer, BatchNorm1d):
            layer.training = training


def train(model, Xtr, Ytr, steps, batch_size=32, lr_schedule=None, log_every=None,
          seed=2147483647):
    """Train ``model`` via minibatch SGD on cross-entropy (NLL) loss.

    Each step: forward -> cross-entropy (``F.cross_entropy`` fuses log-softmax + NLL
    for stability) -> zero grads via ``p.grad = None`` (skips a dense allocation for
    the sparsely-updated embedding table; see the README) -> backward -> SGD update.
    A batch of 32 gives cheap, noisy gradients that also act as light regularization.
    ``lr_schedule`` is a ``step -> lr`` callable (default: step-decay 0.1 -> 0.01 at
    the halfway point). Returns the per-step loss list for plotting.
    """
    g = torch.Generator().manual_seed(seed)
    if lr_schedule is None:
        lr_schedule = lambda step: 0.1 if step < steps // 2 else 0.01

    lossi = []
    set_layer_training(model, True)
    for step in range(steps):
        # Minibatch.
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]

        # Forward + loss.
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)

        # Zero gradients (= None, not .zero_()).
        for p in model.parameters():
            p.grad = None

        # Backward.
        loss.backward()

        # SGD update.
        lr = lr_schedule(step)
        for p in model.parameters():
            p.data += -lr * p.grad

        lossi.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"  step {step:6d} / {steps:6d}   lr {lr:.4f}   loss {loss.item():.4f}")
    return lossi


@torch.no_grad()
def evaluate(model, X, Y):
    """Compute mean cross-entropy loss on a held-out dataset.

    Two things have to be right here:

    1. `@torch.no_grad()` disables autograd tape construction. During
       evaluation we never call `.backward()`, so the tape would be built
       and immediately discarded — wasting memory and time. The decorator
       turns this off globally for the duration of the call.

    2. BatchNorm1d must be switched to eval mode (`training=False`) so it
       uses the EMA running statistics rather than current-batch statistics.
       If we evaluate with batch statistics, the same input produces
       different predictions depending on what other inputs are in the
       batch — non-deterministic and contaminated by the eval-set
       distribution. We restore the prior training state on exit so
       callers can resume training cleanly.

    Returns: scalar float — the mean NLL over (X, Y).
    """
    # Remember whether we were in training mode so we can restore it.
    from makemore.layers import BatchNorm1d
    was_training = any(isinstance(l, BatchNorm1d) and l.training for l in model.layers)
    set_layer_training(model, False)
    try:
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        return loss.item()
    finally:
        set_layer_training(model, was_training)


@torch.no_grad()
def generate_names(model, stoi, itos, block_size, n=5, temperature=1.0, seed=42):
    """Sample `n` names from the model via ancestral sampling.

    Ancestral sampling: at each position, the model produces a distribution
    over the next character given the current context; we sample from that
    distribution, append the sampled character, slide the context window,
    and repeat. Stop when the model emits the boundary token `.` — that's
    the end-of-word signal it was trained to predict.

    Temperature `T` rescales the logits before softmax:
        p_i  ∝  exp(logit_i / T)
    With T = 1.0 we sample from the model's native distribution. T < 1
    sharpens the distribution (lower-probability tokens are suppressed,
    pushing toward greedy decoding); in the T → 0 limit we recover argmax.
    T > 1 flattens the distribution (pushing toward uniform random); in
    the T → ∞ limit we sample uniformly from the vocabulary.

    Practical effect on generated names:
        T = 0.5 — conservative, generates common-sounding names with little
                  variety. Few surprises, few mistakes.
        T = 1.0 — natural samples from the model's learned distribution.
                  Mixes common and rare patterns.
        T = 2.0 — adventurous, generates more unusual character combinations
                  including some that violate the model's confident patterns.
                  More creative, more failures.

    We switch the model to eval mode for the same reason as `evaluate()` —
    BatchNorm using batch statistics on a batch of 1 would be nonsensical
    (variance of a single value is zero).

    Returns: list of `n` generated name strings (with the trailing `.` stripped).
    """
    from makemore.layers import BatchNorm1d
    was_training = any(isinstance(l, BatchNorm1d) and l.training for l in model.layers)
    set_layer_training(model, False)
    g = torch.Generator().manual_seed(seed)
    names = []
    try:
        for _ in range(n):
            out = []
            context = [0] * block_size
            while True:
                logits = model(torch.tensor([context]))
                logits = logits / temperature
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                if ix == 0:                # boundary token — stop here.
                    break
                out.append(itos[ix])
            names.append(''.join(out))
    finally:
        set_layer_training(model, was_training)
    return names
