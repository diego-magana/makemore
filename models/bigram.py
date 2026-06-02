"""
Two bigram models — the statistical and neural baselines for the progression.

`CountBigram` is a pure statistical model: count the (prev_char, next_char)
pairs in the training data, smooth, normalize, and predict by table lookup.
`NeuralBigram` is a single-layer neural network with the *same* expressive
power but trained by gradient descent. At convergence they reach identical
losses — they are two views of the same model class. The point of training
the neural version is to demonstrate that gradient descent on a cross-entropy
objective recovers what counting-and-smoothing already gives us, and to set
up the conceptual bridge to the MLP: the neural bigram's `one-hot @ W` is
mathematically an embedding lookup, which is exactly what the MLP generalizes.
"""

import torch
import torch.nn.functional as F


class CountBigram:
    """Count-based bigram model with Laplace smoothing.

    Fit:
        N[i, j] = (1 + count of bigram (i, j) in training data)
        P[i, j] = N[i, j] / N[i, :].sum()

    The `+1` is Laplace (add-one) smoothing. It prevents `-inf` losses on any
    bigram that never appears in training: an unseen bigram has count 0,
    `log 0 = -inf`, and a single such bigram in the eval set sends the mean
    loss to infinity. Smoothing pushes every probability strictly above zero
    at a small cost to the probabilities of frequent bigrams.

    Why NLL = minimizing negative log likelihood:
        likelihood = prod_t P(x_t | x_{t-1})
        log likelihood = sum_t log P(x_t | x_{t-1})
        max log likelihood = min (-log likelihood) = min NLL
    Log is monotone, so maximizing likelihood is the same as maximizing log
    likelihood, which is the same as minimizing NLL. This is the objective
    every neural language model after this one will optimize as well, via
    cross-entropy loss — `F.cross_entropy(logits, targets)` is literally NLL
    after an implicit log_softmax over the logits.
    """

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.P = None  # set by fit()

    def fit(self, words, stoi):
        """Count bigrams, smooth with +1, normalize rows to probabilities."""
        N = torch.ones((self.vocab_size, self.vocab_size), dtype=torch.int64)  # +1 smoothing
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                N[stoi[ch1], stoi[ch2]] += 1
        # Row-normalize to a conditional probability table P[prev, next].
        P = N.float()
        P = P / P.sum(dim=1, keepdim=True)
        self.P = P
        self.N = N  # keep for the heatmap visualization

    def nll(self, words, stoi):
        """Mean negative log-likelihood per character over `words`.

        This is the same scalar reported as 'val NLL' for every model in the
        progression — keeping the units identical is what makes the count
        baseline directly comparable to the WaveNet number.
        """
        log_likelihood = 0.0
        n = 0
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1, ix2 = stoi[ch1], stoi[ch2]
                log_likelihood += torch.log(self.P[ix1, ix2])
                n += 1
        return -log_likelihood.item() / n

    def sample(self, itos, n=5, seed=42):
        """Generate `n` names by sampling the bigram chain to completion."""
        g = torch.Generator().manual_seed(seed)
        names = []
        for _ in range(n):
            out = []
            ix = 0  # start at boundary token
            while True:
                p = self.P[ix]
                ix = torch.multinomial(p, num_samples=1, generator=g).item()
                if ix == 0:
                    break
                out.append(itos[ix])
            names.append(''.join(out))
        return names


class NeuralBigram:
    """Single-layer linear network — gradient-descent equivalent of CountBigram.

    Architecture:
        one_hot(x) @ W  →  logits  →  log_softmax  →  NLL loss

    Critically, `one_hot(x) @ W` is mathematically identical to `W[x]` —
    a row lookup into the weight matrix. This is the same operation an
    Embedding layer performs. The neural bigram is therefore an embedding
    table of width = vocab_size, followed directly by the softmax output.
    The next step in the progression (the MLP) keeps the embedding lookup
    but replaces "directly to softmax" with "concatenate context, pass
    through hidden layers, then to softmax" — which is exactly the gain
    that gives the MLP its lower loss.

    At optimality this model reaches the same loss as CountBigram: there
    are only `vocab_size^2 = 729` parameters in W, and the cross-entropy
    objective is strictly convex over the softmax simplex, so SGD converges
    to the global minimum, which is the empirical conditional distribution
    — exactly what counting gives you. The point of training it is to
    demonstrate that the count model is a *neural network at convergence*,
    not a separate thing that gets replaced by neural networks. Everything
    that follows is the same idea — fit conditional distributions by
    gradient descent — with progressively richer parametrizations.
    """

    def __init__(self, vocab_size, seed=2147483647):
        g = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    def parameters(self):
        return [self.W]

    def fit(self, xs, ys, steps=200, lr=50.0):
        """Train the neural bigram by cross-entropy + SGD.

        The high learning rate (lr=50) is deliberate and correct for this
        specific architecture: there is exactly one linear layer with no
        hidden state, and the loss landscape is strictly convex. We can
        take large steps without divergence — and at lr=0.1 (the standard
        for the MLP) this model would take 50,000+ steps to converge instead
        of 200. The lesson here is that learning rates are *architecture-
        specific*: there is no universal "good" lr.
        """
        for step in range(steps):
            # Forward: one-hot encode, project, softmax, NLL.
            xenc = F.one_hot(xs, num_classes=self.vocab_size).float()
            logits = xenc @ self.W
            # F.cross_entropy fuses log_softmax + NLL for numerical stability.
            loss = F.cross_entropy(logits, ys)
            # Backward + SGD step.
            self.W.grad = None
            loss.backward()
            with torch.no_grad():
                self.W -= lr * self.W.grad
        return loss.item()

    @torch.no_grad()
    def nll(self, xs, ys):
        """Mean NLL over a held-out (xs, ys) set."""
        xenc = F.one_hot(xs, num_classes=self.vocab_size).float()
        logits = xenc @ self.W
        return F.cross_entropy(logits, ys).item()

    @torch.no_grad()
    def sample(self, itos, n=5, seed=42):
        """Generate `n` names from the trained neural bigram."""
        g = torch.Generator().manual_seed(seed)
        names = []
        for _ in range(n):
            out = []
            ix = 0
            while True:
                xenc = F.one_hot(torch.tensor([ix]), num_classes=self.vocab_size).float()
                logits = xenc @ self.W
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                if ix == 0:
                    break
                out.append(itos[ix])
            names.append(''.join(out))
        return names


def make_bigram_xy(words, stoi):
    """Build (xs, ys) tensors for the neural bigram (no context window).

    Each adjacent character pair becomes a training example:
        word='emma' → pairs (.,e), (e,m), (m,m), (m,a), (a,.)
    """
    xs, ys = [], []
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)
