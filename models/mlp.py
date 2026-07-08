"""
Bengio 2003 multilayer perceptron — the first model in the progression
to use learned distributed representations and a context window.

Architecture:
    integer indices x : (B, block_size)
    embeddings    C[x] : (B, block_size, n_embd)
    concatenated  emb : (B, block_size * n_embd)
    hidden        tanh(emb @ W1 + b1) : (B, n_hidden)
    logits        h @ W2 + b2 : (B, vocab_size)
    loss          cross_entropy(logits, y)

The two architectural ideas this model introduces — relative to the bigram
neural model that preceded it — are:

1. *Learned embeddings*. Each character is a point in continuous `n_embd`-
   dimensional space, and that point's coordinates are learned. Similar
   characters can end up in similar regions of the space, which lets the
   model generalize across them without seeing every specific bigram. This
   is the "distributed representation" that the Bengio 2003 paper is
   primarily famous for.

2. *Context window*. Rather than conditioning on only the previous
   character, condition on the previous `block_size` characters (here 3).
   The conditional distribution P(x_t | x_{t-3}, x_{t-2}, x_{t-1}) carries
   more information than P(x_t | x_{t-1}) — and the price is paid by a
   larger model, not by exponentially many smoothed counts.

Together these reduce val NLL from ~2.45 (bigram) to ~2.16 (this model)
on the names corpus.
"""

import torch
import torch.nn.functional as F


class MLP:
    """Bengio 2003 MLP for character-level next-token prediction.

    Embeds each of ``block_size`` context characters, concatenates the embeddings
    into one context vector, and runs it through a tanh hidden layer to a softmax
    over the vocabulary. Initialization is deliberately naive N(0,1) — not Kaiming —
    so the next model (``mlp_batchnorm.py``) can demonstrate why init matters.
    ~11.9K params at block_size=3, n_embd=10, n_hidden=200.
    """

    def __init__(self, vocab_size, block_size=3, n_embd=10, n_hidden=200,
                 seed=2147483647):
        g = torch.Generator().manual_seed(seed)
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_hidden = n_hidden
        self.vocab_size = vocab_size

        self.C  = torch.randn((vocab_size, n_embd),                 generator=g)
        self.W1 = torch.randn((block_size * n_embd, n_hidden),      generator=g)
        self.b1 = torch.randn(n_hidden,                              generator=g)
        self.W2 = torch.randn((n_hidden, vocab_size),                generator=g)
        self.b2 = torch.randn(vocab_size,                            generator=g)

        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return [self.C, self.W1, self.b1, self.W2, self.b2]

    def __call__(self, X):
        emb = self.C[X]                                              # (B, T, n_embd)
        emb_flat = emb.view(-1, self.block_size * self.n_embd)       # (B, T*n_embd)
        h = torch.tanh(emb_flat @ self.W1 + self.b1)                 # (B, n_hidden)
        logits = h @ self.W2 + self.b2                               # (B, vocab_size)
        return logits

    def loss(self, X, Y):
        return F.cross_entropy(self(X), Y)

    @torch.no_grad()
    def evaluate(self, X, Y):
        return self.loss(X, Y).item()

    @torch.no_grad()
    def sample(self, itos, n=5, seed=42, temperature=1.0):
        """Ancestral sampling — same protocol as the rest of the progression."""
        g = torch.Generator().manual_seed(seed)
        names = []
        for _ in range(n):
            out = []
            context = [0] * self.block_size
            while True:
                logits = self(torch.tensor([context]))
                logits = logits / temperature
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                if ix == 0:
                    break
                out.append(itos[ix])
            names.append(''.join(out))
        return names


def find_lr(model, Xtr, Ytr, batch_size=32, n_steps=1000,
            lr_min=1e-3, lr_max=1.0, seed=2147483647):
    """Learning rate finder — log-scale sweep with loss tracking.

    Procedure: train the model with an exponentially increasing learning
    rate over n_steps, recording (lr, loss) at each step. Plot loss vs
    log(lr). The optimal lr lies in the region just *before* the loss
    starts climbing — typically half to one decade lower than the lr that
    minimizes the loss curve (because lr at the minimum is already
    aggressive enough to cause occasional instability).

    This is a cheap diagnostic — n_steps=1000 takes seconds — and removes
    most of the guesswork from learning rate selection. The first time
    you see a clean elbow in the loss-vs-lr plot, you stop reaching for
    hand-picked schedules.

    Returns (lrs, losses) as parallel lists.
    """
    g = torch.Generator().manual_seed(seed)
    lrs_log = torch.linspace(torch.log10(torch.tensor(lr_min)),
                             torch.log10(torch.tensor(lr_max)), n_steps)
    lrs = 10 ** lrs_log
    losses = []
    for step in range(n_steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
        loss = model.loss(Xtr[ix], Ytr[ix])
        for p in model.parameters():
            p.grad = None
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p.data += -lrs[step].item() * p.grad
        losses.append(loss.item())
    return lrs.tolist(), losses
