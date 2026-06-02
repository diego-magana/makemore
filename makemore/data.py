"""
Character-level tokenization and dataset construction for the names corpus.

The corpus is `data/names.txt` — 32,032 first names, one per line. Character-level
modeling treats each character as a token; the vocabulary is 27 symbols:
26 lowercase letters plus the special `.` boundary token at index 0, which marks
both the start and end of each word. Modeling boundaries explicitly lets the
network learn name *lengths* alongside name spellings — without this, the model
has no signal for when to stop generating.

The supervised problem is next-character prediction: given a context window of
the previous `block_size` characters, predict the next one. This converts each
name into many training examples — one per position in the sequence — turning
the 32K-word corpus into ~228K supervised examples at `block_size=3` (and
proportionally more at larger block sizes, since each name still contributes
roughly len(name)+1 examples regardless of the context window).
"""

from pathlib import Path

import torch


def load_names(path="data/names.txt"):
    """Read the corpus from disk and return a list of lowercase name strings.

    The default path is relative to the repository root — code that imports
    this module should run from the repo root (which is also where `pip
    install -e .` registers the package). Blank lines are filtered out
    defensively so a trailing newline in the data file doesn't introduce
    an empty-string "name" into the vocabulary.
    """
    return [w for w in Path(path).read_text().splitlines() if w]


def build_vocab(words):
    """
    Build stoi/itos mappings from a word list.

    `.` is reserved at index 0 as the start/stop boundary signal. Reserving
    index 0 specifically (rather than a more natural choice like 26 for
    "after all the letters") makes context padding trivial: the initial
    context window is `[0] * block_size`, which represents `block_size`
    boundary tokens, semantically "we are at the very start of a word."
    Without this convention, padding would need a separate sentinel value
    distinct from any letter index.

    Sorting before indexing ensures `stoi['a'] == 1` is stable across runs
    — this matters when comparing models trained on different days, or
    when loading a checkpoint trained by a previous version of the code.
    A non-deterministic vocabulary would make checkpoints non-portable.

    `vocab_size == 27` is the fan_in to the embedding table on the input side
    and the fan_out of the final linear projection on the output side. Both
    sides must use the same vocabulary — a softmax over 27 classes only
    matches the targets if those targets were tokenized with the same map.

    Returns:
        stoi: dict[str, int] — character → integer index
        itos: dict[int, str] — integer index → character
    """
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_dataset(words, stoi, block_size):
    """
    Construct a supervised (X, Y) dataset via sliding-window tokenization.

    `block_size` defines the receptive field: how many past characters the
    model conditions on to predict the next one. This is the explicit
    operationalization of the Markov assumption — at `block_size=3` we are
    fitting a 3rd-order Markov model over characters; at `block_size=8`
    (the WaveNet setting) we are fitting an 8th-order model.

    Each word contributes `len(word) + 1` training examples: one for each
    character in the word, plus one for the final `.` (end-of-word) target.
    The context starts as `[0] * block_size` — `block_size` boundary tokens
    — and rolls forward one position at a time: the oldest character drops
    off the front, the newest appends to the back. This rolling-update
    pattern is the Markov assumption made explicit in code.

    Worked example for the word "emma" at block_size=3:
        context [...]  →  predict 'e'      (X=[0,0,0],  Y=5)
        context [..e]  →  predict 'm'      (X=[0,0,5],  Y=13)
        context [.em]  →  predict 'm'      (X=[0,5,13], Y=13)
        context [emm]  →  predict 'a'      (X=[5,13,13],Y=1)
        context [mma]  →  predict '.'      (X=[13,13,1],Y=0)
    Five examples from a four-letter word.

    Returns:
        X: torch.LongTensor of shape (N, block_size) — context windows
        Y: torch.LongTensor of shape (N,)            — target characters
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # crank the window forward one position
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def split_dataset(words, stoi, block_size, seed=42):
    """
    Partition the corpus into 80/10/10 train/dev/test splits.

    Shuffling before splitting ensures the splits are approximately i.i.d.
    samples from the same distribution. Without shuffling, the names file's
    natural ordering (alphabetical in the source) would create distributional
    shift between train and test — the test set would be dominated by names
    starting with later letters of the alphabet, which is a meaningfully
    different conditional distribution from the training set.

    The dev split exists for architecture search and hyperparameter selection.
    The test split is a held-out final-performance audit and should not be
    touched until the final model is locked. The 80/10/10 ratio is the
    convention Karpathy uses for this corpus; with 32K names, 10% gives a
    ~3.2K-word evaluation set, which is large enough that the loss estimate
    has standard error well below the differences we care about between
    architectures (~0.05 nats).

    The seed makes the partition reproducible — every reader of this repo
    sees the same train/dev/test split, so reported losses are directly
    comparable across runs and machines.

    Returns six tensors: Xtr, Ytr, Xdev, Ydev, Xte, Yte.
    """
    import random
    rng = random.Random(seed)
    words = list(words)
    rng.shuffle(words)

    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    Xtr,  Ytr  = build_dataset(words[:n1],   stoi, block_size)
    Xdev, Ydev = build_dataset(words[n1:n2], stoi, block_size)
    Xte,  Yte  = build_dataset(words[n2:],   stoi, block_size)

    return Xtr, Ytr, Xdev, Ydev, Xte, Yte
