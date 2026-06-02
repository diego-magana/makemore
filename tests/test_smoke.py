"""Smoke tests — verify package imports and core layer math.

These are not exhaustive unit tests; the validation that the implementation
is correct comes from the notebooks themselves (the `cmp()` oracle in
04_backprop_ninja.ipynb proves every gradient is correct; 05_wavenet.ipynb
proves the WaveNet trains to a reasonable loss). These smoke tests catch
the kind of breakage that would prevent the notebooks from running at all —
import failures, shape errors at toy scale, basic numerical sanity.
"""

import torch
import torch.nn.functional as F


def test_package_imports():
    """All public symbols are importable from the top-level package."""
    from makemore import (
        Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential,
        load_names, build_vocab, build_dataset, split_dataset,
        train, evaluate, generate_names,
    )
    assert callable(Linear) and callable(load_names)


def test_models_imports():
    """All model classes are importable."""
    from models.bigram import CountBigram, NeuralBigram, make_bigram_xy
    from models.mlp import MLP, find_lr
    from models.mlp_batchnorm import MLPBatchNorm
    from models.wavenet import build_wavenet, save_checkpoint, load_checkpoint
    assert callable(CountBigram)


def test_linear_shape_and_kaiming():
    """Linear preserves activation variance approximately (Kaiming property)."""
    from makemore import Linear
    torch.manual_seed(0)
    layer = Linear(fan_in=64, fan_out=128, bias=False)
    x = torch.randn(1000, 64)
    out = layer(x)
    assert out.shape == (1000, 128)
    # Kaiming preserves variance; allow generous tolerance for 1000 samples.
    assert 0.7 < out.std().item() < 1.3, f"variance not preserved: std = {out.std().item()}"


def test_batchnorm_zero_mean_unit_var():
    """BatchNorm output should be approximately zero-mean unit-variance in training mode."""
    from makemore import BatchNorm1d
    torch.manual_seed(0)
    bn = BatchNorm1d(32)
    x = torch.randn(64, 32) * 5.0 + 2.0  # deliberately not unit
    out = bn(x)
    assert abs(out.mean().item()) < 1e-5, f"mean not zero: {out.mean().item()}"
    # Bessel's correction makes the variance slightly less than 1 with default init (gamma=1).
    assert 0.9 < out.std().item() < 1.1, f"std not ~1: {out.std().item()}"


def test_flatten_consecutive_shape():
    """FlattenConsecutive folds adjacent positions correctly."""
    from makemore import FlattenConsecutive
    layer = FlattenConsecutive(n=2)
    x = torch.randn(4, 8, 16)
    out = layer(x)
    assert out.shape == (4, 4, 32), f"unexpected shape: {out.shape}"
    # When time dim collapses to 1, it should be squeezed.
    layer2 = FlattenConsecutive(n=4)
    x2 = torch.randn(4, 4, 16)
    out2 = layer2(x2)
    assert out2.shape == (4, 64), f"squeeze failed: {out2.shape}"


def test_embedding_lookup():
    """Embedding lookup matches one-hot @ W."""
    from makemore import Embedding
    torch.manual_seed(0)
    emb = Embedding(num_embeddings=10, embedding_dim=5)
    indices = torch.tensor([0, 3, 7])
    out_lookup = emb(indices)
    # Reference: one-hot @ W
    one_hot = F.one_hot(indices, num_classes=10).float()
    out_matmul = one_hot @ emb.weight
    assert torch.allclose(out_lookup, out_matmul), "Embedding ≠ one-hot @ W"


def test_sequential_propagates_training_mode_implicitly():
    """Sequential aggregates parameters from all sublayers."""
    from makemore import Linear, BatchNorm1d, Tanh, Sequential
    m = Sequential([
        Linear(10, 20, bias=False),
        BatchNorm1d(20),
        Tanh(),
        Linear(20, 5),
    ])
    params = m.parameters()
    # Linear(no bias) + BN(gamma,beta) + Tanh() + Linear(w,b) = 1 + 2 + 0 + 2 = 5 tensors
    assert len(params) == 5


def test_full_training_smoke():
    """A handful of training steps reduces the loss."""
    from pathlib import Path
    from makemore.data import load_names, build_vocab, split_dataset
    from makemore.train import train, evaluate
    from models.mlp_batchnorm import MLPBatchNorm
    # Resolve names.txt relative to this test file, not cwd — pytest may be
    # invoked from any directory.
    names_path = Path(__file__).parent.parent / "data" / "names.txt"
    words = load_names(str(names_path))
    stoi, _ = build_vocab(words)
    Xtr, Ytr, Xdev, Ydev, _, _ = split_dataset(words, stoi, block_size=3)
    m = MLPBatchNorm(vocab_size=27, block_size=3, n_embd=10, n_hidden=64)
    before = evaluate(m, Xdev[:500], Ydev[:500])
    train(m, Xtr, Ytr, steps=500, batch_size=32, lr_schedule=lambda s: 0.1)
    after = evaluate(m, Xdev[:500], Ydev[:500])
    assert after < before, f"training did not reduce loss: {before:.3f} → {after:.3f}"
