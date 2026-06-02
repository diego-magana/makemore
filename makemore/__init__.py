"""makemore — character-level language modeling progression with probing analysis."""

from makemore.layers import (
    Linear, BatchNorm1d, Tanh, Embedding, FlattenConsecutive, Sequential,
)
from makemore.data import load_names, build_vocab, build_dataset, split_dataset
from makemore.train import train, evaluate, generate_names, set_layer_training

__all__ = [
    # Layers
    "Linear", "BatchNorm1d", "Tanh", "Embedding", "FlattenConsecutive", "Sequential",
    # Data
    "load_names", "build_vocab", "build_dataset", "split_dataset",
    # Training
    "train", "evaluate", "generate_names", "set_layer_training",
]
