"""Small shared helpers for the *Attention, Explained* course.

Everything here is deliberately short and readable: the notebooks re-derive each
idea step by step, and this module only keeps the plotting and the toy corpus in
one place so every notebook draws figures the same way.
"""
from __future__ import annotations

import os
import random
import re

import numpy as np
import matplotlib.pyplot as plt

DOCS_DIR = "docs"


# --------------------------------------------------------------------------- setup
def set_docs_dir(path: str) -> None:
    """Folder where `save_fig` writes the course figures (created if missing)."""
    global DOCS_DIR
    DOCS_DIR = path
    os.makedirs(DOCS_DIR, exist_ok=True)


def save_fig(fig, name: str, dpi: int = 140) -> str:
    """Save a figure into docs/ so the README always shows what the notebooks computed."""
    os.makedirs(DOCS_DIR, exist_ok=True)
    path = os.path.join(DOCS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def set_seed(seed: int = 0) -> None:
    """Make numpy, random and (if installed) torch reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch is optional for the NumPy-only notebooks
        pass


# ------------------------------------------------------------------ attention core
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exponentiating)."""
    x = np.asarray(x, dtype=float)
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V  -> (output, weights).

    `mask` is a boolean array broadcastable to the score matrix; True means
    "this query may NOT look at this key" (the score is set to -inf).
    """
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    d_k = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def causal_mask(n: int) -> np.ndarray:
    """Boolean (n, n) mask, True above the diagonal = the future is hidden."""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def sinusoidal_positional_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """The fixed sine/cosine encoding from 'Attention Is All You Need' (2017)."""
    pos = np.arange(n_positions)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000.0, (2 * (i // 2)) / d_model)
    pe = np.zeros((n_positions, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


# ------------------------------------------------------------------------ plotting
def plot_attention(weights, query_labels, key_labels, ax=None, title=None,
                   annotate=True, cmap="Blues", fmt="{:.2f}", vmax=None):
    """Draw an attention matrix: rows = queries (who is looking), columns = keys."""
    weights = np.asarray(weights)
    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(key_labels) + 2, 0.5 * len(query_labels) + 1.5))
    top = vmax if vmax is not None else max(weights.max(), 1e-9)
    im = ax.imshow(weights, cmap=cmap, vmin=0, vmax=top)
    ax.set_xticks(range(len(key_labels)))
    ax.set_xticklabels(key_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(query_labels)))
    ax.set_yticklabels(query_labels)
    ax.set_xlabel("key (attended to)")
    ax.set_ylabel("query (attending)")
    if annotate and weights.size <= 400:
        thr = top * 0.6
        for r in range(weights.shape[0]):
            for c in range(weights.shape[1]):
                ax.text(c, r, fmt.format(weights[r, c]), ha="center", va="center", fontsize=7,
                        color="white" if weights[r, c] > thr else "black")
    if title:
        ax.set_title(title)
    return im


# ------------------------------------------------------------- toy text & vectors
TOY_CORPUS = [
    "the king rules the kingdom", "the queen rules the kingdom",
    "the king wears a crown", "the queen wears a crown",
    "the prince is the son of the king", "the princess is the daughter of the queen",
    "a man walks in the city", "a woman walks in the city",
    "the man is a father", "the woman is a mother",
    "the boy plays with the dog", "the girl plays with the cat",
    "the dog chases the cat", "the cat chases the mouse",
    "a dog is a loyal animal", "a cat is a quiet animal",
    "the horse runs in the field", "the cow eats grass in the field",
    "the dog eats meat", "the cat eats fish", "the mouse eats cheese",
    "i drink coffee in the morning", "i drink tea in the morning",
    "she drinks coffee with milk", "he drinks tea with sugar",
    "we eat bread and cheese", "they eat rice and fish",
    "the apple is a sweet fruit", "the banana is a sweet fruit",
    "the orange is a juicy fruit", "i eat an apple every day",
    "she eats a banana every morning", "paris is a city in france",
    "rome is a city in italy", "berlin is a city in germany",
    "madrid is a city in spain", "france and italy are countries",
    "germany and spain are countries", "the car drives on the road",
    "the bus drives on the road", "the train runs on the track",
    "a bike is faster than walking", "the car is faster than the bike",
    "the king and the queen live in the castle",
    "the man and the woman live in the house",
]


def tokenize(text: str) -> list[str]:
    """A deliberately simple word tokenizer: lowercase, keep letters/digits/apostrophes."""
    return re.findall(r"[a-z0-9']+", text.lower())


def cooccurrence_embeddings(corpus=TOY_CORPUS, window: int = 3, dim: int = 16):
    """Learn tiny word vectors from co-occurrence counts (PPMI + truncated SVD).

    This is the classic count-based route to embeddings (the same idea GloVe
    builds on). It needs no download and runs in milliseconds.
    Returns (vocab list, {word: index}, matrix of shape (len(vocab), dim)).
    """
    sents = [tokenize(s) for s in corpus]
    vocab = sorted({w for s in sents for w in s})
    w2i = {w: i for i, w in enumerate(vocab)}
    C = np.zeros((len(vocab), len(vocab)))
    for s in sents:
        for i, w in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if i != j:
                    C[w2i[w], w2i[s[j]]] += 1.0 / abs(i - j)
    total = C.sum()
    row = C.sum(1, keepdims=True)
    col = C.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C * total) / (row * col))
    ppmi = np.nan_to_num(np.maximum(pmi, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    U, S, _ = np.linalg.svd(ppmi)
    E = U[:, :dim] * np.sqrt(S[:dim])
    return vocab, w2i, E


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
