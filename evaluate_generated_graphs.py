#!/usr/bin/env python3
"""
Evaluate saved generated graphs with:
1) overlap/auc on fixed reference pairs (same as before)
2) stronger GNN link prediction on the generated graph itself, then evaluate on the same fixed reference pairs.

This version keeps the same graph_path + dataset interface, but upgrades the LP model/training to a
more standard GAE-style protocol:
- build train/val splits from each generated graph
- train on TRAIN adjacency only
- use validation AUC + early stopping
- optionally search over a small hyperparameter grid
- evaluate the trained model on fixed reference pairs for AUC/SP/score-SP/EO

Outputs are CSV-only.
"""

import argparse
import csv
import itertools
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.utils import degree, negative_sampling, remove_self_loops, to_undirected


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


def safe_std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanstd(arr))


def safe_group_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values)
    mask = np.asarray(mask, dtype=bool)
    if values.size == 0 or mask.size == 0 or mask.sum() == 0:
        return float("nan")
    return float(values[mask].mean())


def safe_diff(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a - b)


def safe_abs(a: float) -> float:
    if not np.isfinite(a):
        return float("nan")
    return float(abs(a))


def unique_undirected_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    edge_index, _ = remove_self_loops(edge_index)
    row, col = edge_index
    mask = row < col
    return torch.stack([row[mask], col[mask]], dim=0)


def ensure_features(data: Data) -> Data:
    if getattr(data, "x", None) is None or data.x is None:
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes, dtype=torch.float).view(-1, 1)
        data.x = deg
    else:
        data.x = data.x.float()
    return data


def global_id_mapping(data: Data) -> Tuple[List[int], Dict[int, int]]:
    if hasattr(data, "orig_id") and data.orig_id is not None:
        gids = [int(v) for v in data.orig_id.cpu().numpy().tolist()]
    else:
        gids = list(range(data.num_nodes))
    g2l = {g: i for i, g in enumerate(gids)}
    return gids, g2l


def write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def add_compat_metric_aliases(row: Dict[str, float]) -> None:
    """
    Add EDGE-style compatibility aliases so FairWire outputs can be compared
    against the EDGE fairness grid CSVs with minimal post-processing.
    """
    alias_pairs = [
        ("overlap/auc", "value/linkpred_auc"),
        ("lp/score_sp_gap", "fair_gap"),
        ("lp/score_sp_abs_gap", "fair_abs_gap"),
        ("lp/score_sp_gap", "value/fair_gap"),
        ("lp/score_sp_abs_gap", "value/fair_abs_gap"),
    ]
    for src, dst in alias_pairs:
        if src in row and dst not in row:
            row[dst] = float(row[src])


# -----------------------------
# Graph loading
# -----------------------------

def load_saved_graphs(graph_path: str) -> List[Data]:
    obj = torch.load(graph_path, map_location="cpu")
    graphs = obj if isinstance(obj, list) else [obj]
    if not graphs:
        raise ValueError(f"No graphs found in {graph_path}")
    return graphs


def get_local_sensitive_vector(data: Data, sensitive_attr: str, sensitive_value: Optional[int]) -> torch.Tensor:
    if not hasattr(data, sensitive_attr):
        raise ValueError(f"data has no attribute {sensitive_attr!r}")
    s = getattr(data, sensitive_attr)
    if s is None:
        raise ValueError(f"data.{sensitive_attr} is None")
    if s.dim() > 1:
        s = s.squeeze()
    if sensitive_value is None:
        if s.dtype == torch.bool:
            return s.long()
        return (s != 0).long()
    return (s == sensitive_value).long()


def pair_sensitive_mask_from_local_pairs(
    pairs: Sequence[Tuple[int, int]],
    local_sensitive: torch.Tensor,
    mode: str = "either",
) -> np.ndarray:
    local_sensitive = local_sensitive.detach().cpu().bool()
    mask = []
    for u, v in pairs:
        su = bool(local_sensitive[int(u)].item())
        sv = bool(local_sensitive[int(v)].item())
        if mode == "both":
            mask.append(su and sv)
        else:
            mask.append(su or sv)
    return np.asarray(mask, dtype=bool)


# -----------------------------
# Reference graph / fixed pairs
# -----------------------------

def _candidate_reference_paths(graph_path: Path, dataset: str) -> List[Path]:
    candidates: List[Path] = []
    names = [f"{dataset}_feat.pkl", f"{dataset}.pkl"]

    def add_candidates_from_root(root: Path) -> None:
        for name in names:
            candidates.append(root / "graphs" / name)
            candidates.append(root / name)

    gp = graph_path.resolve()
    for root in [gp.parent, *gp.parent.parents]:
        add_candidates_from_root(root)

    cwd = Path.cwd().resolve()
    for root in [cwd, *cwd.parents]:
        add_candidates_from_root(root)

    dedup: List[Path] = []
    seen = set()
    for c in candidates:
        if str(c) not in seen:
            seen.add(str(c))
            dedup.append(c)
    return dedup


def find_reference_graph_path(graph_path: str, dataset: str) -> Path:
    gp = Path(graph_path)
    candidates = _candidate_reference_paths(gp, dataset)
    for c in candidates:
        if c.exists():
            return c
    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find reference graph for dataset={dataset!r}.\n"
        f"Searched these candidate paths:\n{searched}\n"
        f"Expected something like graphs/{dataset}_feat.pkl near graph_path or current working directory."
    )


def load_reference_graph_from_dataset(graph_path: str, dataset: str) -> nx.Graph:
    ref_path = find_reference_graph_path(graph_path, dataset)
    with ref_path.open("rb") as f:
        g_ref = pickle.load(f)
    if not isinstance(g_ref, nx.Graph):
        raise TypeError(f"Reference object at {ref_path} is not a networkx.Graph")
    return g_ref


def build_reference_node_sensitive_map(g_ref: nx.Graph, sensitive_attr: str, sensitive_value: Optional[int]) -> Dict[int, bool]:
    out: Dict[int, bool] = {}
    for n, attrs in g_ref.nodes(data=True):
        if sensitive_attr not in attrs:
            raise KeyError(f"Reference graph node {n} lacks attr {sensitive_attr!r}")
        val = attrs[sensitive_attr]
        if hasattr(val, "item"):
            val = val.item()
        if sensitive_value is None:
            out[int(n)] = bool(val)
        else:
            out[int(n)] = bool(val == sensitive_value)
    return out


def build_fixed_eval_pairs(g_ref: nx.Graph, max_pos_edges: int = 20000, neg_ratio: float = 1.0, seed: int = 0) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    rng = random.Random(seed)
    nodes = [int(n) for n in g_ref.nodes()]

    pos_edges_all: List[Tuple[int, int]] = []
    edge_set = set()
    for u, v in g_ref.edges():
        if u == v:
            continue
        e = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        if e in edge_set:
            continue
        edge_set.add(e)
        pos_edges_all.append(e)

    if max_pos_edges and len(pos_edges_all) > max_pos_edges:
        pos_edges = rng.sample(pos_edges_all, max_pos_edges)
    else:
        pos_edges = pos_edges_all

    num_neg = int(len(pos_edges) * neg_ratio)
    neg_edges: List[Tuple[int, int]] = []
    neg_set = set()
    while len(neg_edges) < num_neg:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in edge_set or e in neg_set:
            continue
        neg_set.add(e)
        neg_edges.append(e)

    pairs = pos_edges + neg_edges
    labels = np.concatenate([
        np.ones(len(pos_edges), dtype=np.int64),
        np.zeros(len(neg_edges), dtype=np.int64),
    ])
    return pairs, labels


def pair_sensitive_mask(pairs: Sequence[Tuple[int, int]], node_sensitive: Dict[int, bool], mode: str = "either") -> np.ndarray:
    mask = []
    for u, v in pairs:
        su = node_sensitive[int(u)]
        sv = node_sensitive[int(v)]
        if mode == "both":
            mask.append(su and sv)
        else:
            mask.append(su or sv)
    return np.asarray(mask, dtype=bool)


# -----------------------------
# Overlap AUC on fixed pairs
# -----------------------------

def edge_overlap_on_fixed_pairs(
    data: Data,
    reference_pairs: Sequence[Tuple[int, int]],
    reference_labels: np.ndarray,
    reference_node_sensitive: Dict[int, bool],
    edge_sensitive_mode: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    _gids, g2l = global_id_mapping(data)
    edge_index = unique_undirected_edge_index(data.edge_index.cpu())
    edge_set = {tuple(sorted((int(u), int(v)))) for u, v in edge_index.t().tolist()}

    keep_idx: List[int] = []
    scores: List[float] = []
    kept_pairs: List[Tuple[int, int]] = []

    for i, (u, v) in enumerate(reference_pairs):
        if u in g2l and v in g2l and u != v:
            keep_idx.append(i)
            lu, lv = g2l[u], g2l[v]
            scores.append(float((min(lu, lv), max(lu, lv)) in edge_set))
            kept_pairs.append((u, v))

    if not keep_idx:
        raise RuntimeError("No reference pairs mapped to this generated graph. Check orig_id alignment.")

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    labels = reference_labels[keep_idx_arr]
    scores_arr = np.asarray(scores, dtype=np.float32)
    sens_mask = pair_sensitive_mask(kept_pairs, reference_node_sensitive, mode=edge_sensitive_mode)

    metrics = {
        "overlap/auc": safe_auc(labels, scores_arr),
        "overlap/num_eval_pairs": float(len(keep_idx)),
        "overlap/edge_presence_rate": float(scores_arr.mean()) if scores_arr.size else float("nan"),
    }
    raw = {
        "keep_idx": keep_idx_arr,
        "labels": labels,
        "scores": scores_arr,
        "sens_mask": sens_mask.astype(bool),
    }
    return metrics, raw


# -----------------------------
# GAE-style LP model
# -----------------------------

class BaseEncoder(nn.Module):
    def encode(self, x, edge_index):
        raise NotImplementedError


class StackedGCNEncoder(BaseEncoder):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GCNConv(in_dim, out_dim))
        else:
            self.layers.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layers.append(GCNConv(hidden_dim, out_dim))
        self.dropout = dropout

    def encode(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class StackedSAGEEncoder(BaseEncoder):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int = 2):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim))
        else:
            self.layers.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))
            self.layers.append(SAGEConv(hidden_dim, out_dim))
        self.dropout = dropout

    def encode(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class StackedGATEncoder(BaseEncoder):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int = 2, heads: int = 4):
        super().__init__()
        assert num_layers >= 1
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(GATConv(in_dim, out_dim, heads=1, concat=False, dropout=dropout))
        else:
            self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout))
            hidden_in = hidden_dim * heads
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_in, hidden_dim, heads=heads, concat=True, dropout=dropout))
                hidden_in = hidden_dim * heads
            self.layers.append(GATConv(hidden_in, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def encode(self, x, edge_index):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if i != len(self.layers) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class DotProductGAE(nn.Module):
    def __init__(self, encoder: BaseEncoder):
        super().__init__()
        self.encoder = encoder

    def encode(self, x, edge_index):
        return self.encoder.encode(x, edge_index)

    @staticmethod
    def decode(z, edge_index):
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


@dataclass
class LPTrialConfig:
    backbone: str
    num_layers: int
    hidden_dim: int
    out_dim: int
    dropout: float
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int
    device: str
    gat_heads: int


def build_encoder(in_dim: int, cfg: LPTrialConfig) -> BaseEncoder:
    if cfg.backbone == "gcn":
        return StackedGCNEncoder(in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout, num_layers=cfg.num_layers)
    if cfg.backbone == "sage":
        return StackedSAGEEncoder(in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout, num_layers=cfg.num_layers)
    if cfg.backbone == "gat":
        return StackedGATEncoder(in_dim, cfg.hidden_dim, cfg.out_dim, cfg.dropout, num_layers=cfg.num_layers, heads=cfg.gat_heads)
    raise ValueError(f"Unknown backbone: {cfg.backbone}")


def build_model(in_dim: int, cfg: LPTrialConfig) -> DotProductGAE:
    return DotProductGAE(build_encoder(in_dim, cfg))


# -----------------------------
# Generated-graph splits (GAE-style)
# -----------------------------

def split_positive_edges(pos_edge_index: torch.Tensor, holdout_ratio: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pos = pos_edge_index.size(1)
    if num_pos < 3:
        raise ValueError("Not enough positive edges to create a split.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_pos, generator=g)

    num_holdout = max(1, int(math.ceil(num_pos * holdout_ratio)))
    num_holdout = min(num_holdout, num_pos - 1)

    holdout_pos = pos_edge_index[:, perm[:num_holdout]]
    remain_pos = pos_edge_index[:, perm[num_holdout:]]
    return remain_pos, holdout_pos

def sample_true_negative_edges(num_nodes: int, pos_edge_index: torch.Tensor, num_samples: int, seed: int = 0) -> torch.Tensor:
    rng = random.Random(seed)
    pos_set = {tuple(sorted((int(u), int(v)))) for u, v in pos_edge_index.t().tolist()}
    neg_set = set()

    max_possible = num_nodes * (num_nodes - 1) // 2 - len(pos_set)
    if num_samples > max_possible:
        raise ValueError(f"Requested {num_samples} negatives, but only {max_possible} true non-edges exist.")

    while len(neg_set) < num_samples:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            continue
        e = (u, v) if u < v else (v, u)
        if e in pos_set or e in neg_set:
            continue
        neg_set.add(e)
    return torch.tensor(sorted(list(neg_set)), dtype=torch.long).t().contiguous()


def build_generated_graph_train_test_split(
    data: Data,
    test_ratio: float = 0.2,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    data = ensure_features(data)
    full_pos = unique_undirected_edge_index(data.edge_index.cpu())
    train_pos, test_pos = split_positive_edges(full_pos, holdout_ratio=test_ratio, seed=seed)
    test_neg = sample_true_negative_edges(
        num_nodes=data.num_nodes,
        pos_edge_index=full_pos,
        num_samples=test_pos.size(1),
        seed=seed + 1,
    )
    return {
        "full_pos": full_pos,
        "train_pos": train_pos,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "train_mp_edge_index": to_undirected(train_pos),
        "full_mp_edge_index": to_undirected(full_pos),
    }


def build_generated_graph_val_split(
    data: Data,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    data = ensure_features(data)
    full_pos = unique_undirected_edge_index(data.edge_index.cpu())
    num_pos = full_pos.size(1)
    if num_pos < 3:
        raise ValueError("Not enough positive edges to create train/val split.")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_pos, generator=g)

    num_val = max(1, int(math.ceil(num_pos * val_ratio)))
    num_val = min(num_val, num_pos - 1)

    val_pos = full_pos[:, perm[:num_val]]
    train_pos = full_pos[:, perm[num_val:]]

    val_neg = sample_true_negative_edges(
        num_nodes=data.num_nodes,
        pos_edge_index=full_pos,
        num_samples=val_pos.size(1),
        seed=seed + 1,
    )
    train_mp_edge_index = to_undirected(train_pos)
    full_mp_edge_index = to_undirected(full_pos)

    return {
        "full_pos": full_pos,
        "train_pos": train_pos,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "train_mp_edge_index": train_mp_edge_index,
        "full_mp_edge_index": full_mp_edge_index,
    }


def build_inner_train_val_split(
    train_pos: torch.Tensor,
    full_pos: torch.Tensor,
    num_nodes: int,
    val_ratio: float = 0.2,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    inner_train_pos, val_pos = split_positive_edges(train_pos, holdout_ratio=val_ratio, seed=seed)
    val_neg = sample_true_negative_edges(
        num_nodes=num_nodes,
        pos_edge_index=full_pos,
        num_samples=val_pos.size(1),
        seed=seed + 1,
    )
    return {
        "full_pos": full_pos,
        "train_pos": inner_train_pos,
        "val_pos": val_pos,
        "val_neg": val_neg,
        "train_mp_edge_index": to_undirected(inner_train_pos),
        "full_mp_edge_index": to_undirected(full_pos),
    }


def evaluate_pairs_auc(model: DotProductGAE, x: torch.Tensor, mp_edge_index: torch.Tensor, pos_edges: torch.Tensor, neg_edges: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        z = model.encode(x, mp_edge_index)
        pos_logits = model.decode(z, pos_edges)
        neg_logits = model.decode(z, neg_edges)
        probs = torch.sigmoid(torch.cat([pos_logits, neg_logits], dim=0)).cpu().numpy()
        labels = np.concatenate([
            np.ones(pos_edges.size(1), dtype=np.int64),
            np.zeros(neg_edges.size(1), dtype=np.int64),
        ])
    return safe_auc(labels, probs)


def sample_train_negatives(full_mp_edge_index: torch.Tensor, num_nodes: int, num_neg_samples: int) -> torch.Tensor:
    return negative_sampling(
        edge_index=full_mp_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    )


def build_search_space(args) -> List[LPTrialConfig]:
    hidden_dims = args.lp_search_hidden_dims if args.lp_search else [args.lp_hidden_dim]
    lrs = args.lp_search_lrs if args.lp_search else [args.lp_lr]
    dropouts = args.lp_search_dropouts if args.lp_search else [args.lp_dropout]
    num_layers_list = args.lp_search_num_layers if args.lp_search else [args.lp_num_layers]

    trials: List[LPTrialConfig] = []
    for num_layers, hidden_dim, lr, dropout in itertools.product(num_layers_list, hidden_dims, lrs, dropouts):
        trials.append(LPTrialConfig(
            backbone=args.lp_model,
            num_layers=int(num_layers),
            hidden_dim=int(hidden_dim),
            out_dim=int(args.lp_out_dim),
            dropout=float(dropout),
            lr=float(lr),
            weight_decay=float(args.lp_weight_decay),
            epochs=int(args.lp_epochs),
            patience=int(args.lp_patience),
            batch_size=int(args.lp_batch_size),
            device=args.device,
            gat_heads=int(args.gat_heads),
        ))
    return trials


def train_gae_with_validation(data: Data, split: Dict[str, torch.Tensor], cfg: LPTrialConfig, seed: int) -> Tuple[DotProductGAE, Dict[str, float]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    data = ensure_features(data)

    x = data.x.to(device)
    train_mp_edge_index = split["train_mp_edge_index"].to(device)
    full_mp_edge_index = split["full_mp_edge_index"].to(device)
    train_pos = split["train_pos"].to(device)
    val_pos = split["val_pos"].to(device)
    val_neg = split["val_neg"].to(device)

    model = build_model(x.size(-1), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val_auc = -1.0
    best_epoch = 0
    bad_epochs = 0

    num_train_pos = train_pos.size(1)
    batch_size = min(cfg.batch_size, num_train_pos)
    g = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        z = model.encode(x, train_mp_edge_index)

        if batch_size < num_train_pos:
            perm = torch.randperm(num_train_pos, generator=g, device=device)[:batch_size]
            batch_pos = train_pos[:, perm]
        else:
            batch_pos = train_pos

        batch_neg = sample_train_negatives(full_mp_edge_index, data.num_nodes, batch_pos.size(1)).to(device)

        pos_logits = model.decode(z, batch_pos)
        neg_logits = model.decode(z, batch_neg)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        val_auc = evaluate_pairs_auc(model, x, train_mp_edge_index, val_pos, val_neg)
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    meta = {
        "best_val_auc": float(best_val_auc) if np.isfinite(best_val_auc) else float("nan"),
        "best_epoch": float(best_epoch),
        "num_layers": float(cfg.num_layers),
        "hidden_dim": float(cfg.hidden_dim),
        "dropout": float(cfg.dropout),
        "lr": float(cfg.lr),
    }
    return model, meta


def choose_best_lp_model(data: Data, split: Dict[str, torch.Tensor], args, seed: int) -> Tuple[DotProductGAE, Dict[str, float]]:
    search_space = build_search_space(args)
    best_model = None
    best_meta: Dict[str, float] = {}
    best_val_auc = -1.0

    for i, cfg in enumerate(search_space):
        trial_seed = seed + 1000 * i
        model, meta = train_gae_with_validation(data, split, cfg, seed=trial_seed)
        val_auc = meta.get("best_val_auc", float("nan"))
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_model = model
            best_meta = {
                "lp_protocol": "gae_val",
                "lp_model": cfg.backbone,
                **meta,
            }

    if best_model is None:
        raise RuntimeError("Failed to train any LP model with finite validation AUC.")
    return best_model, best_meta


def train_gae_on_train_split(data: Data, split: Dict[str, torch.Tensor], cfg: LPTrialConfig, seed: int) -> Tuple[DotProductGAE, Dict[str, float]]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    data = ensure_features(data)

    x = data.x.to(device)
    train_mp_edge_index = split["train_mp_edge_index"].to(device)
    full_mp_edge_index = split["full_mp_edge_index"].to(device)
    train_pos = split["train_pos"].to(device)

    model = build_model(x.size(-1), cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_train_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    num_train_pos = train_pos.size(1)
    batch_size = min(cfg.batch_size, num_train_pos)
    g = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        z = model.encode(x, train_mp_edge_index)

        if batch_size < num_train_pos:
            perm = torch.randperm(num_train_pos, generator=g, device=device)[:batch_size]
            batch_pos = train_pos[:, perm]
        else:
            batch_pos = train_pos

        batch_neg = sample_train_negatives(full_mp_edge_index, data.num_nodes, batch_pos.size(1)).to(device)

        pos_logits = model.decode(z, batch_pos)
        neg_logits = model.decode(z, batch_neg)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        loss_value = float(loss.detach().cpu().item())
        if loss_value < best_train_loss:
            best_train_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best_state)
    meta = {
        "best_val_auc": float("nan"),
        "best_train_loss": float(best_train_loss),
        "best_epoch": float(best_epoch),
        "num_layers": float(cfg.num_layers),
        "hidden_dim": float(cfg.hidden_dim),
        "dropout": float(cfg.dropout),
        "lr": float(cfg.lr),
    }
    return model, meta


def choose_best_lp_config_on_train_split(data: Data, split: Dict[str, torch.Tensor], args, seed: int) -> Tuple[LPTrialConfig, Dict[str, float]]:
    search_space = build_search_space(args)
    best_cfg = None
    best_meta: Dict[str, float] = {}
    best_val_auc = -1.0

    for i, cfg in enumerate(search_space):
        trial_seed = seed + 1000 * i
        model, meta = train_gae_with_validation(data=data, split=split, cfg=cfg, seed=trial_seed)
        val_auc = meta.get("best_val_auc", float("nan"))
        if np.isfinite(val_auc) and val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_cfg = cfg
            best_meta = {
                "lp_protocol": "gae_train_test",
                "lp_model": cfg.backbone,
                **meta,
            }

    if best_cfg is None:
        raise RuntimeError("Failed to pick an LP config with finite inner validation AUC.")
    return best_cfg, best_meta


def train_lp_for_generated_graph(data: Data, split: Dict[str, torch.Tensor], args, seed: int) -> Tuple[DotProductGAE, Dict[str, float]]:
    search_data = ensure_features(data)
    if args.lp_search:
        inner_split = build_inner_train_val_split(
            train_pos=split["train_pos"],
            full_pos=split["full_pos"],
            num_nodes=search_data.num_nodes,
            val_ratio=args.lp_val_ratio,
            seed=seed,
        )
        cfg, meta = choose_best_lp_config_on_train_split(data=search_data, split=inner_split, args=args, seed=seed)
    else:
        cfg = build_search_space(args)[0]
        meta = {
            "lp_protocol": "gae_train_test",
            "lp_model": cfg.backbone,
            "best_val_auc": float("nan"),
        }

    model, train_meta = train_gae_on_train_split(data=search_data, split=split, cfg=cfg, seed=seed + 17)
    meta.update(train_meta)
    return model, meta


def compute_binary_and_score_fairness(probs: np.ndarray, labels: np.ndarray, sens_mask: np.ndarray, threshold: float) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    sens_mask = np.asarray(sens_mask, dtype=bool)

    hard_pred = (probs >= threshold).astype(np.float32)
    score_mean_sensitive = safe_group_mean(probs, sens_mask)
    score_mean_nonsensitive = safe_group_mean(probs, ~sens_mask)
    score_gap = safe_diff(score_mean_sensitive, score_mean_nonsensitive)

    hard_rate_sensitive = safe_group_mean(hard_pred, sens_mask)
    hard_rate_nonsensitive = safe_group_mean(hard_pred, ~sens_mask)
    hard_gap = safe_diff(hard_rate_sensitive, hard_rate_nonsensitive)

    pos_mask = labels == 1
    eo_sensitive = safe_group_mean(hard_pred[pos_mask], sens_mask[pos_mask]) if pos_mask.any() else float("nan")
    eo_nonsensitive = safe_group_mean(hard_pred[pos_mask], ~sens_mask[pos_mask]) if pos_mask.any() else float("nan")
    eo_gap = safe_diff(eo_sensitive, eo_nonsensitive)

    return {
        "auc": safe_auc(labels, probs),
        "score_sp_gap": score_gap,
        "score_sp_abs_gap": safe_abs(score_gap),
        "sp_gap": hard_gap,
        "sp_abs_gap": safe_abs(hard_gap),
        "eo_gap": eo_gap,
        "eo_abs_gap": safe_abs(eo_gap),
        "score_mean_sensitive": score_mean_sensitive,
        "score_mean_nonsensitive": score_mean_nonsensitive,
        "hard_rate_sensitive": hard_rate_sensitive,
        "hard_rate_nonsensitive": hard_rate_nonsensitive,
        "num_eval_pairs": float(len(labels)),
    }


def evaluate_lp_on_generated_test_pairs(
    data: Data,
    model: DotProductGAE,
    train_mp_edge_index: torch.Tensor,
    test_pos: torch.Tensor,
    test_neg: torch.Tensor,
    sensitive_attr: str,
    sensitive_value: Optional[int],
    edge_sensitive_mode: str,
    threshold: float,
    device: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    data = ensure_features(data)
    x = data.x.to(device_t)
    train_mp_edge_index = train_mp_edge_index.to(device_t)

    test_pos = test_pos.cpu()
    test_neg = test_neg.cpu()
    pair_edge_index = torch.cat([test_pos, test_neg], dim=1)
    labels = np.concatenate([
        np.ones(test_pos.size(1), dtype=np.int64),
        np.zeros(test_neg.size(1), dtype=np.int64),
    ])

    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_mp_edge_index)
        logits = model.decode(z, pair_edge_index.to(device_t))
        probs = torch.sigmoid(logits).cpu().numpy()

    local_sensitive = get_local_sensitive_vector(data, sensitive_attr=sensitive_attr, sensitive_value=sensitive_value)
    local_pairs = [tuple(map(int, e)) for e in pair_edge_index.t().tolist()]
    sens_mask = pair_sensitive_mask_from_local_pairs(local_pairs, local_sensitive, mode=edge_sensitive_mode)

    fair = compute_binary_and_score_fairness(probs, labels, sens_mask, threshold)
    metrics = {f"lp/{k}": float(v) for k, v in fair.items()}
    raw = {
        "keep_idx": np.arange(len(labels), dtype=np.int64),
        "labels": labels,
        "scores": probs,
        "sens_mask": sens_mask.astype(bool),
    }
    return metrics, raw


def evaluate_lp_on_fixed_pairs(
    data: Data,
    model: DotProductGAE,
    train_mp_edge_index: torch.Tensor,
    reference_pairs: Sequence[Tuple[int, int]],
    reference_labels: np.ndarray,
    reference_node_sensitive: Dict[int, bool],
    edge_sensitive_mode: str,
    threshold: float,
    device: str,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    device_t = torch.device(device if torch.cuda.is_available() else "cpu")
    data = ensure_features(data)
    x = data.x.to(device_t)
    train_mp_edge_index = train_mp_edge_index.to(device_t)

    _gids, g2l = global_id_mapping(data)
    local_pairs: List[Tuple[int, int]] = []
    keep_idx: List[int] = []
    for i, (u, v) in enumerate(reference_pairs):
        if u in g2l and v in g2l and u != v:
            local_pairs.append((g2l[u], g2l[v]))
            keep_idx.append(i)

    if not local_pairs:
        raise RuntimeError("No reference pairs mapped to this generated graph. Check orig_id alignment.")

    pair_edge_index = torch.tensor(local_pairs, dtype=torch.long, device=device_t).t().contiguous()
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_mp_edge_index)
        logits = model.decode(z, pair_edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    labels = reference_labels[keep_idx_arr]
    kept_pairs = [reference_pairs[i] for i in keep_idx]
    sens_mask = pair_sensitive_mask(kept_pairs, reference_node_sensitive, mode=edge_sensitive_mode)

    fair = compute_binary_and_score_fairness(probs, labels, sens_mask, threshold)
    metrics = {f"lp/{k}": float(v) for k, v in fair.items()}
    raw = {
        "keep_idx": keep_idx_arr,
        "labels": labels,
        "scores": probs,
        "sens_mask": sens_mask.astype(bool),
    }
    return metrics, raw


# -----------------------------
# Ensemble summaries
# -----------------------------

def ensemble_mean_scores(raw_results: List[Dict[str, np.ndarray]], full_num_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    score_sums = np.zeros(full_num_pairs, dtype=np.float64)
    score_counts = np.zeros(full_num_pairs, dtype=np.int64)
    for r in raw_results:
        idx = r["keep_idx"]
        score_sums[idx] += r["scores"]
        score_counts[idx] += 1
    valid = score_counts > 0
    mean_scores = np.zeros(full_num_pairs, dtype=np.float64)
    mean_scores[valid] = score_sums[valid] / score_counts[valid]
    return mean_scores, valid


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved generated graphs with overlap AUC and stronger GAE-style LP, save to CSV.")
    p.add_argument("--graph_path", type=str, required=True, help="Path to *.pyg.pt or *.pyg_full.pt")
    p.add_argument("--dataset", type=str, required=True, help="Dataset name used to find graphs/{dataset}_feat.pkl")
    p.add_argument("--graph_index", type=int, default=None, help="Optional: evaluate only one graph from the saved list")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--sensitive_attr", type=str, default="y")
    p.add_argument("--sensitive_value", type=int, default=3)
    p.add_argument("--edge_sensitive_mode", type=str, default="either", choices=["either", "both"])

    p.add_argument("--max_pos_edges", type=int, default=20000, help="Max positive reference edges for overlap/reference LP pairs")
    p.add_argument("--neg_ratio", type=float, default=1.0, help="Neg/pos ratio for fixed reference pairs")

    # LP / GAE protocol
    p.add_argument("--lp_model", "--backbone", dest="lp_model", type=str, default="gcn", choices=["gcn", "sage", "gat"])
    p.add_argument("--lp_num_layers", type=int, default=2)
    p.add_argument("--lp_hidden_dim", type=int, default=128)
    p.add_argument("--lp_out_dim", type=int, default=64)
    p.add_argument("--lp_dropout", type=float, default=0.1)
    p.add_argument("--lp_lr", type=float, default=1e-2)
    p.add_argument("--lp_weight_decay", type=float, default=0.0)
    p.add_argument("--lp_epochs", type=int, default=300)
    p.add_argument("--lp_patience", type=int, default=30)
    p.add_argument("--lp_batch_size", type=int, default=16384)
    p.add_argument("--lp_test_ratio", type=float, default=0.2, help="Hold-out edge ratio for generated-graph train/test LP evaluation")
    p.add_argument("--lp_val_ratio", type=float, default=0.2, help="Inner validation ratio used only when --lp_search is enabled")
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--threshold", type=float, default=0.5)

    # optional hyperparameter search
    p.add_argument("--lp_search", action="store_true", help="Search over a small validation grid instead of a single config")
    p.add_argument("--lp_search_hidden_dims", type=int, nargs="+", default=[64, 128])
    p.add_argument("--lp_search_lrs", type=float, nargs="+", default=[1e-2, 3e-3])
    p.add_argument("--lp_search_dropouts", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    p.add_argument("--lp_search_num_layers", type=int, nargs="+", default=[1, 2])

    p.add_argument("--out_per_graph_csv", type=str, default=None)
    p.add_argument("--out_summary_csv", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    graph_path = Path(args.graph_path)
    out_dir = graph_path.parent
    stem = graph_path.name[:-3] if graph_path.name.endswith(".pt") else graph_path.name

    if args.out_per_graph_csv is None:
        args.out_per_graph_csv = str(out_dir / f"{stem}.overlap_lp_gae_per_graph.csv")
    if args.out_summary_csv is None:
        args.out_summary_csv = str(out_dir / f"{stem}.overlap_lp_gae_summary.csv")

    graphs = load_saved_graphs(args.graph_path)
    total_loaded = len(graphs)
    if args.graph_index is not None:
        if args.graph_index < 0 or args.graph_index >= total_loaded:
            raise IndexError(f"graph_index={args.graph_index} out of range for {total_loaded} graphs")
        graphs = [graphs[args.graph_index]]

    g_ref = load_reference_graph_from_dataset(args.graph_path, args.dataset)
    reference_pairs, reference_labels = build_fixed_eval_pairs(
        g_ref,
        max_pos_edges=args.max_pos_edges,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    reference_node_sensitive = build_reference_node_sensitive_map(
        g_ref,
        sensitive_attr=args.sensitive_attr,
        sensitive_value=args.sensitive_value,
    )

    per_graph_rows: List[Dict[str, float]] = []
    overlap_raw_rows: List[Dict[str, np.ndarray]] = []
    lp_raw_rows: List[Dict[str, np.ndarray]] = []

    for i, data in enumerate(graphs):
        original_idx = args.graph_index if args.graph_index is not None else i
        row: Dict[str, float] = {
            "graph_idx": float(original_idx),
            "lp/model": args.lp_model,
            "lp_protocol": "gae_train_test",
        }

        overlap_metrics, overlap_raw = edge_overlap_on_fixed_pairs(
            data=data,
            reference_pairs=reference_pairs,
            reference_labels=reference_labels,
            reference_node_sensitive=reference_node_sensitive,
            edge_sensitive_mode=args.edge_sensitive_mode,
        )
        row.update(overlap_metrics)
        overlap_raw_rows.append(overlap_raw)

        try:
            split = build_generated_graph_train_test_split(
                data=data,
                test_ratio=args.lp_test_ratio,
                seed=args.seed + int(original_idx),
            )
            model, meta = train_lp_for_generated_graph(
                data=data,
                split=split,
                args=args,
                seed=args.seed + 100 * int(original_idx),
            )
            row["lp/best_val_auc"] = float(meta.get("best_val_auc", float("nan")))
            row["lp/best_train_loss"] = float(meta.get("best_train_loss", float("nan")))
            row["lp/best_epoch"] = float(meta.get("best_epoch", float("nan")))
            row["lp/best_num_layers"] = float(meta.get("num_layers", float("nan")))
            row["lp/best_hidden_dim"] = float(meta.get("hidden_dim", float("nan")))
            row["lp/best_dropout"] = float(meta.get("dropout", float("nan")))
            row["lp/best_lr"] = float(meta.get("lr", float("nan")))
            row["lp/train_num_pos"] = float(split["train_pos"].size(1))
            row["lp/test_num_pos"] = float(split["test_pos"].size(1))
            row["lp/test_num_neg"] = float(split["test_neg"].size(1))

            lp_metrics, lp_raw = evaluate_lp_on_generated_test_pairs(
                data=data,
                model=model,
                train_mp_edge_index=split["train_mp_edge_index"],
                test_pos=split["test_pos"],
                test_neg=split["test_neg"],
                sensitive_attr=args.sensitive_attr,
                sensitive_value=args.sensitive_value,
                edge_sensitive_mode=args.edge_sensitive_mode,
                threshold=args.threshold,
                device=args.device,
            )
            row.update(lp_metrics)
            lp_raw_rows.append(lp_raw)
        except Exception as e:
            row["lp/error"] = str(e)
            for k in [
                "lp/auc",
                "lp/score_sp_gap",
                "lp/score_sp_abs_gap",
                "lp/sp_gap",
                "lp/sp_abs_gap",
                "lp/eo_gap",
                "lp/eo_abs_gap",
                "lp/score_mean_sensitive",
                "lp/score_mean_nonsensitive",
                "lp/hard_rate_sensitive",
                "lp/hard_rate_nonsensitive",
                "lp/num_eval_pairs",
                "lp/best_val_auc",
                "lp/best_train_loss",
                "lp/best_epoch",
                "lp/best_num_layers",
                "lp/best_hidden_dim",
                "lp/best_dropout",
                "lp/best_lr",
                "lp/train_num_pos",
                "lp/test_num_pos",
                "lp/test_num_neg",
            ]:
                row[k] = float("nan")

        add_compat_metric_aliases(row)
        per_graph_rows.append(row)
        write_csv(per_graph_rows, Path(args.out_per_graph_csv))

    summary: Dict[str, float] = {
        "num_loaded_graphs": float(total_loaded),
        "num_evaluated_graphs": float(len(per_graph_rows)),
        "reference_num_pairs": float(len(reference_pairs)),
        "reference_pos_pairs": float(reference_labels.sum()),
        "reference_neg_pairs": float((reference_labels == 0).sum()),
        "lp/model": args.lp_model,
        "lp_protocol": "gae_train_test",
        "lp_search": float(bool(args.lp_search)),
        "lp_test_ratio": float(args.lp_test_ratio),
        "lp_val_ratio": float(args.lp_val_ratio),
    }

    numeric_keys = []
    seen = set()
    for row in per_graph_rows:
        for k, v in row.items():
            if isinstance(v, (int, float, np.floating)) and k not in seen:
                seen.add(k)
                numeric_keys.append(k)

    for k in numeric_keys:
        vals = [float(r[k]) for r in per_graph_rows if k in r]
        summary[f"{k}_mean"] = safe_mean(vals)
        summary[f"{k}_std"] = safe_std(vals)

    overlap_mean_scores, overlap_valid = ensemble_mean_scores(overlap_raw_rows, len(reference_pairs))
    if overlap_valid.sum() > 0:
        summary["ensemble_overlap/auc"] = safe_auc(reference_labels[overlap_valid], overlap_mean_scores[overlap_valid])
        summary["ensemble_overlap/coverage_pairs"] = float(overlap_valid.sum())

    if lp_raw_rows:
        # Important: this remains a convenience aggregate of scores from different trained models.
        # Prefer lp/auc_mean and lp/*_mean for model-selection and reporting.
        all_lp_labels = np.concatenate([r["labels"] for r in lp_raw_rows], axis=0)
        all_lp_scores = np.concatenate([r["scores"] for r in lp_raw_rows], axis=0)
        all_lp_sens = np.concatenate([r["sens_mask"] for r in lp_raw_rows], axis=0)
        agg_lp = compute_binary_and_score_fairness(
            probs=all_lp_scores,
            labels=all_lp_labels,
            sens_mask=all_lp_sens,
            threshold=args.threshold,
        )
        for k, v in agg_lp.items():
            summary[f"aggregate_lp/{k}"] = v
        summary["aggregate_lp/num_graphs"] = float(len(lp_raw_rows))

    add_compat_metric_aliases(summary)
    if "aggregate_lp/score_sp_gap" in summary and "aggregate_fair_gap" not in summary:
        summary["aggregate_fair_gap"] = float(summary["aggregate_lp/score_sp_gap"])
    if "aggregate_lp/score_sp_abs_gap" in summary and "aggregate_fair_abs_gap" not in summary:
        summary["aggregate_fair_abs_gap"] = float(summary["aggregate_lp/score_sp_abs_gap"])
    if "ensemble_overlap/auc" in summary and "ensemble_value/linkpred_auc" not in summary:
        summary["ensemble_value/linkpred_auc"] = float(summary["ensemble_overlap/auc"])

    write_csv([summary], Path(args.out_summary_csv))


if __name__ == "__main__":
    main()
