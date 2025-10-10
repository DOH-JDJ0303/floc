from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, Tuple, List, Optional  # ← add List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
import plotly.graph_objects as go
import sourmash as sm

from .sm_ops import DistanceCache, _distance, _merged_distance

Rec = Dict[str, Any]
MHMap = Dict[str, Rec]
ClusterMap = Dict[str, Dict[str, Rec]]


def group_clusters(mh_map: MHMap) -> ClusterMap:
    groups: ClusterMap = {}
    for qid, rec in mh_map.items():
        c = rec.get('cluster')
        if c is None:
            logging.debug(f"Record {qid} has no 'cluster' field; skipping in group_clusters.")
            continue
        clabel = str(c)  # ← normalize to string
        groups.setdefault(clabel, {})[qid] = rec
    return groups

def assign_to_cluster(
    mh_fx: MHMap,
    mh_clust: ClusterMap,
    threshold: float
) -> Tuple[MHMap, MHMap]:
    if not mh_clust:
        return {}, {qid: qinfo for qid, qinfo in mh_fx.items()}

    # Merge MinHash per cluster
    mh_merged: Dict[str, Dict[str, Rec]] = {}
    for clabel, members in mh_clust.items():
        for i, (_, rec) in enumerate(members.items()):
            rmh = rec['mh'].to_mutable()
            if i == 0:
                mhm = rmh
            else:
                mhm.merge(rmh)
        mh_merged[clabel] = {clabel: {'mh': mhm}}

    assigned: MHMap = {}
    remainder: MHMap = {}

    merged_dist_cache = DistanceCache()

    min_dist = 1
    for qid, qinfo in mh_fx.items():
        candidate_clusters: List[str] = []
        for clabel, cinfo in mh_merged.items():
            local_map = {qid: qinfo} | cinfo
            d = _merged_distance(qid, clabel, mh_map=local_map, cache=merged_dist_cache)
            if d < threshold and d <= min_dist:
                # logging.info(f"{qid} distance to {clabel}: {d}")
                min_dist = d
                candidate_clusters.append(clabel)

        if not candidate_clusters:
            remainder[qid] = qinfo
            continue

        qinfo['cluster'] = candidate_clusters[0]
        assigned[qid] = qinfo

    return assigned, remainder


def _next_cluster_label(existing: ClusterMap) -> str:
    """
    Generate a new **string** cluster label not present in `existing`.
    Handles int-like or string keys, returns '1', '2', ...
    """
    taken = set()
    for k in existing.keys():
        try:
            taken.add(int(k))
        except Exception:
            # if you ever mix 'c0001' style, you could parse here
            pass
    cur = 1
    while cur in taken:
        cur += 1
    return str(cur)  # ← always string


def _pairwise_distance_matrix(ids: List[str], mh_map: MHMap, cache: DistanceCache) -> np.ndarray:
    n = len(ids)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            qid, sid = ids[i], ids[j]
            local_map = {qid: mh_map[qid], sid: mh_map[sid]}
            d = _distance(qid, sid, mh_map=local_map, cache=cache)
            D[i, j] = D[j, i] = float(d)
    return D


def _add_members_to_cluster(
    clabel: str,
    member_ids: List[str],
    mh_fx: MHMap,
    mh_clust: ClusterMap,
) -> None:
    clabel = str(clabel)  # ← normalize
    if clabel not in mh_clust:
        mh_clust[clabel] = {}
    for qid in member_ids:
        rec = mh_fx[qid]
        rec['cluster'] = clabel
        mh_clust[clabel][qid] = rec


def create_new_clusters(
    mh_fx: MHMap,
    dist_cache: DistanceCache,
    eps: float,
    batch_size: int,
    min_samples: int = 1,
    existing: Optional[ClusterMap] = None
) -> ClusterMap:
    mh_clust: ClusterMap = {} if existing is None else {str(k): dict(v) for k, v in existing.items()}

    qids_all = list(mh_fx.keys())
    if len(qids_all) == 0:
        return mh_clust

    if len(qids_all) == 1:
        clabel = _next_cluster_label(mh_clust)
        _add_members_to_cluster(clabel, [qids_all[0]], mh_fx, mh_clust)
        return mh_clust

    remainder_ids = qids_all[:]

    while remainder_ids:
        batch_ids = remainder_ids[:batch_size]

        if len(batch_ids) == 1:
            clabel = _next_cluster_label(mh_clust)
            _add_members_to_cluster(clabel, batch_ids, mh_fx, mh_clust)
            remainder_ids = remainder_ids[1:]
        else:
            D = _pairwise_distance_matrix(batch_ids, mh_fx, dist_cache)
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
            labels = db.fit_predict(D)

            label_to_members: Dict[int, List[str]] = {}
            for idx, lbl in enumerate(labels):
                label_to_members.setdefault(int(lbl), []).append(batch_ids[idx])

            for lbl, members in label_to_members.items():
                if lbl == -1:
                    continue
                clabel = _next_cluster_label(mh_clust)
                _add_members_to_cluster(clabel, members, mh_fx, mh_clust)

            for noise_id in label_to_members.get(-1, []):
                clabel = _next_cluster_label(mh_clust)
                _add_members_to_cluster(clabel, [noise_id], mh_fx, mh_clust)

            remainder_ids = remainder_ids[len(batch_ids):]

        if remainder_ids:
            remaining_map: MHMap = {qid: mh_fx[qid] for qid in remainder_ids}
            assigned, still_left = assign_to_cluster(
                remaining_map, mh_clust, threshold=eps, dist_cache=dist_cache,
            )

            # *** IMPORTANT: merge assigned into mh_clust ***
            for qid, rec in assigned.items():
                clabel = str(rec['cluster'])
                mh_clust.setdefault(clabel, {})[qid] = rec

            remainder_ids = list(still_left.keys())

    return mh_clust


def _pairwise_dist_full(ids: List[str], mh_map: MHMap, cache: DistanceCache) -> np.ndarray:
    """Compute full symmetric distance matrix for given ids using _distance."""
    n = len(ids)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ids[i], ids[j]
            local_map = {a: mh_map[a], b: mh_map[b]}
            d = _distance(a, b, mh_map=local_map, cache=cache)
            D[i, j] = D[j, i] = float(d)
    return D

def _pcoa_from_dist(D: np.ndarray, n_components: int = 2):
    """
    Classical MDS / PCoA from a (n x n) distance matrix D.
    Returns: coords (n x k), eigvals (k,), explained (k,)
    """
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.allclose(D, D.T, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(D), 0.0, atol=1e-12):
        logging.warning("Distance matrix diagonal is not ~0; forcing zeros.")
        D = D.copy()
        np.fill_diagonal(D, 0.0)

    # Double-center the squared distances: B = -0.5 * J D^2 J
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J

    # Eigen-decomposition of B (symmetric)
    # eigh returns ascending order; take largest eigenpairs from the end
    eigvals, eigvecs = np.linalg.eigh(B)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Keep only positive eigenvalues (within tolerance)
    pos = eigvals > 1e-12
    if not np.any(pos):
        raise ValueError("No positive eigenvalues found; check the distance matrix.")
    eigvals_pos = eigvals[pos]
    eigvecs_pos = eigvecs[:, pos]

    k = min(n_components, eigvals_pos.shape[0])
    eigvals_k = eigvals_pos[:k]
    eigvecs_k = eigvecs_pos[:, :k]

    coords = eigvecs_k * np.sqrt(eigvals_k)

    # Variance explained (proportion) by each retained axis relative to sum of positive eigvals
    total_pos = np.sum(eigvals_pos)
    explained = eigvals_k / total_pos if total_pos > 0 else np.zeros_like(eigvals_k)

    return coords, eigvals_k, explained


def pcoa_plot(
    mh_map,
    dist_cache,
    *,
    n_components: int = 2,          # set to 3 for 3D
    title: str = "PCoA of MinHash distances",
    save_html: Optional[str] = None # e.g., "pcoa.html"
) -> go.Figure:
    """
    Make an interactive Plotly PCoA (Classical MDS) plot between all samples in mh_map.
    - Uses your _pairwise_dist_full(...) to build the dissimilarity matrix (D).
    - Deterministic: no random init; identical samples overlap.
    - Colors points by 'cluster' if present, else 'unassigned'.
    - Returns a Plotly Figure; optionally writes to HTML if save_html is given.
    """
    ids = list(mh_map.keys())
    if len(ids) < 2:
        raise ValueError("Need at least 2 samples to compute a PCoA plot.")

    # Build distance matrix
    D = _pairwise_dist_full(ids, mh_map, dist_cache)

    # Classical MDS / PCoA
    coords, eigvals_k, explained = _pcoa_from_dist(D, n_components=n_components)

    # Cluster labels
    clusters: List[str] = []
    for sid in ids:
        c = mh_map[sid].get("cluster")
        clusters.append(str(c) if c is not None else "unassigned")

    # Axis titles with % variance explained
    def ax_label(i: int) -> str:
        pct = 100.0 * (explained[i] if i < len(explained) else 0.0)
        return f"PCoA{i+1} ({pct:.1f}%)"

    # Build Plotly figure per cluster (clean legend)
    fig = go.Figure()
    for cl in sorted(set(clusters), key=lambda s: (s == "unassigned", s)):
        idxs = [i for i, c in enumerate(clusters) if c == cl]
        if n_components == 3:
            fig.add_trace(go.Scatter3d(
                x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2],
                mode="markers", name=str(cl), marker=dict(size=8, opacity=0.9),
                text=[f"sid={ids[i]}<br>cluster={cl}" for i in idxs],
                hovertemplate="%{text}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=coords[idxs, 0], y=coords[idxs, 1],
                mode="markers", name=str(cl), marker=dict(size=10, opacity=0.9),
                text=[f"sid={ids[i]}<br>cluster={cl}" for i in idxs],
                hovertemplate="%{text}<extra></extra>",
            ))

    if n_components == 3:
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=ax_label(0),
                yaxis_title=ax_label(1),
                zaxis_title=ax_label(2),
            ),
            legend_title_text="cluster"
        )
    else:
        fig.update_layout(
            title=title,
            xaxis_title=ax_label(0),
            yaxis_title=ax_label(1),
            legend_title_text="cluster",
        )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")

    return fig


def mds_plot(
    mh_map,
    dist_cache,
    *,
    n_components: int = 2,
    metric: bool = True,            # kept for API compatibility; ignored
    random_state: int = 42,         # ignored
    max_iter: int = 1000,           # ignored
    n_init: int = 4,                # ignored
    title: str = "PCoA of MinHash distances",
    save_html: Optional[str] = None
) -> go.Figure:
    """
    Backward-compatible wrapper: MDS -> PCoA.
    Ignores MDS-specific args and calls pcoa_plot(...).
    """
    logging.info("mds_plot(): Using PCoA (Classical MDS) instead of iterative MDS. "
                 "Args metric/random_state/max_iter/n_init are ignored.")
    return pcoa_plot(
        mh_map,
        dist_cache,
        n_components=n_components,
        title=title,
        save_html=save_html,
    )


