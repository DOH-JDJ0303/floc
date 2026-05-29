from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import os
import gzip

from .sm_ops import DistanceCache, _distance, _merged_distance
from .io_ops import drop_ext

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
        clabel = str(c)
        groups.setdefault(clabel, {})[qid] = rec
    return groups


def assign_to_cluster(
    mh_fx: MHMap,
    mh_clust: ClusterMap,
    threshold: float,
    dist_cache: Optional[DistanceCache] = None
) -> Tuple[MHMap, MHMap]:
    if not mh_clust:
        return {}, {qid: qinfo for qid, qinfo in mh_fx.items()}

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
    merged_dist_cache = dist_cache if dist_cache is not None else DistanceCache()

    for qid, qinfo in mh_fx.items():
        min_dist = 1
        candidate_clusters: List[str] = []
        for clabel, cinfo in mh_merged.items():
            local_map = {qid: qinfo} | cinfo
            d = _merged_distance(qid, clabel, mh_map=local_map, cache=merged_dist_cache)
            if d < threshold and d <= min_dist:
                min_dist = d
                candidate_clusters.append(clabel)

        if not candidate_clusters:
            remainder[qid] = qinfo
            continue

        qinfo['cluster'] = candidate_clusters[0]
        assigned[qid] = qinfo

    return assigned, remainder


def _next_cluster_label(cmap: ClusterMap) -> str:
    current = [0]
    for c in cmap.keys():
        try:
            current.append(int(c))
        except Exception:
            raise ValueError(f"Cluster names must be integers: {c}")
    return str(max(current) + 1)


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
    clabel = str(clabel)
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
            for qid, rec in assigned.items():
                clabel = str(rec['cluster'])
                mh_clust.setdefault(clabel, {})[qid] = rec
            remainder_ids = list(still_left.keys())

    return mh_clust


def _pcoa_from_dist(D: np.ndarray, n_components: int = 2):
    """Classical MDS / PCoA from a (n x n) distance matrix D."""
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    if not np.allclose(D, D.T, atol=1e-12):
        raise ValueError("Distance matrix must be symmetric.")
    if not np.allclose(np.diag(D), 0.0, atol=1e-12):
        logging.warning("Distance matrix diagonal is not ~0; forcing zeros.")
        D = D.copy()
        np.fill_diagonal(D, 0.0)

    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    pos = eigvals > 1e-12
    if not np.any(pos):
        raise ValueError("No positive eigenvalues found; check the distance matrix.")
    eigvals_pos = eigvals[pos]
    eigvecs_pos = eigvecs[:, pos]

    k = min(n_components, eigvals_pos.shape[0])
    eigvals_k = eigvals_pos[:k]
    eigvecs_k = eigvecs_pos[:, :k]

    coords = eigvecs_k * np.sqrt(eigvals_k)
    total_pos = np.sum(eigvals_pos)
    explained = eigvals_k / total_pos if total_pos > 0 else np.zeros_like(eigvals_k)

    return coords, eigvals_k, explained


def _cluster_labels(ids: List[str], mh_map: MHMap) -> List[str]:
    """Extract cluster label strings for a list of IDs, falling back to 'unassigned'."""
    return [
        str(mh_map[sid]["cluster"]) if mh_map[sid].get("cluster") is not None else "unassigned"
        for sid in ids
    ]


def pcoa_plot(
    mh_map: MHMap,
    dist_cache: DistanceCache,
    *,
    n_components: int = 2,
    title: str = "PCoA of MinHash distances",
    save_html: Optional[str] = None,
) -> go.Figure:
    """Interactive Plotly PCoA (Classical MDS) plot for all samples in mh_map."""
    ids = list(mh_map.keys())
    if len(ids) < 2:
        logging.warning("Two or more samples required to create PCoA plot!")
        return None

    D = _pairwise_distance_matrix(ids, mh_map, dist_cache)
    coords, _, explained = _pcoa_from_dist(D, n_components=n_components)
    clusters = _cluster_labels(ids, mh_map)

    def ax_label(i: int) -> str:
        pct = 100.0 * (explained[i] if i < len(explained) else 0.0)
        return f"PCoA{i+1} ({pct:.1f}%)"

    fig = go.Figure()
    for cl in sorted(set(clusters), key=lambda s: (s == "unassigned", s)):
        idxs = [i for i, c in enumerate(clusters) if c == cl]
        hover = [f"sid={ids[i]}<br>cluster={cl}" for i in idxs]
        if n_components == 3:
            fig.add_trace(go.Scatter3d(
                x=coords[idxs, 0], y=coords[idxs, 1], z=coords[idxs, 2],
                mode="markers", name=str(cl),
                marker=dict(size=8, opacity=0.9),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=coords[idxs, 0], y=coords[idxs, 1],
                mode="markers", name=str(cl),
                marker=dict(size=10, opacity=0.9),
                text=hover, hovertemplate="%{text}<extra></extra>",
            ))

    if n_components == 3:
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title=ax_label(0), yaxis_title=ax_label(1), zaxis_title=ax_label(2)),
            legend_title_text="cluster",
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


def nj_tree(
    mh_map: MHMap,
    dist_cache: DistanceCache,
    *,
    title: str = "Neighbor-Joining Tree",
    save_html: Optional[str] = None,
    save_newick: Optional[str] = None,
) -> go.Figure:
    """
    Build a Neighbor-Joining tree and render it as an interactive Plotly figure.
    Requires: pip install biopython
    """
    from Bio import Phylo
    from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor

    ids = list(mh_map.keys())
    n = len(ids)
    if n < 2:
        logging.warning("Two or more samples required to build NJ tree!")
        return None

    D = _pairwise_distance_matrix(ids, mh_map, dist_cache)

    # Bio.Phylo expects a lower-triangle matrix (row i has i+1 entries)
    lower = [[float(D[i, j]) for j in range(i + 1)] for i in range(n)]
    dm = DistanceMatrix(names=ids, matrix=lower)
    tree = DistanceTreeConstructor().nj(dm)
    tree.root_at_midpoint()  # ← midpoint root

    if save_newick:
        Phylo.write(tree, save_newick, "newick")
        logging.info(f"Newick tree written to {save_newick}")

    # x = cumulative branch length from root; y = evenly spaced leaves
    def _x_positions(tree) -> Dict[int, float]:
        pos: Dict[int, float] = {}
        def _walk(clade, cum: float):
            cum += clade.branch_length or 0.0
            pos[id(clade)] = cum
            for child in clade.clades:
                _walk(child, cum)
        _walk(tree.root, 0.0)
        return pos

    def _y_positions(tree) -> Dict[int, float]:
        leaf_y = {id(lf): i for i, lf in enumerate(tree.get_terminals())}
        pos: Dict[int, float] = {}
        def _walk(clade):
            if clade.is_terminal():
                pos[id(clade)] = leaf_y[id(clade)]
            else:
                for child in clade.clades:
                    _walk(child)
                pos[id(clade)] = sum(pos[id(c)] for c in clade.clades) / len(clade.clades)
        _walk(tree.root)
        return pos

    x_pos = _x_positions(tree)
    y_pos = _y_positions(tree)

    # Collect branch edges
    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []

    def _collect_edges(clade, parent_x: float, parent_y: float):
        cx, cy = x_pos[id(clade)], y_pos[id(clade)]
        edge_x.extend([parent_x, cx, None])   # horizontal
        edge_y.extend([cy, cy, None])
        edge_x.extend([parent_x, parent_x, None])  # vertical
        edge_y.extend([parent_y, cy, None])
        for child in clade.clades:
            _collect_edges(child, cx, cy)

    root = tree.root
    for child in root.clades:
        _collect_edges(child, x_pos[id(root)], y_pos[id(root)])

    clusters = {sid: cl for sid, cl in zip(ids, _cluster_labels(ids, mh_map))}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines", line=dict(color="grey", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    # Group leaves by cluster for a clean legend
    cluster_groups: Dict[str, List] = {}
    for leaf in tree.get_terminals():
        cl = clusters.get(leaf.name, "unassigned")
        cluster_groups.setdefault(cl, []).append(
            (leaf.name, x_pos[id(leaf)], y_pos[id(leaf)])
        )

    for cl in sorted(cluster_groups, key=lambda s: (s == "unassigned", s)):
        members = cluster_groups[cl]
        fig.add_trace(go.Scatter(
            x=[m[1] for m in members],
            y=[m[2] for m in members],
            mode="markers+text",
            name=cl,
            text=[m[0] for m in members],
            textposition="middle right",
            marker=dict(size=8, opacity=0.9),
            customdata=[f"sid={m[0]}<br>cluster={cl}" for m in members],
            hovertemplate="%{customdata}<extra></extra>",
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(title="Branch length", zeroline=False),
        yaxis=dict(showticklabels=False, zeroline=False),
        legend_title_text="cluster",
        plot_bgcolor="white",
    )

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")

    return fig


def calculate_cluster_distances(new_ids, mh_clust, dist_cache, outdir):
    dist_dir = os.path.join(outdir, "dist")
    os.makedirs(dist_dir, exist_ok=True)

    for c, s in mh_clust.items():
        cluster_ids = list(s.keys())

        if not (set(cluster_ids) & set(new_ids)):
            continue

        D = _pairwise_distance_matrix(cluster_ids, s, dist_cache)

        dvals = [D[i, j] for i in range(len(cluster_ids)) for j in range(i + 1, len(cluster_ids))]

        if dvals:
            logging.info(
                f"Cluster {c}: N={len(cluster_ids)}, "
                f"mean={statistics.mean(dvals):.6f}, "
                f"stdev={statistics.stdev(dvals) if len(dvals) > 1 else 0.0:.6f}, "
                f"min={min(dvals):.6f}, max={max(dvals):.6f}"
            )
        else:
            logging.info(f"Cluster {c}: only one element; no pairwise distance stats.")

        wide_file = os.path.join(dist_dir, f"{c}.tsv.gz")
        with gzip.open(wide_file, "wt", encoding="utf-8") as f:
            f.write('\t' + '\t'.join(cluster_ids) + '\n')
            for i, row_id in enumerate(cluster_ids):
                f.write(row_id + '\t' + '\t'.join(str(D[i, j]) for j in range(len(cluster_ids))) + '\n')