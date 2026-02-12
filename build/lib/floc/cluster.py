from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, Tuple, List, Optional  # ← add List, Optional

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
import plotly.graph_objects as go
import sourmash as sm
import os
import gzip
import csv

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
        logging.warning("Two or more samples required to create PCoA plot!")
        return

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

def calculate_cluster_distances(new_ids, mh_clust, dist_cache, outdir):

    dist_dir = os.path.join(outdir, "dist")
    os.makedirs(dist_dir, exist_ok=True)
    
    for c, s in mh_clust.items():
        cluster_ids = list(s.keys())

        # Only process clusters containing at least one of the new IDs
        if not (set(cluster_ids) & set(new_ids)):
            continue
            
        # Calculate pairwise distance matrix
        D = _pairwise_dist_full(cluster_ids, s, dist_cache)

        # --- Compute statistics (upper triangle only, i < j) ---
        dvals = []
        n = len(cluster_ids)
        for i in range(n):
            for j in range(i + 1, n):
                dvals.append(D[i, j])

        if dvals:
            avg_val = statistics.mean(dvals)
            stdev_val = statistics.stdev(dvals) if len(dvals) > 1 else 0.0
            min_val = min(dvals)
            max_val = max(dvals)

            logging.info(
                f"Cluster {c}: N={len(cluster_ids)}, "
                f"mean={avg_val:.6f}, stdev={stdev_val:.6f}, "
                f"min={min_val:.6f}, max={max_val:.6f}"
            )
        else:
            logging.info(f"Cluster {c}: only one element; no pairwise distance stats.")

        # --- Save wide form (matrix format) as gzipped TSV ---
        wide_file = os.path.join(dist_dir, f"{c}.tsv.gz")
        with gzip.open(wide_file, "wt", encoding="utf-8") as f:
            # Header row
            f.write('\t' + '\t'.join(cluster_ids) + '\n')

            # Data rows
            for i, row_id in enumerate(cluster_ids):
                row_vals = '\t'.join(str(D[i, j]) for j in range(n))
                f.write(row_id + '\t' + row_vals + '\n')


import csv
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)
from scipy.optimize import linear_sum_assignment


# ----------------------------
# Result container (no pandas)
# ----------------------------
@dataclass
class BenchmarkResult:
    # Global agreement
    ari: float
    nmi: float
    ami: float
    v_measure: float
    homogeneity: float
    completeness: float

    # Counts
    n_overlap: int
    n_query_only: int
    n_target_only: int
    n_query_clusters: int
    n_target_clusters: int

    # Sample-level diagnostics
    sample_rows: List[Dict[str, str]]      # [{'sample':..., 'query':..., 'target':..., 'changed': '0/1'}, ...]
    changed_rows: List[Dict[str, str]]     # subset where changed==1
    instability_by_query: List[Tuple[str, float, int]]   # (query_cluster, frac_changed, n)
    instability_by_target: List[Tuple[str, float, int]]  # (target_cluster, frac_changed, n)

    # Contingency (cluster-to-cluster)
    query_clusters: List[str]
    target_clusters: List[str]
    contingency: Dict[str, Dict[str, int]]         # counts: contingency[q][t] -> n
    contingency_row_norm: Dict[str, Dict[str, float]]  # per-query normalized
    contingency_col_norm: Dict[str, Dict[str, float]]  # per-target normalized

    # Best-match mapping (Hungarian)
    mapping_query_to_target: Dict[str, str]
    mapping_target_to_query: Dict[str, str]

    # Split/merge hints
    split_hints: List[Dict[str, Any]]   # per query cluster
    merge_hints: List[Tuple[str, int]]  # (target_cluster, n_query_clusters_contributing)


def _drop_ext_default(x: str) -> str:
    s = str(x)
    for suf in (".fastq.gz", ".fq.gz", ".fasta.gz", ".fa.gz", ".fna.gz", ".vcf.gz"):
        if s.endswith(suf):
            return s[: -len(suf)]
    if "." in s:
        return s.rsplit(".", 1)[0]
    return s


def _sorted_unique(vals: List[str]) -> List[str]:
    return sorted(set(vals), key=lambda v: (str(v)))


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else float("nan")


def _top_n(items, n=10, key=None, reverse=True):
    items = list(items)
    items.sort(key=key, reverse=reverse)
    return items[:n]


def benchmark(
    query: Dict[Any, Dict[Any, Any]],
    target_csv: str,
    *,
    target_assembly_col: str = "assembly",
    target_cluster_col: str = "cluster",
    drop_ext_fn=_drop_ext_default,
    error_on_duplicate_query_membership: bool = True,
    print_report: bool = True,
    top_k_changed: int = 50,
    max_print_table: int = 30,   # avoid printing huge contingency tables
) -> BenchmarkResult:
    """
    Compare query vs target clustering with:
      - ARI, NMI, AMI, V-measure, homogeneity, completeness
      - Contingency table + row/col-normalized tables
      - Sample-level moved list + per-cluster instability
      - Split/merge hints
      - Best-match mapping (Hungarian assignment on overlaps)

    query format: {query_cluster_id: {sample_id: ...}, ...}
    target_csv: CSV with columns [assembly, cluster] (configurable).
    """

    # ----------------------------
    # Load target labels: {sample -> target_cluster}
    # ----------------------------
    target_label_by_sample: Dict[str, str] = {}
    with open(target_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            asm = row.get(target_assembly_col)
            clu = row.get(target_cluster_col)
            if not asm or not clu:
                continue
            sid = drop_ext_fn(str(asm))
            target_label_by_sample[sid] = str(clu)

    # ----------------------------
    # Build query labels: {sample -> query_cluster}
    # ----------------------------
    query_label_by_sample: Dict[str, str] = {}
    duplicates: Dict[str, List[str]] = {}

    for q_cluster, members in (query or {}).items():
        if not isinstance(members, dict):
            continue
        q_cluster = str(q_cluster)
        for raw_sid in members.keys():
            sid = drop_ext_fn(str(raw_sid))
            if sid in query_label_by_sample and query_label_by_sample[sid] != q_cluster:
                duplicates.setdefault(sid, [query_label_by_sample[sid]]).append(q_cluster)
            query_label_by_sample[sid] = q_cluster

    if duplicates and error_on_duplicate_query_membership:
        example = list(duplicates.items())[:10]
        msg = "Samples appear in multiple query clusters (showing up to 10):\n"
        msg += "\n".join([f"  {sid}: {clist}" for sid, clist in example])
        raise ValueError(msg)

    # ----------------------------
    # Align samples present in both
    # ----------------------------
    shared = sorted(set(query_label_by_sample) & set(target_label_by_sample))
    if not shared:
        raise ValueError("No overlapping samples between query and target clusters.")

    q_labels = [query_label_by_sample[s] for s in shared]
    t_labels = [target_label_by_sample[s] for s in shared]

    n_overlap = len(shared)
    n_query_only = len(query_label_by_sample) - n_overlap
    n_target_only = len(target_label_by_sample) - n_overlap
    n_query_clusters = len(set(q_labels))
    n_target_clusters = len(set(t_labels))

    # ----------------------------
    # Global agreement metrics
    # ----------------------------
    ari = float(adjusted_rand_score(t_labels, q_labels))
    nmi = float(normalized_mutual_info_score(t_labels, q_labels))
    ami = float(adjusted_mutual_info_score(t_labels, q_labels))
    v_measure = float(v_measure_score(t_labels, q_labels))
    homogeneity = float(homogeneity_score(t_labels, q_labels))
    completeness = float(completeness_score(t_labels, q_labels))

    # ----------------------------
    # Sample-level rows + moved list
    # ----------------------------
    sample_rows: List[Dict[str, str]] = []
    changed_rows: List[Dict[str, str]] = []

    for s, q, t in zip(shared, q_labels, t_labels):
        changed = "1" if q != t else "0"
        row = {"sample": s, "query": q, "target": t, "changed": changed}
        sample_rows.append(row)
        if changed == "1":
            changed_rows.append(row)

    # ----------------------------
    # Per-cluster instability (fraction changed)
    # ----------------------------
    # For query clusters
    q_tot: Dict[str, int] = {}
    q_chg: Dict[str, int] = {}
    for row in sample_rows:
        qc = row["query"]
        q_tot[qc] = q_tot.get(qc, 0) + 1
        if row["changed"] == "1":
            q_chg[qc] = q_chg.get(qc, 0) + 1

    instability_by_query = []
    for qc, tot in q_tot.items():
        chg = q_chg.get(qc, 0)
        instability_by_query.append((qc, _safe_div(chg, tot), tot))
    instability_by_query.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # For target clusters
    t_tot: Dict[str, int] = {}
    t_chg: Dict[str, int] = {}
    for row in sample_rows:
        tc = row["target"]
        t_tot[tc] = t_tot.get(tc, 0) + 1
        if row["changed"] == "1":
            t_chg[tc] = t_chg.get(tc, 0) + 1

    instability_by_target = []
    for tc, tot in t_tot.items():
        chg = t_chg.get(tc, 0)
        instability_by_target.append((tc, _safe_div(chg, tot), tot))
    instability_by_target.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # ----------------------------
    # Contingency table: contingency[q][t] = count
    # ----------------------------
    query_clusters = _sorted_unique(q_labels)
    target_clusters = _sorted_unique(t_labels)

    contingency: Dict[str, Dict[str, int]] = {qc: {tc: 0 for tc in target_clusters} for qc in query_clusters}
    for q, t in zip(q_labels, t_labels):
        contingency[q][t] += 1

    # Row-normalized (per query cluster)
    contingency_row_norm: Dict[str, Dict[str, float]] = {}
    for qc in query_clusters:
        row_sum = sum(contingency[qc].values())
        contingency_row_norm[qc] = {tc: _safe_div(contingency[qc][tc], row_sum) for tc in target_clusters}

    # Col-normalized (per target cluster)
    col_sums: Dict[str, int] = {tc: 0 for tc in target_clusters}
    for qc in query_clusters:
        for tc in target_clusters:
            col_sums[tc] += contingency[qc][tc]

    contingency_col_norm: Dict[str, Dict[str, float]] = {qc: {} for qc in query_clusters}
    for qc in query_clusters:
        for tc in target_clusters:
            contingency_col_norm[qc][tc] = _safe_div(contingency[qc][tc], col_sums[tc])

    # ----------------------------
    # Best-match mapping (Hungarian) on overlaps
    # ----------------------------
    ct = np.array([[contingency[qc][tc] for tc in target_clusters] for qc in query_clusters], dtype=int)
    cost = -ct  # maximize overlap
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping_q2t: Dict[str, str] = {}
    mapping_t2q: Dict[str, str] = {}
    for i, j in zip(row_ind, col_ind):
        qc = str(query_clusters[i])
        tc = str(target_clusters[j])
        mapping_q2t[qc] = tc
        mapping_t2q[tc] = qc

    # ----------------------------
    # Split / merge hints
    # ----------------------------
    split_hints: List[Dict[str, Any]] = []
    for qc in query_clusters:
        row = contingency[qc]
        size = sum(row.values())
        if size == 0:
            continue
        touched = sum(1 for v in row.values() if v > 0)
        best_tc = max(row.items(), key=lambda kv: kv[1])[0]
        best_overlap = row[best_tc]
        purity = _safe_div(best_overlap, size)
        hung_tc = mapping_q2t.get(qc)
        hung_overlap = row.get(hung_tc, 0) if hung_tc is not None else 0
        split_hints.append(
            {
                "query_cluster": qc,
                "size": size,
                "targets_touched": touched,
                "best_target_by_max": best_tc,
                "best_overlap": best_overlap,
                "purity_to_best_target": purity,
                "hungarian_target": hung_tc,
                "hungarian_overlap": hung_overlap,
            }
        )
    # “splitty” clusters first: many targets touched, then low purity, then size
    split_hints.sort(key=lambda d: (d["targets_touched"], -d["purity_to_best_target"], d["size"]), reverse=True)

    # Merge hints: for each target cluster, how many query clusters contribute?
    merge_hints: List[Tuple[str, int]] = []
    for tc in target_clusters:
        contributors = sum(1 for qc in query_clusters if contingency[qc][tc] > 0)
        merge_hints.append((tc, contributors))
    merge_hints.sort(key=lambda x: x[1], reverse=True)

    # ----------------------------
    # Reporting
    # ----------------------------
    if print_report:
        print(
            "=== Cluster Benchmark ===\n"
            f"Overlap samples: {n_overlap}\n"
            f"Query-only samples: {n_query_only}\n"
            f"Target-only samples: {n_target_only}\n"
            f"Query clusters (overlap): {n_query_clusters}\n"
            f"Target clusters (overlap): {n_target_clusters}\n"
        )

        print(
            "=== Agreement (label-invariant) ===\n"
            f"ARI:         {ari:.6f}\n"
            f"NMI:         {nmi:.6f}\n"
            f"AMI:         {ami:.6f}\n"
            f"V-measure:   {v_measure:.6f}\n"
            f"Homogeneity: {homogeneity:.6f}\n"
            f"Completeness:{completeness:.6f}\n"
        )

        print("=== Most unstable query clusters (frac_changed, n) ===")
        for qc, frac, n in instability_by_query[:10]:
            print(f"{qc}\t{frac:.3f}\t{n}")
        print()

        print("=== Most unstable target clusters (frac_changed, n) ===")
        for tc, frac, n in instability_by_target[:10]:
            print(f"{tc}\t{frac:.3f}\t{n}")
        print()

        print("=== Split hints (query clusters touching many targets) ===")
        for d in split_hints[:10]:
            print(
                f"{d['query_cluster']}\t"
                f"n={d['size']}\t"
                f"targets_touched={d['targets_touched']}\t"
                f"purity={d['purity_to_best_target']:.3f}\t"
                f"best_target={d['best_target_by_max']}\t"
                f"hungarian_target={d['hungarian_target']}"
            )
        print()

        print("=== Merge hints (targets with many contributing query clusters) ===")
        for tc, contributors in merge_hints[:10]:
            print(f"{tc}\tcontributors={contributors}")
        print()

        n_changed = len(changed_rows)
        print(f"=== Samples that changed (n={n_changed}) ===")
        if n_changed == 0:
            print("(none)\n")
        else:
            for row in changed_rows[:top_k_changed]:
                print(f"{row['sample']}\tquery={row['query']}\ttarget={row['target']}")
            if n_changed > top_k_changed:
                print(f"... ({n_changed - top_k_changed} more)\n")

        # Print contingency table (counts) if not too large
        if len(query_clusters) <= max_print_table and len(target_clusters) <= max_print_table:
            print("=== Contingency table (query x target) [counts] ===")
            header = ["query\\target"] + target_clusters
            print("\t".join(header))
            for qc in query_clusters:
                row = [qc] + [str(contingency[qc][tc]) for tc in target_clusters]
                print("\t".join(row))
            print()
        else:
            print(
                f"=== Contingency table not printed (size {len(query_clusters)}x{len(target_clusters)} "
                f"> max_print_table={max_print_table}) ===\n"
            )

    return BenchmarkResult(
        ari=ari,
        nmi=nmi,
        ami=ami,
        v_measure=v_measure,
        homogeneity=homogeneity,
        completeness=completeness,
        n_overlap=n_overlap,
        n_query_only=n_query_only,
        n_target_only=n_target_only,
        n_query_clusters=n_query_clusters,
        n_target_clusters=n_target_clusters,
        sample_rows=sample_rows,
        changed_rows=changed_rows,
        instability_by_query=instability_by_query,
        instability_by_target=instability_by_target,
        query_clusters=query_clusters,
        target_clusters=target_clusters,
        contingency=contingency,
        contingency_row_norm=contingency_row_norm,
        contingency_col_norm=contingency_col_norm,
        mapping_query_to_target=mapping_q2t,
        mapping_target_to_query=mapping_t2q,
        split_hints=split_hints,
        merge_hints=merge_hints,
    )

