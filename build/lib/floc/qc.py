from __future__ import annotations

import os
import csv
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
import sourmash as sm
from collections import defaultdict

from .sm_ops import DistanceCache, _merged_distance

MHMap = Dict[str, Dict[str, Any]]


def _build_filtered_subject_global(
    subject_map: MHMap,
    min_hash_freq: float
) -> Tuple[sm.MinHash, dict]:
    """
    Build a SUBJECT GLOBAL MinHash filtered by cohort abundance:
      - Sum abundances of each hash across SUBJECTS (or treat as 1 if not tracking).
      - Keep hash h if (total_abundance[h] / n_subjects) > min_hash_freq.
      - Construct a new MinHash (track_abundance=True) with kept hashes/abundances.

    Returns:
      (filtered_global_mh, stats)
    """
    if not subject_map:
        raise ValueError("subject_map is empty; cannot build subject global.")

    n_subjects = len(subject_map)

    # Count number of observations per hash
    counts = {}
    first_mh = next(iter(subject_map.values()))["mh"]

    counts = defaultdict(lambda: [0, 0])

    for rec in subject_map.values():
        mh = rec["mh"]
        for hash_val, abundance in mh.hashes.items():
            counts[hash_val][0] += 1  # sample count
            counts[hash_val][1] += abundance  # total abundance

    # Filter hashes below the minimum frequency
    kept = {h: a for h, [n, a] in counts.items() if (n / n_subjects) > min_hash_freq}

    # Construct a new global MinHash; use abundance to preserve weights
    global_mh = sm.MinHash(
        n=0 if first_mh.scaled else first_mh.num,
        scaled=first_mh.scaled,
        ksize=first_mh.ksize,
        seed=first_mh.seed,
        track_abundance=True  # ensure we keep totals
    )
    global_mh.set_abundances(kept)

    n_total = len(counts)
    n_kept  = len(kept)


    stats = {
        "n_total": n_total,
        "n_kept": n_kept,
        "p_kept": 100 * n_kept / (n_total if n_total else 1)
    }
    return global_mh, stats


def global_containment(
    query_map: MHMap,
    subject_map: MHMap,
    *,
    min_hash_frac: float = 0.5,
    min_hash_freq: float = 0.05,
    outfile: Optional[str] = "global_containment_results.csv",
) -> Tuple[dict, List[dict], MHMap]:
    
    if not subject_map:
        logging.error("global_containment: subject_map is empty.")
        return {"n_queries": len(query_map), "n_subjects": 0}, [], {}

    if not query_map:
        logging.warning("global_containment: query_map is empty.")
        return {"n_queries": 0, "n_subjects": len(subject_map)}, [], {}

    # 1) Build filtered subject global (filtering based on subject cohort abundance)
    subj_global, subj_stats = _build_filtered_subject_global(subject_map, min_hash_freq)
    mh_global = {"global": {"mh": subj_global}}

    # 2) Compute distances for each query to SUBJECT GLOBAL
    cache = DistanceCache()
    ids: List[str] = []
    fracs: List[float] = []
    results: List[dict] = []

    for qid, qinfo in query_map.items():
        local_map = {qid: qinfo} | mh_global
        f = 1 - _merged_distance(qid, "global", mh_map=local_map, cache=cache)
        ids.append(qid)
        fracs.append(f)
        results.append({
            "id": qid,
            "frac_hash_in_global": f,
            "within_threshold": f >= min_hash_frac
        })

    mean_f = statistics.mean(fracs) if fracs else 0.0
    sd_f = statistics.pstdev(fracs) if len(fracs) > 1 else 0.0
    min_f, max_f = min(fracs), max(fracs)
    failing = [r for r in results if not r["within_threshold"]]
    failing_ids = {r["id"] for r in failing}

    # 3) Logging summary then only failures
    start_msg = "=== Global Containment (QC): Queries vs Filtered Subject Global ==="
    logging.info(start_msg)
    logging.info(f"Queries               : {len(query_map)}")
    logging.info(f"Subjects              : {len(subject_map)}")

    logging.info(f"Global hash frequency : threshold={min_hash_freq}; {subj_stats['n_total']} hashes; kept {subj_stats['n_kept']} ({subj_stats['p_kept']:.2f}%)")
    logging.info(f"Query hash fraction   : threshold={min_hash_frac}; mean={mean_f:.4f} stdev={sd_f:.4f} range=[{min_f:.4f}, {max_f:.4f}]")

    if failing:
        logging.warning(f"Queries below threshold (d > {min_hash_frac:.4f}):")
        for r in failing:
            logging.warning(f"id={r['id']} hash_freq={r['dist_to_global']:.4f}")
    else:
        logging.info("All queries are within the hash frequency threshold.")

    # 4) Save per-query results (CSV) using os
    if outfile:
        outdir = os.path.dirname(outfile)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "frac_hash_in_global", "within_threshold"]
            )
            writer.writeheader()
            writer.writerows(results)
        logging.info(f"Saved per-query results to {outfile}")

    # 5) Return a NEW query_map with failing queries removed
    query_map_pass: MHMap = {k: v for k, v in query_map.items() if k not in failing_ids}

    logging.warning("="*len(start_msg))
    return query_map_pass