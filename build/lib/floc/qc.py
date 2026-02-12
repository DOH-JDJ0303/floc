from __future__ import annotations

import os
import csv
import logging
import statistics
from typing import Dict, Any, List, Optional, Tuple
import sourmash as sm

from .sm_ops import DistanceCache, _merged_distance

MHMap = Dict[str, Dict[str, Any]]


def _build_filtered_subject_global(
    subject_map: MHMap,
    min_abund: float
) -> Tuple[sm.MinHash, dict]:
    """
    Build a SUBJECT GLOBAL MinHash filtered by cohort abundance:
      - Sum abundances of each hash across SUBJECTS (or treat as 1 if not tracking).
      - Keep hash h if (total_abundance[h] / n_subjects) > min_abund.
      - Construct a new MinHash (track_abundance=True) with kept hashes/abundances.

    Returns:
      (filtered_global_mh, stats)
    """
    if not subject_map:
        raise ValueError("subject_map is empty; cannot build subject global.")

    n_subjects = len(subject_map)

    # Accumulate total abundance per hash across SUBJECTS
    totals = {}  # hash -> summed abundance over subjects
    first_mh = next(iter(subject_map.values()))["mh"]

    for rec in subject_map.values():
        mh = rec["mh"]
        # If not tracking abundance, values are typically 1; that's fine.
        for h, cnt in mh.hashes.items():
            totals[h] = totals.get(h, 0) + cnt

    # Filter by average abundance per subject
    kept = {h: tot for h, tot in totals.items() if (tot / n_subjects) > min_abund}

    # Construct a new global MinHash; use abundance to preserve weights
    global_mh = sm.MinHash(
        n=0 if first_mh.scaled else first_mh.num,
        scaled=first_mh.scaled,
        ksize=first_mh.ksize,
        seed=first_mh.seed,
        track_abundance=True  # ensure we keep totals
    )
    # (Optional: include moltype if available in your sourmash version)
    try:
        if hasattr(first_mh, "moltype") and first_mh.moltype is not None:
            global_mh = sm.MinHash(
                n=0 if first_mh.scaled else first_mh.num,
                scaled=first_mh.scaled,
                ksize=first_mh.ksize,
                seed=first_mh.seed,
                track_abundance=True,
                moltype=first_mh.moltype
            )
    except TypeError:
        # Older sourmash versions may not accept moltype in ctor; ignore gracefully.
        pass

    global_mh.set_abundances(kept)

    stats = {
        "n_subjects": n_subjects,
        "unique_hashes_subjects": len(totals),
        "kept_hashes": len(kept),
        "kept_frac": len(kept) / max(1, len(totals)),
        "min_abund": min_abund,
        "total_abundance_all_subjects": sum(totals.values()),
        "total_abundance_kept": sum(kept.values()),
    }
    return global_mh, stats


def global_containment(
    query_map: MHMap,
    subject_map: MHMap,
    *,
    max_dist: float = 0.5,
    min_abund: float = 0.05,
    outfile: Optional[str] = "global_containment_results.csv",
) -> Tuple[dict, List[dict], MHMap]:
    """
    Queries vs SUBJECT GLOBAL (two inputs):
      • Build SUBJECT GLOBAL from `subject_map`, filtering hashes by avg abundance per subject.
      • Do NOT filter queries.
      • Compute distance for each query to the filtered SUBJECT GLOBAL via _merged_distance.
      • Log an overall summary + only failing queries (distance > max_dist).
      • Save full per-query results to CSV.
      • Return a NEW query_map excluding queries that failed.

    Returns: (summary_dict, results_list, filtered_query_map)
    """
    if not subject_map:
        logging.error("global_containment_two_sets: subject_map is empty.")
        return {"n_queries": len(query_map), "n_subjects": 0}, [], {}

    if not query_map:
        logging.warning("global_containment_two_sets: query_map is empty.")
        return {"n_queries": 0, "n_subjects": len(subject_map)}, [], {}

    # 1) Build FILTERED SUBJECT GLOBAL (filtering based on subject cohort abundance)
    subj_global, subj_stats = _build_filtered_subject_global(subject_map, min_abund)
    mh_global = {"global": {"mh": subj_global}}

    # 2) Compute distances for each (UNFILTERED) query to SUBJECT GLOBAL
    cache = DistanceCache()
    ids: List[str] = []
    dists: List[float] = []
    results: List[dict] = []

    for qid, qinfo in query_map.items():
        local_map = {qid: qinfo} | mh_global
        d = _merged_distance(qid, "global", mh_map=local_map, cache=cache)
        ids.append(qid)
        dists.append(d)
        results.append({
            "id": qid,
            "distance_to_subject_global": d,
            "within_threshold": d <= max_dist
        })

    mean_d = statistics.mean(dists) if dists else 0.0
    sd_d = statistics.pstdev(dists) if len(dists) > 1 else 0.0
    failing = [r for r in results if not r["within_threshold"]]
    failing_ids = {r["id"] for r in failing}

    # 3) Logging summary then only failures
    logging.info("=== Global Containment: Queries vs Filtered Subject Global ===")
    logging.info("Queries                  : %d", len(query_map))
    logging.info("Subjects                 : %d", len(subject_map))
    logging.info("Distance threshold       : <= %.4f", max_dist)
    logging.info("Subject filter threshold : >  %.4f (avg abundance per subject)", min_abund)

    logging.info("Subject cohort unique    : %d hashes; kept %d (%.2f%%)",
                 subj_stats["unique_hashes_subjects"],
                 subj_stats["kept_hashes"],
                 100.0 * subj_stats["kept_frac"])
    logging.info("Subject abundance kept   : %d / %d",
                 subj_stats["total_abundance_kept"],
                 subj_stats["total_abundance_all_subjects"])
    logging.info("Distances                : mean=%.4f σ=%.4f min=%.4f max=%.4f",
                 mean_d, sd_d, min(dists) if dists else 0.0, max(dists) if dists else 0.0)

    if failing:
        logging.warning("=== Queries exceeding distance threshold (d > %.4f) ===", max_dist)
        for r in failing:
            logging.warning("id=%s distance=%.4f", r["id"], r["distance_to_subject_global"])
    else:
        logging.info("All queries are within the distance threshold.")

    # 4) Save per-query results (CSV) using os
    if outfile:
        outdir = os.path.dirname(outfile)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "distance_to_subject_global", "within_threshold"]
            )
            writer.writeheader()
            writer.writerows(results)
        logging.info("Saved per-query results to %s", outfile)

    # 5) Return a NEW query_map with failing queries removed
    query_map_pass: MHMap = {k: v for k, v in query_map.items() if k not in failing_ids}

    summary = {
        "n_queries": len(query_map),
        "n_subjects": len(subject_map),
        "max_dist": max_dist,
        "min_abund": min_abund,
        "subject_unique_hashes": subj_stats["unique_hashes_subjects"],
        "subject_kept_hashes": subj_stats["kept_hashes"],
        "subject_kept_frac": subj_stats["kept_frac"],
        "mean_distance": mean_d,
        "stdev_distance": sd_d,
        "n_failing": len(failing_ids),
        "n_kept": len(query_map_pass),
        "n_removed": len(failing_ids),
    }
    return query_map_pass
    

def abundance_zscores(
    mh_map: MHMap,
    sids: List[str],
    *,
    threshold: float = 2.58,
    outfile: Optional[str] = "abundance_zscores.csv",
) -> MHMap:
    """
    Compute abundance z-scores ONLY over the provided subset `sids`,
    log a summary + only outliers for that subset, save a CSV for that subset,
    and return a new mh_map containing ONLY the subset members that PASS.

    Args:
        mh_map: { sample_id: { 'mh': MinHash, ... }, ... }
        sids: list of sample IDs to include in the z-score cohort and filtering
        threshold: |z| cutoff (default 2.58 ~99% two-tailed)
        outfile: path to CSV (subset results). Set None to skip writing.

    Returns:
        mh_map_pass: MHMap with ONLY those IDs from `sids` whose |z| <= threshold.
    """
    if not mh_map:
        logging.warning("abundance_zscores: mh_map is empty; nothing to do.")
        return {}

    if not sids:
        logging.warning("abundance_zscores: no sids provided; nothing to do.")
        return {}

    # Keep only IDs that exist in mh_map; warn on missing
    present = [sid for sid in sids if sid in mh_map]
    missing = [sid for sid in sids if sid not in mh_map]
    if missing:
        logging.warning("abundance_zscores: %d sids not found and will be ignored: %s",
                        len(missing), ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""))

    if not present:
        logging.warning("abundance_zscores: none of the provided sids exist in mh_map.")
        return {}

    # ---- compute total abundance per sample (subset only) ----
    ids: List[str] = []
    totals: List[int] = []
    for qid in present:
        mh = mh_map[qid]["mh"]
        total_abundance = sum(mh.hashes.values())
        ids.append(qid)
        totals.append(total_abundance)

    # ---- compute summary statistics over the subset ----
    mean_a = statistics.mean(totals)
    stdev_a = statistics.pstdev(totals) if len(totals) > 1 else 0.0

    def z(x, mu, sd):
        return 0.0 if sd == 0 else (x - mu) / sd

    # ---- compute z-scores for the subset ----
    results: List[dict] = []
    for qid, total in zip(ids, totals):
        z_a = z(total, mean_a, stdev_a)
        within = abs(z_a) <= threshold
        results.append({
            "id": qid,
            "total_abundance": total,
            "z_abundance": z_a,
            "within_threshold": within,
            "outlier": not within,
        })

    # ---- log summary (subset only) ----
    logging.info("=== Abundance Summary (subset) ===")
    logging.info("Subset size         : %d", len(ids))
    logging.info("Z-score threshold   : ±%.2f", threshold)
    logging.info("Mean abundance      : %.2f (σ=%.2f)", mean_a, stdev_a)

    # ---- log only outliers (subset only) ----
    outliers = [r for r in results if r["outlier"]]
    if outliers:
        logging.warning("=== Subset outliers (|z| > %.2f) ===", threshold)
        for r in outliers:
            logging.warning("id=%s abundance=%d z=%.2f",
                            r["id"], r["total_abundance"], r["z_abundance"])
    else:
        logging.info("All subset samples within threshold (no outliers).")

    # ---- save CSV for the subset ----
    if outfile:
        outdir = os.path.dirname(outfile)
        if outdir and not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
        with open(outfile, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["id", "total_abundance", "z_abundance", "within_threshold", "outlier"],
            )
            writer.writeheader()
            writer.writerows(results)
        logging.info("Saved subset abundance z-score results to %s", outfile)

    # ---- return ONLY the portion of the subset that passes ----
    failing_ids = {r["id"] for r in outliers}
    mh_map_pass: MHMap = {sid: mh_map[sid] for sid in present if sid not in failing_ids}
    return mh_map_pass
