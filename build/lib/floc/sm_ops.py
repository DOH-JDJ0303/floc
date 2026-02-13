from __future__ import annotations

import os
import logging
from typing import Any, Dict, Tuple, Optional, Union

import sourmash as sm
from sourmash.signature import SourmashSignature
import screed
import gzip


# ----------------------------
# Distance cache & distance fn
# ----------------------------

class DistanceCache:
    """Symmetric key cache: cache[(a, b)] == cache[(b, a)]."""
    def __init__(self) -> None:
        self._d: Dict[Tuple[Any, Any], float] = {}

    @staticmethod
    def _key(a: Any, b: Any) -> Tuple[Any, Any]:
        return (a, b) if a <= b else (b, a)

    def get(self, a: Any, b: Any) -> Optional[float]:
        return self._d.get(self._key(a, b))

    def set(self, a: Any, b: Any, val: float) -> None:
        self._d[self._key(a, b)] = float(val)


def _distance(
    a: str,
    b: str,
    *,
    mh_map: Dict[str, Dict[str, Any]],  # expects {'mh': sm.MinHash, 'cluster': ...}
    cache: DistanceCache,
) -> float:
    """Cached pairwise distance = 1 - avg_containment."""
    cached = cache.get(a, b)
    if cached is not None:
        return cached

    try:
        mha: sm.MinHash = mh_map[a]['mh']
        mhb: sm.MinHash = mh_map[b]['mh']
    except KeyError as e:
        raise KeyError(f"Missing MinHash for key {e} in mh_map.") from e

    d = 1.0 - float(mha.avg_containment(mhb))
    cache.set(a, b, d)
    return d


def _merged_distance(
    a: str,
    b: str,
    *,
    mh_map: Dict[str, Dict[str, Any]],  # expects {'mh': sm.MinHash, 'cluster': ...}
    cache: DistanceCache,
) -> float:
    """Cached pairwise distance = 1 - avg_containment."""
    cached = cache.get(a, b)
    if cached is not None:
        return cached

    try:
        mha: sm.MinHash = mh_map[a]['mh']
        mhb: sm.MinHash = mh_map[b]['mh']
    except KeyError as e:
        raise KeyError(f"Missing MinHash for key {e} in mh_map.") from e

    d = 1.0 - float(mha.contained_by(mhb))
    cache.set(a, b, d)
    return d


# ----------------------------
# Helpers for parameter checks
# ----------------------------

def rescale_mh(mh: sm.MinHash, desired_scaled: int) -> sm.MinHash:
    """Downsample to desired_scaled. Only possible if desired_scaled >= mh.scaled (coarser)."""
    if mh.num:  # num-based sketches cannot be converted to scaled
        raise ValueError("Cannot convert a fixed-num MinHash to scaled.")
    if mh.scaled == desired_scaled:
        return mh
    if mh.scaled and desired_scaled >= mh.scaled:
        return mh.downsample(scaled=desired_scaled)
    raise ValueError(
        f"Cannot downsample to smaller scaled (requested {desired_scaled}, have {mh.scaled})."
    )


# ----------------------------
# MinHash builders / loaders
# ----------------------------

def build_minhash_map(
    files_fx: Dict[str, str],
    ksize: int,
    *,
    scaled: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    
    mh_map: Dict[str, Dict[str, Any]] = {}

    for sid, path in files_fx.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{sid}: sequence file not found: {path}")

        seq_chunks = []
        for rec in screed.open(path):
            seq_chunks.append(rec.sequence if hasattr(rec, "sequence") else rec["sequence"])
        seq = "".join(seq_chunks)

        mh = sm.MinHash(
            n=0,
            ksize=ksize,
            seed=seed,
            scaled=scaled,
            track_abundance=True,
        )

        mh.add_sequence(seq, force=True)
        mh_map[sid] = {'mh': mh, 'cluster': None}
    return mh_map


def _get_cluster_from_signature(sig: sm.SourmashSignature) -> int:
    """Extract an integer cluster label from the signature's 'filename' field."""
    try:
        cluster_str = (sig.filename or "").strip()
    except Exception:
        cluster_str = ""

    if not cluster_str:
        logging.error(
            f"No cluster information found in signature {sig}. "
            f"Cluster info must be stored in the 'filename' field."
        )
        raise ValueError("Missing cluster label in signature 'filename'.")

    try:
        return int(cluster_str)
    except ValueError as e:
        logging.error(f"Clusters must be integers; got '{cluster_str}' for signature {sig}.")
        raise


def load_minhash_map(
    files_sig: Dict[str, str],
    *,
    ksize: int,
    seed: int = 42,
    scaled: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:

    mh_map: Dict[str, Dict[str, Any]] = {}

    for sid, path in files_sig.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{sid}: signature file not found: {path}")

        try:
            sig: sm.SourmashSignature = sm.load_one_signature(path)
        except Exception as e:
            raise RuntimeError(f"{sid}: failed to load signature from {path}: {e}") from e

        # Cluster label from 'filename'
        cluster = _get_cluster_from_signature(sig)

        mh = getattr(sig, "minhash", None)
        if mh is None or not isinstance(mh, sm.MinHash):
            raise ValueError(f"{sid}: no valid MinHash found in signature {path}.")

        # Validate core params
        if mh.ksize != ksize:
            raise ValueError(f"{sid}: ksize mismatch (have {mh.ksize}, want {ksize}).")
        if mh.seed != seed:
            raise ValueError(f"{sid}: seed mismatch (have {mh.seed}, want {seed}).")

        # Abundance tracking
        if not mh.track_abundance:
            # cannot infer counts retroactively
            raise ValueError(
                f"{sid}: signature does not track abundance."
            )

        mh = rescale_mh(mh, scaled)

        mh_map[sid] = {'mh': mh, 'cluster': cluster}

    return mh_map


# ----------------------------
# Signature writer (cluster in filename, abundance preserved)
# ----------------------------

def _as_signature(
    sid: str,
    obj: Union[SourmashSignature, sm.MinHash, Dict[str, Any]]
    ) -> SourmashSignature:
    """
    Normalize different input shapes to a SourmashSignature, ensuring:
      - signature.name = sid
      - signature.filename = str(cluster)   <-- cluster goes here
    """
    # Extract minhash and cluster depending on input type
    if isinstance(obj, SourmashSignature):
        mh = obj.minhash
        cluster = getattr(obj, "filename", None)
    elif isinstance(obj, sm.MinHash):
        mh = obj
        cluster = None
    elif isinstance(obj, dict):
        mh = obj.get("mh")
        cluster = obj.get("cluster", None)
    else:
        raise TypeError(f"{sid}: unsupported object type {type(obj)}")

    if mh is None or not isinstance(mh, sm.MinHash):
        raise ValueError(f"{sid}: missing/invalid MinHash")

    if cluster is None:
        raise ValueError(f"{sid}: cluster not provided; cannot write without cluster")

    s = SourmashSignature(
        mh,
        name=sid,
        filename=str(cluster)
    )

    return s


def write_sigs(
    data: Dict[str, Union[SourmashSignature, sm.MinHash, Dict[str, Any]]],
    outdir: str,
    *,
    subdir: str = "sigs"
) -> None:
    """
    Save {sample_id: obj} to outdir/<subdir>/<sample_id>.sig
    with cluster label encoded in Signature.filename and abundance preserved.
    """
    sig_dir = os.path.join(outdir, subdir)
    os.makedirs(sig_dir, exist_ok=True)

    for sid, obj in data.items():
        sig = _as_signature(sid, obj)
        outpath = os.path.join(sig_dir, f"{sid}.sig.gz")
        with gzip.open(outpath, "wt") as fp:
            sm.save_signatures([sig], fp)
