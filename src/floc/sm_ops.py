from __future__ import annotations

import os
import logging
from typing import Any, Dict, Tuple, Optional, Union

import sourmash as sm
from sourmash.signature import SourmashSignature
import screed


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

def _norm_moltype(s: Optional[str]) -> str:
    """Normalize molecule type to a canonical lower-case string."""
    if not s:
        return ""
    s = str(s).lower()
    # common aliases
    if s in {"dna", "dayhoff", "hp", "protein"}:
        return s
    # fall back to as-is string
    return s


def _check_or_downsample_to_scaled(mh: sm.MinHash, desired_scaled: int) -> sm.MinHash:
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


def _check_or_downsample_to_num(mh: sm.MinHash, desired_num: int) -> sm.MinHash:
    """Downsample to desired_num. Only possible if desired_num <= mh.num."""
    if mh.scaled:  # scaled-based sketches cannot be converted to num
        raise ValueError("Cannot convert a scaled MinHash to fixed-num.")
    if mh.num == desired_num:
        return mh
    if mh.num and desired_num <= mh.num:
        return mh.downsample(num=desired_num)
    raise ValueError(
        f"Cannot downsample to larger num (requested {desired_num}, have {mh.num})."
    )


# ----------------------------
# MinHash builders / loaders
# ----------------------------

def build_minhash_map(
    files_fx: Dict[str, str],
    ksize: int,
    *,
    scaled: Optional[int] = None,
    num: Optional[int] = None,
    seed: int = 42,
    moltype: str = "dna",
    track_abundance: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a MinHash for each input sequence file with the EXACT requested parameters:
      - ksize, seed, moltype, track_abundance
      - either scaled (recommended) or num (fixed-size)

    Returns:
      { sample_id: {'mh': MinHash, 'cluster': None}, ... }
    """
    if (scaled is None) == (num is None):
        raise ValueError("Specify exactly one of 'scaled' or 'num'.")

    mtype = _norm_moltype(moltype)
    mh_map: Dict[str, Dict[str, Any]] = {}

    for sid, path in files_fx.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{sid}: sequence file not found: {path}")

        seq_chunks = []
        for rec in screed.open(path):
            seq_chunks.append(rec.sequence if hasattr(rec, "sequence") else rec["sequence"])
        seq = "".join(seq_chunks)

        if scaled is not None:
            mh = sm.MinHash(
                n=0,
                ksize=ksize,
                seed=seed,
                scaled=scaled,
                track_abundance=track_abundance,
            )
        else:
            mh = sm.MinHash(
                n=num,
                ksize=ksize,
                seed=seed,
                track_abundance=track_abundance,
            )
        # set molecule type at the signature level when writing; MinHash carries k/seed/abundance.
        mh.add_sequence(seq, force=True)
        mh_map[sid] = {'mh': mh, 'cluster': None, 'moltype': mtype}
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
    moltype: str = "dna",
    track_abundance: bool = True,
    scaled: Optional[int] = None,
    num: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Load signatures and COERCE/VALIDATE them to match requested params.
    For scaled-based: will downsample to 'scaled' if needed (only to larger scaled).
    For num-based: will downsample to 'num' if needed (only to smaller num).

    Raises on incompatible inputs (different ksize/seed/molecule; wrong sketch type;
    trying to go to smaller scaled or larger num; enabling abundance when absent).
    """
    if (scaled is None) == (num is None):
        raise ValueError("Specify exactly one of 'scaled' or 'num'.")

    want_mol = _norm_moltype(moltype)
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

        # Molecule type (ask signature first, then mh if present)
        have_mol = _norm_moltype(getattr(sig, "moltype", None) or getattr(mh, "molecule", None))
        if have_mol and want_mol and have_mol != want_mol:
            raise ValueError(f"{sid}: molecule type mismatch (have {have_mol}, want {want_mol}).")

        # Abundance tracking
        if track_abundance and not mh.track_abundance:
            # cannot infer counts retroactively
            raise ValueError(
                f"{sid}: requested track_abundance=True but signature does not track abundance."
            )
        if not track_abundance and mh.track_abundance:
            # okay to accept; downstream can ignore counts

            pass

        # Coerce sketch type/size
        if scaled is not None:
            mh = _check_or_downsample_to_scaled(mh, scaled)
        else:
            mh = _check_or_downsample_to_num(mh, num)  # type: ignore[arg-type]

        mh_map[sid] = {'mh': mh, 'cluster': cluster, 'moltype': want_mol or have_mol}

    return mh_map


# ----------------------------
# Signature writer (cluster in filename, abundance preserved)
# ----------------------------

def _as_signature(
    sid: str,
    obj: Union[SourmashSignature, sm.MinHash, Dict[str, Any]],
    *,
    require_cluster: bool = True,
    moltype: str = "dna",
) -> SourmashSignature:
    """
    Normalize different input shapes to a SourmashSignature, ensuring:
      - signature.name = sid
      - signature.filename = str(cluster)   <-- cluster goes here
      - signature.moltype (if available) reflects requested type
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

    if cluster is None and require_cluster:
        raise ValueError(f"{sid}: cluster not provided; cannot write without cluster")

    filename_val = "" if cluster is None else str(cluster)
    s = SourmashSignature(
        mh,
        name=sid,
        filename=filename_val,  # store cluster label here
    )
    # Set moltype on the signature when possible (newer sourmash supports this metadata)
    try:
        s.moltype = _norm_moltype(moltype)  # type: ignore[attr-defined]
    except Exception:
        pass
    return s


def write_sigs(
    data: Dict[str, Union[SourmashSignature, sm.MinHash, Dict[str, Any]]],
    outdir: str,
    *,
    require_cluster: bool = True,
    subdir: str = "sigs",
    moltype: str = "dna",
) -> None:
    """
    Save {sample_id: obj} to outdir/<subdir>/<sample_id>.sig
    with cluster label encoded in Signature.filename and abundance preserved.
    """
    sig_dir = os.path.join(outdir, subdir)
    os.makedirs(sig_dir, exist_ok=True)

    for sid, obj in data.items():
        sig = _as_signature(sid, obj, require_cluster=require_cluster, moltype=moltype)
        outpath = os.path.join(sig_dir, f"{sid}.sig")
        with open(outpath, "wt") as fp:
            sm.save_signatures([sig], fp)
