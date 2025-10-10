import os
import sys
import time
import argparse
import logging

from .utils import set_up_logging
from .io_ops import parse_input, write_cluster_csv
from .sm_ops import DistanceCache, build_minhash_map, load_minhash_map, write_sigs
from .cluster import group_clusters, assign_to_cluster, create_new_clusters, mds_plot
from .qc import global_containment, abundance_zscores

__version__ = "1.0"


def build_parser():
    p = argparse.ArgumentParser(
        description="floc: Genomic clustering tool for population-level surveillance"
    )
    p.add_argument(
        "-i", "--input", nargs="+", default=["./"],
        help="Directories containing input files (FASTA, FASTQ, or Sourmash signatures)"
    )
    p.add_argument("-o", "--outdir", default="./", help="Output directory")
    p.add_argument(
        "-d", "--distance", default=0.1, type=float,
        help="Distance threshold used to create clusters"
    )
    p.add_argument("--ksize", default=31, type=int, help="k-mer size")
    p.add_argument("--scaled", default=100, type=int, help="Scaling factor for signatures")
    p.add_argument("--batch", default=100, type=int, help="Number of sequences to cluster per batch")
    p.add_argument("--ignore-qc", action="store_true", help="Remove samples that fail QC without failing")
    p.add_argument("--skip-qc", action="store_true", help="Skip QC")
    p.add_argument("--abund-z", type=float, default=2.58, help="Z-score threshold used for quality filtering based on hash abundance")
    p.add_argument("--min-global-abund", type=float, default=0.05, help="Minimum abundance of a global hash to be included in global distance calculation")
    p.add_argument("--max-global-dist", type=float, default=0.5, help="Maximum fraction of sample hashes missing from global hashes after abundance filtering")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing signature files")
    p.add_argument("--mds-plot", dest="mds_plot", action="store_true", help="Create MDS plot. Can take a long time with many samples.")
    p.add_argument("--version", action="version", version=__version__)
    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    set_up_logging()
    logger = logging.getLogger(__name__)
    parser = build_parser()
    args = parser.parse_args(argv)

    start = time.time()
    logger.info(f"bb-clust v{__version__} starting")

    try:
        dist_cache = DistanceCache()

        # Parse inputs: separate FASTA/FASTQ and signatures
        input_fx, input_sig = parse_input(args.input, args.overwrite)
        logger.info(f"Found {len(input_fx)} FASTA/FASTQ and {len(input_sig)} signature files")

        # --- Build new MinHashes (FASTA/FASTQ) ---
        if input_fx:
            logger.info("Building new MinHash signatures from sequence files...")
            mh_fx = build_minhash_map(input_fx, args.ksize, scaled=args.scaled)
            logger.info(f"Built {len(mh_fx)} MinHash objects")
        else:
            mh_fx = {}
            logger.info("No FASTA/FASTQ inputs detected; skipping new MinHash generation.")

        # --- Load existing signatures ---
        if input_sig:
            logger.info("Loading existing signature files...")
            mh_sig = load_minhash_map(input_sig, ksize=args.ksize, scaled=args.scaled)
            logger.info(f"Loaded {len(mh_sig)} signature MinHashes")
        else:
            mh_sig = {}
            logger.info("No existing signature files found; starting from scratch.")

        # --- Proceed only if new sequences exist ---
        if not mh_fx:
            logger.warning("No new sequences to cluster. Nothing to do.")
            return 0

        # --- Quality control ---
        if not args.skip_qc:
            fx_keys = list(mh_fx.keys())

            # Filter samples with low global containment
            mh_global = mh_fx | mh_sig if mh_sig else mh_fx
            mh_fx = global_containment(mh_fx, mh_global, max_dist=args.max_global_dist, min_abund=args.min_global_abund, outfile=os.path.join(args.outdir, 'global_containment.csv'))

            # Filter samples with low kmer counts
            mh_global = mh_fx | mh_sig if mh_sig else mh_fx
            mh_fx = abundance_zscores(mh_global, list(mh_fx.keys()), threshold=args.abund_z, outfile=os.path.join(args.outdir, 'abundance_zscores.csv'))
            
            if not args.ignore_qc:
                if len(fx_keys) != len(mh_fx):
                    failed = '\n\t'.join([sid for sid in fx_keys if sid not in mh_fx]) + '\n'
                    logging.error(f"One or more sample failed QC. Remove them to continue or re-run with '--ignore-qc':\n\n\t{failed}")
                    exit()

        # --- If there are existing clusters, group them ---
        if mh_sig:
            mh_clust = group_clusters(mh_sig)
            logger.info(f"Grouped existing signatures into {len(mh_clust)} clusters.")
        else:
            mh_clust = {}
            logger.info("No pre-existing clusters detected.")

        # --- Assign new samples to clusters ---
        if mh_clust:
            logger.info("Assigning new sequences to existing clusters...")
            assigned, remainder = assign_to_cluster(mh_fx, mh_clust, args.distance)
            logger.info(f"Assigned {len(assigned)} sequences; {len(remainder)} unassigned remain.")

            # IMPORTANT: merge assigned into mh_clust so they appear in outputs
            for qid, rec in assigned.items():
                clabel = str(rec['cluster'])
                mh_clust.setdefault(clabel, {})[qid] = rec
        else:
            assigned, remainder = {}, mh_fx  # everything unassigned

        # --- Cluster the remaining sequences (new cluster creation) ---
        if remainder:
            logger.info("Clustering remaining sequences...")
            mh_clust = create_new_clusters(
                remainder, dist_cache, args.distance, args.batch,
                min_samples=1, existing=mh_clust
            )
            logger.info(f"Total clusters after processing: {len(mh_clust)}")
        else:
            logger.info("No remaining unassigned sequences to cluster.")

        # --- Flatten and write outputs ---
        if mh_clust:
            mh_out = {sid: rec for clabel, members in mh_clust.items() for sid, rec in members.items()}
            logger.info(f"Writing {len(mh_out)} signatures to {args.outdir}")
            write_sigs(mh_out, args.outdir)

            csv_path = write_cluster_csv(mh_clust, args.outdir)
            if csv_path:
                logger.info(f"Wrote cluster summary CSV → {csv_path}")

            if args.mds_plot:
                mds_plot(mh_out, dist_cache, save_html=os.path.join(args.outdir, 'pcoa.html'))
        else:
            logger.warning("No clusters found; nothing to write.")

        elapsed = time.time() - start
        logger.info(f"bb-clust completed successfully in {elapsed:.1f}s")
        return 0

    except Exception:
        logger.exception("Fatal error after %.1fs", time.time() - start)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
