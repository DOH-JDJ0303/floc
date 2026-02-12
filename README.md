# Floc :beer:
Floc is an iterative genome clustering tool. Genome assemblies are placed into new or existing clusters based on pairwise genetic distances. Genetic distance is estimated using the Sourmash k-mer containment method. 

:rainbow: **Fun Fact:** Floc is short for _flocculate_, which is a term often used in fermentation (like a **sourmash**!) to describe the coalescence of suspended particles (yeast, protein, etc.,)!

## Installation
### From source
```
git clone https://github.com/DOH-JDJ0303/floc
cd floc
pip install --upgrade pip
pip install .
```

### With Docker
```
docker build -t floc:latest .
docker run --rm -v "$PWD":/work -w /work floc:latest floc --help
```

## Usage
```
usage: floc [-h] [-o OUTDIR] [-d DISTANCE] [--ksize KSIZE] [--scaled SCALED] [--batch BATCH] [--ignore-qc] [--skip-qc] [--abund-z ABUND_Z] [--min-global-abund MIN_GLOBAL_ABUND] [--max-global-dist MAX_GLOBAL_DIST]
            [--overwrite] [--plot] [--pairwise] [--benchmark BENCHMARK] [--version]
            input [input ...]

floc: Genomic clustering tool for population-level surveillance

positional arguments:
  input                 Directories containing input files (FASTA, FASTQ, or Sourmash signatures)

options:
  -h, --help            show this help message and exit
  -o, --outdir OUTDIR   Output directory
  -d, --distance DISTANCE
                        Distance threshold used to create clusters
  --ksize KSIZE         k-mer size
  --scaled SCALED       Scaling factor for signatures
  --batch BATCH         Number of sequences to cluster per batch
  --ignore-qc           Remove samples that fail QC without failing
  --skip-qc             Skip QC
  --abund-z ABUND_Z     Z-score threshold used for sample-level quality filtering based on total hash abundance (default: 2.58). Similar to genome size but allows for comparison between FASTA, FASTQ, and existing
                        signatures. Higher values will increase the number of 'outliers'. Example - threshold of 2.58 includes ~99% of samples across a normal distribution.
  --min-global-abund MIN_GLOBAL_ABUND
                        Minimum abundance of a global hash to be included in global distance calculation (default: 0.05). Larger values can reduce the impact of uncommon genomic elements and contamination on cluster
                        assignment. Example - threshold of 0.05 requires that a hash is present in at least 5% of the samples.
  --max-global-dist MAX_GLOBAL_DIST
                        Maximum fraction of sample hashes missing from global hashes after abundance filtering (default: 0.5). Larger values require greater similarity between samples and vice versa. Example -
                        threshold of 0.5 requires at least 50% of the sample hashes are also in the filtered global hash.
  --overwrite           Overwrite existing signature files
  --plot                Create PCoA plot. Can take a long time with many samples.
  --pairwise            Compute pairwise distance matrix for each cluster with a new sample.
  --benchmark BENCHMARK
                        Comma separated file containing expected cluster values for each sample.
  --version             show program's version number and exit
```

## Outputs
|File / Directory|Description|
|-|-|
|`clusters.csv`|Cluster results for each sample in input|
|`sigs/*`|Sourmash signature files. These contain cluster information in the `filename` field and can be re-used for iterative clustering|
|`pcoa.html`|PCoA plot generated from all pairwise distances. Only created if run with `--plot`|
|`abundance_zscores.csv`|Per-sample summary of minhash abundance z-scores|
|`global_containment.csv`|Per-sample summary of minhash global containment|

## Example
### 1. Running for the first time
```
floc -i assemblies/ -o db/
```

### 2. Running any time after
```
floc -i new_assemblies/ db/sig/ -o db/
```


