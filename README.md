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
usage: floc [-h] [-o OUTDIR] [-d DISTANCE] [--ksize KSIZE] [--scaled SCALED] [--batch BATCH] [--ignore-qc] [--skip-qc] [--min-hash-freq MIN_HASH_FREQ] [--min-hash-frac MIN_HASH_FRAC] [--overwrite] [--plot] [--pairwise]
            [--version]
            input [input ...]

floc: Genomic clustering tool for population-level surveillance

positional arguments:
  input                 Directories containing input files (FASTA, FASTQ, or Sourmash signatures)

options:
  -h, --help            show this help message and exit
  -o, --outdir OUTDIR   Output directory
  -d, --distance DISTANCE
                        Distance threshold used to create clusters (default: 0.03)
  --ksize KSIZE         k-mer size
  --scaled SCALED       Scaling factor for signatures
  --batch BATCH         Number of sequences to cluster per batch
  --ignore-qc           Remove samples that fail QC without failing
  --skip-qc             Skip QC
  --min-hash-freq MIN_HASH_FREQ
                        Minimum frequency for a hash to be included in the global hashes
  --min-hash-frac MIN_HASH_FRAC
                        Minimum fraction of sample hashes shared with the global hashes after filtering
  --overwrite           Overwrite existing signature files
  --plot                Create PCoA plot. Can take a long time with many samples.
  --pairwise            Compute pairwise distance matrix for each cluster with a new sample.
  --version             show program's version number and exit
```

## Outputs
|File / Directory|Description|
|-|-|
|`clusters.csv`|Cluster results for each sample in input|
|`sigs/*`|Sourmash signature files. These contain cluster information in the `filename` field and can be re-used for iterative clustering|
|`pcoa.html`|PCoA plot generated from all pairwise distances. Only created if run with `--plot`|
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