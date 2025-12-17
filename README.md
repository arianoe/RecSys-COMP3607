## Installation

Clone this repository and go into the folder.

### Set up (Windows - Conda)

1. Create environment
```bash
conda create -n (env-name) python=3.11 -y
conda activate (env-name)
```

2. Install annoy from conda-forge (required for fashion-clip)
```bash
conda install -c conda-forge python-annoy -y
```

3. Install remaining dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Generate Recommendations

Some user IDs to test:

| user_id                         | total_interactions | pos_interactions | avg_rating |
|--------------------------------|--------------------|------------------|------------|
| AGQINWGQVZAEVOWX53EULJRN7NZQ     | 5                  | 5                | 5.000000   |
| AG7IMLEUU735AHDE3FAK6UXU7RMQ     | 4                  | 4                | 4.750000   |
| AE6KFN34L2VZMLL2RZXJLBSAJYWA     | 4                  | 3                | 3.500000   |
| AE7SHTRCMCKRBA2AGS3A5M5CQWCQ     | 3                  | 3                | 4.333333   |
| AEID6FXIYSRLHBMW447FKYWDMV6A     | 3                  | 3                | 5.000000   |


```bash
# RS1 (TF-IDF baseline)
python cli.py --model rs1 --user <user_id> --top_k 10

# RS2 (FashionCLIP)
python cli.py --model rs2 --user <user_id> --top_k 10
```

### Evaluate Systems
```bash
python run_eval.py
```

### Note
In the first run of RS2 with FashionCLIP, the embeddings will take some time to run. After one run, everything will be saved in embeddings folder.