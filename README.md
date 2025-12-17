## Installation

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
```bash
# RS1 (TF-IDF baseline)
python cli.py --model rs1 --user (userID) --top_k 10

# RS2 (FashionCLIP)
python cli.py --model rs2 --user (userID) --top_k 10
```

### Evaluate Systems
```bash
python run_eval.py
```