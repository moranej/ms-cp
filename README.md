# ms-cp

**"Reliable Molecular Retrieval from Mass Spectra using Conformal Prediction"**.

This repository contains code for:

- Dataset preparation based on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark
- Retrieval model training adapted from [ms-mole](https://github.com/gdewael/ms-mole)
- Marginal conformal prediction (MCP) for molecular retrieval using **LAC**, **APS**, and **RAPS**

---

## Repository Structure

```text
ms-cp/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── MassSpecGym_S1.tsv
│   └── MassSpecGym_S23.tsv
├── ms_cp/
│   ├── retrieval/
│   │   ├── data.py
│   │   ├── data_module.py
│   │   ├── models.py
│   │   ├── loss.py
│   │   └── train_retriever.py
│   ├── mcp/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── inference.py
│   │   ├── metrics.py
│   │   ├── reporting.py
│   │   ├── utils.py
│   │   └── scores/
│   │       ├── base.py
│   │       ├── lac.py
│   │       ├── aps.py
│   │       └── raps.py
│   ├── cccp/          # coming soon
│   └── ccpnn/         # coming soon
├── examples/
│   └── run_mcp.py
├── checkpoints/
│   └── .gitkeep
└── results/
    └── .gitkeep
```

---

## 1. Environment Setup

```bash
conda create -n ms-cp python=3.11
conda activate ms-cp

# Install PyTorch with CUDA support
conda install pytorch=2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install MassSpecGym (dependency)
pip install massspecgym

# Clone and install remaining dependencies
git clone https://github.com/moranej/ms-cp.git
cd ms-cp
pip install -r requirements.txt
```

---

## 2. Data

This repository uses two dataset files produced from `MassSpecGym.tsv`, based on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark:

* `data/MassSpecGym_S1.tsv` for **Scenario 1**
* `data/MassSpecGym_S23.tsv` for **Scenarios 2 and 3**

Helper files (precomputed fingerprints, InChIKeys, candidate sets) must be downloaded separately from the [MassSpecGym HuggingFace repository](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/main) and placed in `data/`.

---

## 3. Retrieval Model

Training and model code in `ms_cp/retrieval/` is adapted from [ms-mole](https://github.com/gdewael/ms-mole). Note that `data_module.py` is a modified version of the original MassSpecGym data module, updated to support four folds (`train`, `val`, `calib`, `test`) instead of three.

### Training

```bash
python -m ms_cp.retrieval.train_retriever \
  data/MassSpecGym_S1.tsv \
  data/ \
  logs/retriever_S1 \
  --rankwise_loss cross \
  --batch_size 128 \
  --lr 0.0001
```

Place trained checkpoints in `checkpoints/`.

---

## 4. Marginal Conformal Prediction (MCP)

Three nonconformity scores are implemented in `ms_cp/mcp/scores/`: **LAC**, **APS**, and **RAPS**.

### Running MCP

```bash
python -m ms_cp.mcp.main \
  --dataset_tsv  data/MassSpecGym_S1.tsv \
  --helper_dir   data/ \
  --checkpoint   checkpoints/R3_10_ranker1.ckpt \
  --score        raps \
  --alpha        0.1 \
  --output_dir   results/mcp_raps_S1 \
  --device       cuda
```

| Flag | Description |
|------|-------------|
| `--dataset_tsv` | Path to a scenario-specific TSV split |
| `--helper_dir` | Directory with MassSpecGym helper files |
| `--checkpoint` | Trained retrieval model checkpoint (`.ckpt`) |
| `--score` | Nonconformity score: `lac`, `aps`, or `raps` |
| `--alpha` | Miscoverage level (e.g., `0.1` for 90% target coverage) |
| `--output_dir` | Directory to save results |

RAPS-specific options: `--tune_n`, `--lambda_grid`, `--randomized`, `--allow_zero_sets`.

Outputs are saved to `--output_dir`: `per_sample.csv`, `summary.json`, `summary.txt`, `calibration.json`, and `config.json`.

---

## Acknowledgements

The retrieval model architecture is adapted from **[ms-mole](https://github.com/gdewael/ms-mole)** by De Waele et al.
This work builds on the **[MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)** benchmark by Bushuiev et al.

---

## Citation
```

---

## License

MIT License. See **[LICENSE](LICENSE)** for details.
