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
├── .gitattributes
├── data/                  # MassSpecGym data, helper files, and generated split TSVs
│   ├── MassSpecGym_S1.tsv
│   └── MassSpecGym_S23.tsv
├── ms_cp/
│   ├── retrieval/         # Retrieval model code (adapted from ms-mole)
│   │   ├── data.py
│   │   ├── data_module.py
│   │   ├── models.py
│   │   ├── loss.py
│   │   └── train_retriever.py
│   ├── mcp/               # Marginal conformal prediction
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
│   ├── cccp/              # Reserved for future use
│   └── ccpnn/             # Reserved for future use
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

### Helper files

Helper files must be downloaded from the [MassSpecGym HuggingFace repository](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym/tree/main) and placed in `data/`.

The retrieval and MCP code expect the following helper files:

* `fp_4096.npy`
* `inchis.npy`
* `MassSpecGym_retrieval_candidates_formula.json`
* `MassSpecGym_retrieval_candidates_formula_fps.npz`
* `MassSpecGym_retrieval_candidates_formula_inchi.npz`

---

## 3. Important Note on the Data Module

Use the local four-fold data module in `ms_cp/retrieval/data_module.py`. This version supports the folds `train`, `val`, `calib`, and `test`. It should be used instead of the original MassSpecGym data module with the same class name when running code that requires an explicit calibration split.

All imports should use:

```python
from ms_cp.retrieval.data_module import MassSpecDataModule
```

---

## 4. Training the Retrieval Model

Training and model code in `ms_cp/retrieval/` is adapted from [ms-mole](https://github.com/gdewael/ms-mole).
Training entry point:

```bash
python -m ms_cp.retrieval.train_retriever \
  data/MassSpecGym_S1.tsv \
  data \
  logs/retriever_S1 \
  --rankwise_loss cross \
  --rankwise_temp 0.5 \
  --rankwise_dropout 0.2 \
  --rankwise_projector False \
  --rankwise_listwise True \
  --bonus_challenge True \
  --bin_width 0.1 \
  --batch_size 64 \
  --n_layers 2 \
  --layer_dim 512 \
  --precision bf16-mixed
```

Positional arguments:

* `dataset_path` — path to the split TSV
* `helper_files_dir` — path to the helper-data directory
* `logs_path` — directory for TensorBoard logs and checkpoints

Common optional arguments:

* `--skip_test`, `--df_test_path`, `--bonus_challenge`
* `--bin_width`, `--batch_size`, `--devices`, `--precision`
* `--layer_dim`, `--n_layers`, `--dropout`, `--lr`
* `--bitwise_loss`, `--fpwise_loss`, `--rankwise_loss`
* `--checkpoint_path`, `--freeze_checkpoint`

Place trained checkpoints in `checkpoints/`.

---

## 5. Running MCP

Three nonconformity scores are implemented in `ms_cp/mcp/scores/`: **LAC**, **APS**, and **RAPS**.

Main entry point:

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

Required arguments:

| Flag | Description |
|------|-------------|
| `--dataset_tsv` | Path to a scenario-specific TSV split |
| `--helper_dir` | Directory with MassSpecGym helper files |
| `--checkpoint` | Trained retrieval model checkpoint (`.ckpt`) |
| `--score` | Nonconformity score: `lac`, `aps`, or `raps` |
| `--alpha` | Miscoverage level (e.g., `0.1` for 90% target coverage) |
| `--output_dir` | Directory to save results |

Common optional arguments:

* `--batch_size`, `--num_workers`, `--device`, `--seed`
* `--max_mz`, `--bin_width`, `--fp_size`

RAPS-specific optional arguments:

* `--tune_n`, `--lambda_grid`, `--randomized`, `--allow_zero_sets`

### MCP outputs

Each MCP run writes results to `--output_dir`:

* `config.json` — run configuration
* `calibration.json` — calibration state
* `per_sample.csv` — per-sample results
* `summary.json` and `summary.txt` — summary metrics

---

## Acknowledgements

The model architecture and training code are adapted from **[ms-mole](https://github.com/gdewael/ms-mole)** by De Waele et al.
This work builds on the **[MassSpecGym](https://github.com/pluskal-lab/MassSpecGym)** benchmark by Bushuiev et al.

---

## License

MIT License. See **[LICENSE](LICENSE)** for details.
