# ms-cp

**"Reliable Molecular Retrieval from Mass Spectra using Conformal Prediction"**.

This repository contains code for:
- dataset preparation based on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark
- retrieval model training adapted from [ms-mole](https://github.com/gdewael/ms-mole)
- marginal conformal prediction (MCP) for molecular retrieval using **LAC**, **APS**, and **RAPS**

---
## Repository structure

```text
ms-cp/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── MassSpecGym_S1.tsv
│   └── MassSpecGym_S23.tsv
├── retrieval/
│   ├── train_retriever.py
│   └── ...
├── mcp/
│   ├── main.py
│   ├── dataset.py
│   ├── model.py
│   ├── inference.py
│   ├── metrics.py
│   ├── reporting.py
│   └── scores/
│       ├── lac.py
│       ├── aps.py
│       └── raps.py
├── checkpoints/
│   └── .gitkeep
└── results/
    └── .gitkeep
```

---
## 1. Environment setup
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

This repository uses two dataset files which produced from MassSpecGym.tsv, based on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark:

* `data/MassSpecGym_S1.tsv` for **Scenario 1**
* `data/MassSpecGym_S23.tsv` for **Scenarios 2 and 3**
