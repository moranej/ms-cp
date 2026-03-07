# ms-cp

**"Reliable Molecular Retrieval from Mass Spectra using Conformal Prediction"**.

This repository contains code for:
- dataset preparation based on the [MassSpecGym](https://github.com/pluskal-lab/MassSpecGym) benchmark
- retrieval model training adapted from [ms-mole](https://github.com/gdewael/ms-mole)
- marginal conformal prediction (MCP) for molecular retrieval using **LAC**, **APS**, and **RAPS**

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
