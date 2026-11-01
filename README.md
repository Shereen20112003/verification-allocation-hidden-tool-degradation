# Verification Allocation in Stylized LLM Agent Pipelines under Hidden Tool Degradation

This repository provides reproducible synthetic experiments for a stylized verification-allocation model with hidden tool degradation. The analysis is conditional on the stated model; the experiments are not evaluations of deployed LLM systems.

The associated manuscript and its source files are not included in this repository.

## Reproducing the experiments

Requirements: Python 3.10 or later, NumPy, and Matplotlib. From the repository root:

```powershell
python -m pip install -r requirements.txt
python experiments.py
```

The script uses a fixed random seed and writes CSV summaries, figures, and `results/summary.json` to `results/`.

## Contents

- `experiments.py`: CPU-only synthetic experiments and figure generation
- `results/`: generated experiment outputs and the corresponding result summary

## Authors

Shereen Anand — Independent Researcher<br>
Prakash Chakraborty — Pennsylvania State University
