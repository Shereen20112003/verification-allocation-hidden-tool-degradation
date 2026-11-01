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

The exploratory hosted-LLM pilot records are in `results/e7_pilot_records.json`. They include the synthetic prompts, responses, per-request usage, and evaluation results. Service/model identifiers, session timestamps, local paths, and raw service metadata are excluded. The pilot is a single exploratory run, not a multi-agent-system evaluation or evidence about deployed systems; reproducing it requires access to the original hosted inference service.

## Contents

- `experiments.py`: CPU-only synthetic experiments and figure generation
- `results/`: generated experiment outputs, the result summary, and privacy-reviewed E7 pilot records

## Authors

Shereen Anand — Google DeepMind (affiliation during this research); currently at Microsoft Research. This work was conducted independently of Microsoft Research.<br>
Prakash Chakraborty — Pennsylvania State University
