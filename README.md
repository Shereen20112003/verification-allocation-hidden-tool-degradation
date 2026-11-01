# Verification Allocation in Stylized LLM Agent Pipelines under Hidden Tool Degradation

Reproducible synthetic experiments for a discrete-time, stylized verification-allocation model with binary inherited corruption and hidden tool degradation. The results are checks of this model. They are not evaluations of deployed language-model systems and do not establish a general mean-field-game theorem.

The analysis concerns serial and multi-role propagation, a finite-population deviation bound under explicit independence assumptions, and verification choices that do not affect future errors or observations. Under the stated restrictions, the optimal choice is myopic. The experiments report where the stated threshold and interval structure hold and where synthetic counterexamples leave those restrictions.

The manuscript source is maintained separately and is not included in this code repository.

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

- Shereen Anand — affiliation during this research: Google DeepMind. Currently at Microsoft Research; this work was conducted independently of Microsoft Research.
- Prakash Chakraborty — Pennsylvania State University.
