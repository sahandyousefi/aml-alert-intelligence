# AML Alert Intelligence

An end-to-end decision system for anti-money-laundering (AML) alert handling: detect suspicious transactions, explain each alert, prioritize investigator workload, and draft Suspicious Activity Report (SAR) narratives.

## Problem

Banks generate far more anti-money-laundering alerts than investigators can review, and the large majority are false positives. Compliance teams spend their limited capacity on noise while real laundering goes unreported. This project builds an end-to-end decision system that combines predictive analytics, mathematical optimization, and AI to close that gap.

- **Predict:** a graph neural network scores each transaction's laundering risk.
- **Optimize:** a mixed-integer program assigns alerts to investigators to maximize expected illicit dollars caught under capacity, skill, and SLA constraints.
- **AI-assisted reporting:** a language model drafts evidence-grounded SAR narratives, and every alert is explained for human review.

## Why it matters

Money laundering finances fraud, trafficking, and other serious crime, and it undermines the integrity of the U.S. financial system. Integrating predictive models with optimization-driven decisions lets financial institutions, including community banks and credit unions with small compliance teams, find more illicit activity with the same resources. Investigators stay accountable for every final decision.

## Approach

| Stage | Method | Output |
|---|---|---|
| Detect | GraphSAGE, GAT, and a temporal GNN variant, benchmarked against XGBoost on engineered features | Laundering risk score per transaction |
| Explain | GNNExplainer and Captum for the GNN, SHAP for XGBoost | Evidence subgraph and key drivers per alert |
| Prioritize | Mixed-integer program for alert assignment, plus decision threshold optimization | Investigator worklists under capacity, skill, and SLA constraints |
| Report | Small open-source language model grounded in structured evidence | Draft SAR narratives for investigator review |

## Success metrics

| Stage | Metrics | Comparison |
|---|---|---|
| Detect | PR-AUC, minority-class F1, recall@k (k = daily alert budget) | GNN vs. XGBoost on the same chronological test split |
| Explain | Fidelity, sparsity | Small explanations that actually drive the prediction |
| Prioritize | Illicit dollars caught, SLA violations, investigator utilization | MILP vs. greedy top-k vs. fixed threshold under the same constraints |
| Report | Fact precision (hard gate), LLM-judge score | Zero untraceable numbers or entities |
| System | Reproducibility | One command runs the full pipeline from a config file |

Numeric targets are not set until baseline results exist.

## Data

[IBM Transactions for Anti-Money Laundering (AMLworld)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml), a synthetic transaction dataset with laundering labels (Altman et al., NeurIPS 2023 Datasets and Benchmarks). The primary dataset is HI-Small.

The data is synthetic and contains no real customer information. It is not stored in this repository; it is downloaded through the Kaggle API.

## Repository structure

```
aml-alert-intelligence/
├── configs/                  # YAML configs: paths, seeds, hyperparameters
├── docs/                     # Architecture diagram, model card
├── notebooks/                # Exploration and experiments
├── scripts/                  # Pipeline entry points
├── src/aml_alert_intelligence/  # Package source code
├── tests/                    # Unit tests (pytest)
├── pyproject.toml
└── requirements.txt
```

## Setup (Google Colab)

```bash
git clone https://github.com/sahandyousefi/aml-alert-intelligence.git
cd AML-alert-intelligence
pip install -e .
pip install -r requirements.txt
```

Restart the Colab session after installing so the package can be imported.

Kaggle access: store your API token in Colab Secrets as `KAGGLE_API_TOKEN` and `KAGGLE_USERNAME`. Never commit credentials.

## Project status

| Phase | Scope | Status |
|---|---|---|
| 0 | Framing and setup | In progress |
| 1 | Data pipeline and exploratory analysis (PySpark) | Planned |
| 2 | Feature engineering and XGBoost baseline | Planned |
| 3 | Graph construction (PyTorch Geometric) | Planned |
| 4 | GNN modeling and benchmark | Planned |
| 5 | Explainability | Planned |
| 6 | Alert triage optimization (MILP) | Planned |
| 7 | SAR narrative generation and evaluation | Planned |
| 8 | End-to-end pipeline and investigator dashboard | Planned |
| 9 | Testing, documentation, and model card | Planned |

## Results

No results yet. Results will be added as each phase is completed.

## Responsible use

This system supports investigators. It does not make filing decisions. Every alert comes with an explanation, and every drafted narrative is checked against the underlying evidence before review. A model card documenting limitations and intended use will be added in Phase 9.

## License

MIT
