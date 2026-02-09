# BALF

This repository contains the implementation of **BALF**, a debiasing method for bundle recommendation. BALF is designed as a **dual level regularization-based framework** that can be seamlessly integrated into existing bundle recommendation backbone models and trained jointly with them.

The goal of BALF is to mitigate bias in bundle recommendation by incorporating additional regularization terms during model training, and to systematically evaluate the debiasing effectiveness across multiple OOD datasets.


## Overview

* **Method name**: BALF
* **Task**: Bundle Recommendation
* **Core idea**: Joint training of backbone bundle recommendation models with BALF regularization
* **Evaluation focus**: Debiasing effectiveness on multiple benchmark OOD datasets

BALF is model-agnostic and has been implemented on top of four representative bundle recommendation backbones:

* **BGCN_BALF**
* **CrossCBR_BALF**
* **MultiCBR_BALF**
* **BundleGT_BALF**

Each variant corresponds to applying the same BALF debiasing strategy to a different backbone model.


## Repository Structure

```text
.
├── BGCN_BALF/        # BALF integrated with BGCN
├── CrossCBR_BALF/    # BALF integrated with CrossCBR
├── MultiCBR_BALF/    # BALF integrated with MultiCBR
├── BundleGT_BALF/    # BALF integrated with BundleGT
└── README.md
```

Each subdirectory contains the code for a specific backbone augmented with BALF.


## Backbone Models

BALF is implemented on top of the following backbone models:

* **BGCN**
* **CrossCBR**
* **MultiCBR**
* **BundleGT**

These models are proposed in their respective original papers. Their implementations, environment configurations, and execution instructions are **not redefined** in this repository.


## Environment Setup

The required environment (e.g., Python version, dependencies, and libraries) for each backbone model is **identical to that of the original implementation**.

Please refer to the `README.md` files provided in the original backbone repositories or papers for detailed environment setup instructions.

BALF does **not** introduce additional environment requirements beyond those already specified by the corresponding backbone.


## How to Run

BALF follows the **same training and evaluation pipeline** as the original backbone models.

To run a BALF-enhanced model:

1. Navigate to the corresponding backbone directory (e.g., `CrossCBR_BALF/`).
2. Follow the **exact commands and procedures** described in the original backbone's README.
3. As an example, to run BALF on CrossCBR with the OOD Youshu dataset:
```bash
python train.py -m CrossCBR_BALF -d Youshu
```
    
For all other backbones and datasets, the execution commands and settings are identical to those specified in the corresponding original papers and repositories.


## Datasets

Experiments are conducted on several publicly available bundle recommendation datasets, including Youshu, NetEase, iFashion, and MealRec+.

All these datasets are originally provided in an IID setting. To ensure their suitability for evaluating the debiasing effectiveness of BALF, we apply additional preprocessing to transform each dataset into its corresponding OOD version.

The preprocessed datasets are organized separately for each backbone model and can be found in the corresponding directories:

* `BGCN_BALF/data`
* `CrossCBR_BALF/datasets`
* `MultiCBR_BALF/datasets`
* `BundleGT_BALF/datasets`

Unless otherwise specified, all experiments are conducted using these preprocessed versions of the datasets.


## Notes

* BALF is **backbone-agnostic** and can be extended to other bundle recommendation models with minimal modification.
* This repository focuses on methodological validation rather than reimplementation of backbone models.


## Acknowledgement

We sincerely thank the authors of BGCN, CrossCBR, MultiCBR, and BundleGT for making their code and datasets publicly available.