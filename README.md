# HFEDTI: A DTI Prediction Model Integrating Local-Global Feature Fusion and Weighted Ensemble Learning
HFEDTI is a deep learning framework for predicting drug–target interactions (DTIs). It combines local-global feature extraction, attention-based heterogeneous fusion, and a performance-weighted ensemble strategy to improve robustness and generalization in DTI prediction.

This repository contains the full implementation, pretrained model scripts, and data preprocessing utilities.


## Overview

HFEDTI is a deep learning model for drug–target interaction (DTI) prediction. It addresses two major challenges in DTI tasks: the inability of single models to capture heterogeneous sequence features and the insufficient fusion of local and global information. HFEDTI integrates residual CNNs and hierarchical BiLSTMs to extract local and global features, respectively. A hierarchical attention mechanism aligns and fuses multi-level cross-modal representations. A weighted ensemble strategy based on validation performance improves robustness and generalization. The model outperforms several state-of-the-art methods, especially under cold-start settings.


## Environment
HFEDTI has been tested under the following environment:
OS: Ubuntu 22.04
GPU: NVIDIA RTX 4090
Framework: PyTorch 2.2.0
Python: 3.10+
numpy==1.24.4  
prefetch_generator==1.0.3  
scikit_learn==1.2.0  
torch==2.2.0a0+81ea7a4  
tqdm==4.66.1  


## Case Study Tools

This study employed the following third-party software tools for molecular docking and visualization:

- **OpenBabel** (version 3.1.1): Used to convert drug 3D structure files from SDF format to PDB format.
- **AutoDockTools** (version 1.5.7): Utilized to perform docking between protein targets and drug molecules.
- **PyMOL** (version 2.4): Used for visualizing the molecular docking results.

These tools supported the structural validation and interpretability of the predicted drug–target interactions.


## Run the Model
Once dependencies are installed and data is prepared, you can run the full pipeline (preprocessing → training → evaluation) with a single command:
python RunModel.py


