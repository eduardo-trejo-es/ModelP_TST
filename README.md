# ModelP_TST
Patch TST model implementation


ModelP_TST is a project focused on exploring and applying the PatchTST (Patch Time Series Transformer) model — a deep learning architecture specifically designed for time series forecasting.

PatchTST is inspired by Vision Transformers (ViT), but adapted to handle sequential temporal data instead of images. The key idea is that it divides the time series into small "patches" (segments) and applies attention mechanisms across these patches. This approach allows the model to efficiently capture long-term dependencies, much better than traditional methods like sliding windows or recurrent networks (e.g., LSTM).

Core Concepts Behind PatchTST:
Input Structure: Instead of feeding one timestep at a time (like RNNs/LSTMs), the model processes patches (segments) of the time series at once.

Pure Attention: The model is fully based on self-attention mechanisms, with no convolutional or recurrent components.

Generalization Ability: By working with patches, the model captures both local and global patterns effectively.

Flexibility: PatchTST can be used for both univariate and multivariate forecasting tasks.

The main goals of ModelP_TST are to:

Understand the theoretical foundations of PatchTST.

Implement and experiment with PatchTST on different datasets.

Focus particularly on:

Financial time series forecasting (e.g., oil prices, stocks).

Industrial or engineering signal forecasting (e.g., temperature control systems, sensor data).


ModelP_TST/
│
├── README.md          # Project overview and instructions
├── requirements.txt   # Python dependencies
├── config/             # Configuration files (hyperparameters, model settings)
│    └── config.yaml
├── data/               # Datasets (raw and processed)
│    ├── raw/
│    └── processed/
├── notebooks/          # Jupyter notebooks for exploration, testing, visualization
│    └── 01_initial_exploration.ipynb
├── src/                # Source code
│    ├── models/        # Model architecture (PatchTST implementation/adaptation)
│    │    └── patchtst.py
│    ├── data_utils/    # Data loading, preprocessing, patching
│    │    └── dataset.py
│    ├── train/         # Training loop, loss functions, evaluation metrics
│    │    ├── trainer.py
│    │    └── metrics.py
│    └── utils/         # General utilities (plots, logs, etc.)
│         └── plotting.py
├── experiments/        # Training results, saved models, logs
│    ├── models/
│    └── logs/
└── scripts/            # Scripts to run training, evaluation
     ├── train_model.py
     └── evaluate_model.py
