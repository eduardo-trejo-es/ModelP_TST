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



===============================================
Creer venv mac M1 arch -arm64 python3 -m venv .venv


to activate the virtual environnement on mac : source .venv/bin/activate
(on windows: source .venv/Scripts/activate )
to desactivate the virtual environnement on mac : deactivate

to update the requirements.txt: pip  freeze > requirements.txt
to use it : pip install -r requirements-m1-clean.txt

Local deploy