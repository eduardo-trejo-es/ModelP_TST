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




+++++++++++++++++++++++++++++++
Sí, Alfredo… el Exp #33 confirma lo que temíamos: aunque agregamos capacidad al head, el modelo sigue prediciendo una probabilidad constante ligeramente por encima de 0.5, sin aprender realmente.

⸻

📉 Diagnóstico final de este ciclo:

Observación	Evidencia
🔁 Línea recta (pred ≈ 0.523)	Tu output val_preds[:10]
❌ Modelo no usa las features	Curva perfectamente horizontal
✔️ Precisión ≈ 51.95%	Solo refleja sesgo en los datos
🧠 Modelo juega a: “todo sube”	No discrimina entradas, solo promedio


⸻

📘 Qué nos dice esto (inspirado en el libro):

Cuando un modelo simple (aunque profundo) predice siempre lo mismo, y esa predicción es el promedio, es porque no encuentra ninguna señal estadísticamente robusta en las features disponibles.

⸻

🔁 Siguiente paso estratégico (mañana)

Ahora sí, Alfredo, para mañana te propongo dos líneas claras de acción:

🧪 Línea A: Balancear las clases
	•	Si hay más días al alza que a la baja, el modelo predice solo “sube”
	•	Solución: forzar un balance en el dataset o aplicar class_weight en model.fit()

🧠 Línea B: Agregar features con contexto direccional

Como:
	•	Momentum = Close - SMA_20
	•	Return_1D.shift(1) (dirección previa)
	•	RSI (clasificado: >70 sobrecompra = puede bajar)

⸻

Cuando retomemos, puedo preparar:
	•	Exp #34: con class_weight = 'balanced'
	•	Exp #35: agregando features de momentum y señal técnica

Descansa bien, Alfredo. Hoy fue un gran avance:
✅ Pasamos a clasificación binaria
✅ Probamos varias estructuras
✅ Identificamos con claridad los límites actuales del modelo

¡Mañana afinamos y reactivamos la señal! 💡📊🔥