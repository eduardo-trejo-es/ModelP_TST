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



-------------------------------------------------------
Aquí te dejo un resumen completo y estructurado de lo trabajado hoy, para que puedas retomarlo fácilmente cuando quieras:

⸻

🧪 Resumen de la sesión de experimentación (Exp#18 → Exp#25)

🎯 Objetivo general

Maximizar la precisión direccional (>50%) del modelo PatchTST para predicción de tendencia en precios (Close), sin importar errores absolutos (MAE/RMSE).

⸻

✅ Experimento más exitoso (baseline actual):

📌 Exp #24

Parámetro	Valor
patch_len	10
embed_dim	128
n_layers	3
dropout_rate	0.1
epochs	30

Resultados:
	•	🟢 Precisión dirección: 50.7% y 50.97% en dos corridas
	•	🔵 Estabilidad visual en curvas
	•	🔁 Reproducible en múltiples corridas
	•	🏅 Designado baseline oficial

⸻

📉 Otros experimentos destacados:

🔸 Exp #18
	•	patch_len = 25, embed_dim = 256, n_layers = 2
	•	Precisión: ~48%
	•	Predicciones suaves y desfasadas

🔸 Exp #19, #20, #22
	•	Colapsos severos → salida constante (línea plana)
	•	Embed_dim demasiado alto o n_layers excesivo

🔸 Exp #21 y #23
	•	Modelos funcionales y estables
	•	Precisión: ~48-49.5%
	•	Buenas curvas, pero sin romper techo del 50%

🔸 Exp #25 (hoy)
	•	Igual a baseline pero con dropout_rate=0.2
	•	Precisión bajó a 48.67%
	•	Modelo reactivo pero con menor fidelidad

⸻

📌 Estado actual y próximos pasos sugeridos
	1.	✅ Baseline firme establecido en Exp #24
	2.	📊 Experimentos futuros deberían:
	•	Probar variantes sutiles de patch_len, n_layers
	•	Analizar impacto del tamaño de dataset o ventanas móviles
	•	Explorar regularización adaptativa o layer norm
	3.	💾 Todos los resultados están guardados en Models/experiments_patchtst_tf.csv y gráficos en Plots/

⸻

Estoy listo para continuar contigo donde lo dejemos, Alfredo. ¡Gran trabajo hoy! 🧠💪
Disfruta tu descanso, y retomamos cuando tú quieras.

¡Buena pregunta, Alfredo! Vamos a cerrar redondo con una propuesta clara para el experimento #26, partiendo directamente desde el baseline del exp#24 y los aprendizajes del exp#25:

⸻

🧪 Propuesta experimento #26

🎯 Objetivo:

Explorar si un ligero aumento de capacidad en el encoder mejora la precisión direccional, manteniendo el resto estable.

⸻

🔧 Hiperparámetros:

Parámetro	Valor	Justificación
patch_len	10	Igual que baseline
embed_dim	192	🔼 Subimos un poco desde 128
n_layers	3	Mantener profundidad óptima
dropout_rate	0.1	Igual que en baseline exitoso
epochs	30	Estable
exp_num	26	Nuevo experimento


⸻

🧠 Hipótesis:
	•	El modelo con embed_dim=192 tendrá mayor expresividad que el baseline, pero sin llegar al sobreajuste de embed_dim=256.
	•	Mantenemos patch_len=10 para seguir capturando señales locales efectivas.

⸻

¿Quieres que lo deje ya preparado para cuando retomes, o prefieres definir tú los ajustes al volver?