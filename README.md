# Radiology Explain AI 2.0 — MVP académico

Prototipo académico (UNIR, Maestría en Inteligencia Artificial) para revisar radiografías de tórax
con catorce observaciones CheXpert, incertidumbre calibrada, Grad-CAM auxiliar, preinforme
restringido, verificador factual y decisión humana obligatoria. No es un dispositivo médico
y no emite diagnósticos. Solo deben utilizarse imágenes públicas y desidentificadas.

Instalación: `pip install -r requirements.txt`
Ejecución: `streamlit run app.py`
Pruebas: `pytest -q tests`
Experimentos del capítulo 6: `python validacion/experimentos.py` (genera `validacion/resultados.json`).

Las versiones v0.1.0 y v0.1.1 se conservan en `validacion/referencia/` como líneas base de comparación.
