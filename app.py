"""
Radiology Explain AI 2.0 — demo Streamlit del MVP (prototipo académico, UNIR).

Flujo: entrada controlada -> inferencia -> calibración e incertidumbre ->
Grad-CAM auxiliar -> preinforme restringido + verificador factual -> decisión humana.

NO es un dispositivo médico ni emite diagnósticos. Usar solo imágenes públicas
y desidentificadas.

Ejecutar:  streamlit run app.py
"""
import datetime as dt
import difflib
import hashlib
import io
import json
import re

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageFilter

APP_VERSION = "0.1.2-demo"
TEMPLATE_VERSION = "plantilla-v1"

# Orden estándar de salida CheXpert (debe coincidir con la cabeza del checkpoint)
LABELS = [
    ("No Finding", "Sin hallazgos", "ausencia"),
    ("Enlarged Cardiomediastinum", "Cardiomediastino ensanchado", "patologia"),
    ("Cardiomegaly", "Cardiomegalia", "patologia"),
    ("Lung Opacity", "Opacidad pulmonar", "patologia"),
    ("Lung Lesion", "Lesión pulmonar", "patologia"),
    ("Edema", "Edema", "patologia"),
    ("Consolidation", "Consolidación", "patologia"),
    ("Pneumonia", "Neumonía", "patologia"),
    ("Atelectasis", "Atelectasia", "patologia"),
    ("Pneumothorax", "Neumotórax", "patologia"),
    ("Pleural Effusion", "Derrame pleural", "patologia"),
    ("Pleural Other", "Otra alteración pleural", "patologia"),
    ("Fracture", "Fractura", "patologia"),
    ("Support Devices", "Dispositivos de soporte", "dispositivo"),
]

# Vocabulario que el verificador reconoce como afirmación sobre cada observación
PATTERNS = {
    "No Finding": r"sin hallazgos",
    "Enlarged Cardiomediastinum": r"cardiomediastino ensanchado|ensanchamiento (del )?(cardio)?mediast",
    "Cardiomegaly": r"cardiomegalia|silueta card[ií]aca aumentada",
    "Lung Opacity": r"opacidad",
    "Lung Lesion": r"lesi[oó]n pulmonar|n[oó]dulo|masa pulmonar",
    "Edema": r"\bedema\b",
    "Consolidation": r"consolidaci[oó]n",
    "Pneumonia": r"neumon[ií]a",
    "Atelectasis": r"atelectasia",
    "Pneumothorax": r"neumot[oó]rax",
    "Pleural Effusion": r"derrame( pleural)?",
    "Pleural Other": r"otra alteraci[oó]n pleural|engrosamiento pleural",
    "Fracture": r"fractura",
    "Support Devices": r"dispositivos? de soporte|sonda|cat[eé]ter|marcapasos|tubo endotraqueal|c[aá]nula",
}
# Marcadores de polaridad. La negación se resuelve por alcance: cada mención toma la
# polaridad del marcador más cercano que la precede dentro de la misma oración, de modo
# que «No se observa neumotórax, se identifica derrame» no niega el derrame.
NEGATION = re.compile(
    r"\b(no se (observa|observan|identifica|identifican|aprecia|aprecian|evidencia|evidencian)|no hay|"
    r"sin (evidencia|signos|datos) de|se descarta|descarta|ausencia de|negativo para|sin|ni)\b"
)
AFFIRMATION = re.compile(
    r"\b(se (observa|observan|identifica|identifican|aprecia|aprecian|evidencia|evidencian)|"
    r"hay|presenta|compatible con|sugestivo de|sugiere|con)\b"
)
# Negación pospuesta dentro del mismo sintagma («Opacidad pulmonar: descartada», «derrame ausente»)
POST_NEGATION = re.compile(r"^[^.:,;]{0,30}[:,]?\s*(ausente|no (visible|identificad[oa])|descartad[oa])\b")
STATE_TXT = {"alto": "Soporte alto", "incierto": "Incierta: revisar", "bajo": "Soporte bajo"}

# Zonas aproximadas (x, y normalizados) para el mapa simulado
SIM_CENTERS = {
    "No Finding": [(0.5, 0.5)], "Enlarged Cardiomediastinum": [(0.5, 0.38)],
    "Cardiomegaly": [(0.55, 0.62)], "Lung Opacity": [(0.3, 0.55)],
    "Lung Lesion": [(0.66, 0.4)], "Edema": [(0.3, 0.5), (0.7, 0.5)],
    "Consolidation": [(0.3, 0.62)], "Pneumonia": [(0.3, 0.66)],
    "Atelectasis": [(0.32, 0.72)], "Pneumothorax": [(0.7, 0.2)],
    "Pleural Effusion": [(0.24, 0.8)], "Pleural Other": [(0.76, 0.7)],
    "Fracture": [(0.18, 0.35)], "Support Devices": [(0.5, 0.28)],
}


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def img_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_upload(file) -> np.ndarray:
    """Convierte a escala de grises de 8 bits sin saturar PNG de 16 bits y respetando EXIF."""
    from PIL import ImageOps

    im = ImageOps.exif_transpose(Image.open(file))
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        # Se reescala según la profundidad de bits efectiva (10, 12, 14 o 16) y no según los
        # valores extremos, para no convertir el fondo en saturación artificial.
        a = np.array(im, dtype=np.float64).clip(0, None)
        bits = next((b for b in (8, 10, 12, 14, 16) if a.max() < 2 ** b), 16)
        return np.round(a / (2 ** bits - 1) * 255).clip(0, 255).astype(np.uint8)
    return np.array(im.convert("L"))


def phantom(seed: int, size: int = 512) -> np.ndarray:
    """Maniquí sintético con forma de tórax (no es imagen de paciente)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    img = np.full((size, size), 0.08)
    body = (x / 0.86) ** 2 + (y / 0.98) ** 2 < 1
    img[body] = 0.55
    for cx in (-0.38, 0.38):
        lung = ((x - cx) / 0.3) ** 2 + ((y + 0.05) / 0.62) ** 2 < 1
        img[lung] = 0.2
    img[(np.abs(x) < 0.12) & (y < 0.25) & body] = 0.62
    heart = ((x - 0.08) / 0.28) ** 2 + ((y - 0.3) / 0.24) ** 2 < 1
    img[heart] = 0.66
    img[(y > 0.58) & body] = 0.7
    for i in range(9):
        rib = np.abs(y - (-0.7 + i * 0.15 + 0.18 * x ** 2)) < 0.012
        img[rib & body & (np.abs(x) > 0.1)] += 0.12
    if rng.random() < 0.7:  # opacidad sintética aleatoria
        cx, cy = rng.uniform(-0.55, 0.55), rng.uniform(-0.3, 0.5)
        img += 0.25 * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / 0.02))
    img += rng.normal(0, 0.02, img.shape)
    out = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.array(out.filter(ImageFilter.GaussianBlur(2.2)))


def unfit_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    return (128 + rng.normal(0, 3, (200, 200))).clip(0, 255).astype(np.uint8)


def quality_checks(arr: np.ndarray):
    h, w = arr.shape
    sat = float(((arr < 5) | (arr > 250)).mean())
    return [
        ("Resolución mínima (≥256 px)", min(h, w) >= 256, f"{w}×{h}"),
        ("Contraste suficiente (σ ≥ 20)", arr.std() >= 20, f"σ = {arr.std():.1f}"),
        ("Saturación < 30 %", sat < 0.30, f"{sat:.0%}"),
        ("Proporción plausible", 0.6 <= w / h <= 1.6, f"{w / h:.2f}"),
    ]


def colormap(v: np.ndarray) -> np.ndarray:
    stops = np.array([[0, 0, .5], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0]])
    pos = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[..., None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def overlay(arr: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    h, w = arr.shape
    cam = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)) / 255.0
    base = np.repeat(arr[..., None] / 255.0, 3, axis=2)
    a = (alpha * cam)[..., None]
    return ((base * (1 - a) + colormap(cam) * a) * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Motores de inferencia
# --------------------------------------------------------------------------- #
def simulated_raw_logits(h: str) -> np.ndarray:
    """Logits deterministas por imagen. NO provienen de un modelo entrenado."""
    rng = np.random.default_rng(int(h[:12], 16))
    prior = np.array([0.0, -1.6, -0.9, -0.4, -1.8, -1.2, -1.5, -1.4, -0.8, -2.0, -0.6, -2.4, -2.6, -0.5])
    z = prior + rng.normal(0, 1.3, 14)
    z[0] = -0.2 - 1.2 * max(0.0, z[1:13].max())  # coherencia con "Sin hallazgos"
    return 2.2 * z  # sobreconfianza deliberada, para que la calibración sea visible


def simulated_cam(h: str, label: str, grid: int = 64) -> np.ndarray:
    # hashlib en lugar de hash(): hash() de Python cambia entre procesos y rompería el determinismo
    salt = int(hashlib.sha256(label.encode()).hexdigest()[:6], 16)
    rng = np.random.default_rng(int(h[12:24], 16) + salt)
    y, x = np.mgrid[0:1:grid * 1j, 0:1:grid * 1j]
    cam = np.zeros((grid, grid))
    for cx, cy in SIM_CENTERS[label]:
        cx, cy = cx + rng.normal(0, 0.04), cy + rng.normal(0, 0.04)
        s = 0.2 if label == "No Finding" else 0.07
        cam += np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * s ** 2)))
    if rng.random() < 0.35:  # artefacto fuera de anatomía, como advierten los usuarios
        cam += 0.6 * np.exp(-(((x - rng.uniform(0, 1)) ** 2 + (y - 0.03) ** 2) / 0.002))
    return cam / cam.max()


@st.cache_resource(show_spinner="Cargando checkpoint DenseNet121…")
def load_densenet(path: str):
    import torch
    import torchvision

    model = torchvision.models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(1024, 14)
    try:  # carga segura primero; solo se admite un módulo completo si el archivo lo requiere
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        obj = obj.state_dict()
    for k in ("state_dict", "model_state_dict", "model"):
        if isinstance(obj, dict) and isinstance(obj.get(k), dict):
            obj = obj[k]
            break
    legacy = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
    sd = {}
    for k, v in obj.items():
        k = re.sub(r"^(module\.|model\.)+", "", k)
        m = legacy.match(k)
        sd[m.group(1) + m.group(2) if m else k] = v
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if any(k.startswith("classifier") for k in missing):
        raise ValueError("El checkpoint no contiene una cabeza de 14 salidas compatible.")
    if len(missing) > 0.1 * len(model.state_dict()):
        raise ValueError(f"Faltan {len(missing)} pesos del extractor; el checkpoint no corresponde a DenseNet121.")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()[:12]
    return model.eval(), len(missing), len(unexpected), digest


def _to_tensor(arr: np.ndarray, size: int):
    import torch

    x = np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    x = np.repeat(x[None], 3, axis=0)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return torch.from_numpy((x - mean) / std)[None]


@st.cache_data(show_spinner="Ejecutando DenseNet121…")
def densenet_logits(_model, key: str, arr: np.ndarray, size: int) -> np.ndarray:
    import torch

    with torch.no_grad():
        return _model(_to_tensor(arr, size))[0].numpy()


@st.cache_data(show_spinner="Calculando Grad-CAM…")
def densenet_gradcam(_model, key: str, arr: np.ndarray, size: int, k: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    for prm in _model.parameters():
        prm.requires_grad_(False)
    feats = _model.features(_to_tensor(arr, size)).detach().requires_grad_(True)
    act = F.relu(feats)
    act.retain_grad()
    logits = _model.classifier(F.adaptive_avg_pool2d(act, 1).flatten(1))
    logits[0, k].backward()
    w = act.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * act).sum(1))[0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


# --------------------------------------------------------------------------- #
# Calibración, abstención, preinforme y verificador
# --------------------------------------------------------------------------- #
def build_outputs(raw: np.ndarray, T: float, t_low: float, t_high: float):
    outs = []
    for (en, es, kind), z in zip(LABELS, raw):
        p = float(sigmoid(z / T))
        state = "alto" if p >= t_high else "bajo" if p < t_low else "incierto"
        outs.append(dict(en=en, es=es, kind=kind, p_raw=float(sigmoid(z)), p=p, state=state))
    return outs


def abstention_reasons(outs, qc, max_unc):
    reasons = [f"Imagen no apta: {n.lower()} ({d})" for n, ok, d in qc if not ok]
    # Solo se cuentan las observaciones de tipo "patología": "Sin hallazgos" y
    # "Dispositivos de soporte" no son diagnósticos y no deben exigir certeza.
    n_unc = sum(o["state"] == "incierto" and o["kind"] == "patologia" for o in outs)
    if n_unc > max_unc:
        reasons.append(f"{n_unc} patologías inciertas (máximo permitido: {max_unc})")
    nf = next(o for o in outs if o["en"] == "No Finding")
    if nf["state"] == "alto" and any(o["kind"] == "patologia" and o["state"] == "alto" for o in outs):
        reasons.append("Salidas contradictorias: «Sin hallazgos» y una patología con soporte alto")
    return reasons


def build_draft(outs):
    """Plantilla determinista: solo afirma observaciones con soporte alto."""
    path = sorted([o for o in outs if o["kind"] == "patologia" and o["state"] == "alto"], key=lambda o: -o["p"])
    dev = next(o for o in outs if o["en"] == "Support Devices")
    nf = next(o for o in outs if o["en"] == "No Finding")
    lines, trace = ["Hallazgos:"], []

    def add(sentence, o, rule):
        lines.append(f"- {sentence}")
        trace.append(dict(frase=sentence, observacion=o["es"], estado=STATE_TXT[o["state"]],
                          p_calibrada=round(o["p"], 3), regla=rule, version=TEMPLATE_VERSION))

    for o in path:
        add(f"Se identifica {o['es'].lower()}.", o, "R1: patología con soporte alto")
    if dev["state"] == "alto":
        add("Se identifican dispositivos de soporte.", dev, "R2: dispositivo con soporte alto")
    unc = any(o["kind"] == "patologia" and o["state"] == "incierto" for o in outs)
    if not path and not unc and nf["state"] == "alto":
        add("Sin hallazgos radiológicos significativos según el modelo.", nf, "R3: ausencia con soporte alto")
    if len(lines) == 1:
        lines.append("- Ninguna observación alcanza soporte suficiente; se requiere redacción manual.")
    lines += ["", "Impresión:",
              "Borrador generado por plantilla determinista. Requiere revisión, corrección y firma del especialista."]
    return "\n".join(lines), trace


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip(" -\t").lower())


def verify(text: str, outs, original: str = ""):
    by_en = {o["en"]: o for o in outs}
    orig_sentences = {_norm(s) for s in re.split(r"[.\n;]", original) if _norm(s)}
    rows = []
    for s in re.split(r"[.\n;]", text):
        ns = _norm(s)
        if not ns:
            continue
        neg_spans = [m.span() for m in NEGATION.finditer(ns)]
        cues = sorted([(a, True) for a, _ in neg_spans] +
                      [(m.start(), False) for m in AFFIRMATION.finditer(ns)
                       if not any(a <= m.start() < z for a, z in neg_spans)])
        for en, pat in PATTERNS.items():
            m = re.search(pat, ns)
            if not m:
                continue
            o = by_en[en]
            before = [neg for pos, neg in cues if pos < m.start()]
            negated = (before[-1] if before else False) or bool(POST_NEGATION.match(ns[m.end():]))
            positive = True if en == "No Finding" else not negated
            if positive:
                verdict = "Respaldada" if o["state"] == "alto" else "No respaldada"
            elif o["state"] == "alto":
                verdict = "Contradicción"
            elif o["state"] == "incierto":
                verdict = "Certeza sobre incertidumbre"
            else:
                verdict = "Respaldada"
            rows.append(dict(frase=s.strip(" -"), observacion=o["es"],
                             polaridad="afirma" if positive else "niega",
                             estado_modelo=STATE_TXT[o["state"]], veredicto=verdict,
                             origen="plantilla" if ns in orig_sentences else "revisor"))
    return rows


# --------------------------------------------------------------------------- #
# Estado de sesión y acciones
# --------------------------------------------------------------------------- #
def log(action, **extra):
    st.session_state.audit.append(dict(
        hora=dt.datetime.now().isoformat(timespec="seconds"), app=APP_VERSION,
        revisor=st.session_state.get("reviewer", "").strip() or "sin código",
        motor=st.session_state.engine_id, imagen_sha256=st.session_state.case_hash[:12],
        accion=action, **extra))


def on_edit():
    st.session_state.editing = True


def on_save(case_key):
    new = st.session_state[f"txt_{case_key}"]
    old = st.session_state.text
    diff = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), "antes", "después", lineterm=""))
    st.session_state.text = new
    st.session_state.editing = False
    if diff:
        log("Corregido", cambios=diff)


def on_decide(action, verif_rows):
    reason = st.session_state.get("reject_reason", "").strip()
    st.session_state.decision = action
    log(action, motivo=reason if action == "Rechazado" else "",
        borrador_original=st.session_state.original, texto_final=st.session_state.text,
        verificacion=[{k: r[k] for k in ("frase", "observacion", "veredicto", "origen")} for r in verif_rows])


def on_new_case():
    st.session_state.case_key = None
    st.session_state.reject_reason = ""


# --------------------------------------------------------------------------- #
# Interfaz
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Radiology Explain AI 2.0", page_icon="🩻", layout="wide")
st.markdown("""
<style>
.rx-head{background:#00688b;color:#fff;padding:14px 20px;border-radius:6px;
  display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px}
.rx-head h1{font-size:1.35rem;margin:0;color:#fff;font-weight:600}
.rx-head span{font-size:.85rem;font-weight:600;opacity:.95}
.rx-sim{background:#fff4dc;border-left:4px solid #b86e00;padding:8px 12px;font-size:.88rem;margin-bottom:10px}
.obs{display:flex;justify-content:space-between;align-items:baseline;padding:7px 10px;
  border:1px solid #d5e3ea;border-radius:4px;margin-bottom:5px;background:#fff}
.obs .n{font-size:.92rem;color:#1d2a33}.obs .p{font-variant-numeric:tabular-nums;font-weight:600;color:#00688b}
.obs .s{display:block;font-size:.74rem}
.s-alto{color:#1f7a5a}.s-incierto{color:#b86e00}.s-bajo{color:#6b7580}
.grp{font-size:.8rem;color:#52616b;margin:10px 0 4px}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("audit", [])
ss.setdefault("case_key", None)

with st.sidebar:
    st.header("Caso")
    source = st.radio("Imagen", ["Maniquí sintético A", "Maniquí sintético B", "Maniquí sintético C",
                                 "Imagen no apta (prueba de abstención)", "Subir radiografía pública"])
    upload = None
    if source == "Subir radiografía pública":
        upload = st.file_uploader("PNG o JPG desidentificado", type=["png", "jpg", "jpeg"])
        st.caption("No subas imágenes con datos de pacientes. DICOM no está soportado en la demo.")

    st.text_input("Código del revisor (p. ej., R01)", key="reviewer",
                  help="Se registra en la auditoría como autor de correcciones y decisiones. No uses nombres.")

    st.header("Motor de inferencia")
    engine = st.radio("Motor", ["Simulado (demo)", "Checkpoint DenseNet121"], label_visibility="collapsed")
    ckpt, size = "", 224
    if engine == "Checkpoint DenseNet121":
        ckpt = st.text_input("Ruta local del checkpoint (.pt / .pth)")
        size = st.selectbox("Tamaño de entrada", [224, 320], index=0)
        st.caption("Cabeza de 14 salidas en orden CheXpert estándar. Requiere torch y torchvision.")

    st.header("Calibración e incertidumbre")
    T = st.slider("Temperatura T (ajustada en validación interna)", 0.5, 4.0, 1.8, 0.1)
    t_low, t_high = st.slider("Zona de incertidumbre (p calibrada)", 0.0, 1.0, (0.2, 0.6), 0.05)
    max_unc = st.slider("Máximo de patologías inciertas antes de abstenerse", 0, 12, 4)
    hide_unc = st.checkbox("Ocultar probabilidad de observaciones inciertas", value=True)

# ---- Carga de imagen
if source.startswith("Maniquí"):
    arr = phantom({"A": 11, "B": 23, "C": 42}[source[-1]])
elif source.startswith("Imagen no apta"):
    arr = unfit_image()
elif upload is not None:
    try:
        arr = load_upload(upload)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo leer la imagen: {e}")
        st.stop()
else:
    st.info("Sube una radiografía pública desidentificada o elige un maniquí en la barra lateral.")
    st.stop()
h = img_hash(arr)

# ---- Inferencia
model = None
if engine == "Simulado (demo)":
    raw = simulated_raw_logits(h)
    engine_id = "simulado"
else:
    if not ckpt:
        st.info("Indica la ruta del checkpoint en la barra lateral.")
        st.stop()
    try:
        model, n_miss, n_unexp, ck_hash = load_densenet(ckpt)
    except Exception as e:  # noqa: BLE001
        st.error(f"No se pudo cargar el checkpoint: {e}")
        st.stop()
    raw = densenet_logits(model, h + str(size), arr, size)
    engine_id = f"densenet121:{ck_hash}"
ss.engine_id, ss.case_hash = engine_id, h

qc = quality_checks(arr)
outs = build_outputs(raw, T, t_low, t_high)
reasons = abstention_reasons(outs, qc, max_unc)
draft, trace = ("", []) if reasons else build_draft(outs)

case_key = hashlib.md5(f"{h}{engine_id}{size}{T}{t_low}{t_high}{max_unc}".encode()).hexdigest()[:10]
if ss.case_key != case_key:
    ss.case_key, ss.original, ss.text = case_key, draft, draft
    ss[f"txt_{case_key}"] = draft  # el widget siempre refleja el texto vigente del caso
    ss.editing, ss.decision = bool(reasons), None
    log("Caso cargado", abstencion=reasons)

# ---- Encabezado
st.markdown('<div class="rx-head"><h1>Radiology Explain AI 2.0</h1>'
            '<span>Apoyo clínico · Revisión obligatoria · No diagnóstico</span></div>', unsafe_allow_html=True)
if engine_id == "simulado":
    st.markdown('<div class="rx-sim">Motor simulado: las probabilidades y mapas son sintéticos y deterministas por '
                'imagen. Sirven para probar el flujo y la interfaz, no representan un modelo entrenado.</div>',
                unsafe_allow_html=True)
elif n_miss or n_unexp:
    st.warning(f"Checkpoint cargado con {n_miss} claves faltantes y {n_unexp} inesperadas. Revisa la compatibilidad.")

tab_rev, tab_log, tab_ver = st.tabs(["Revisión", "Registro de auditoría", "Pruebas del verificador"])

with tab_rev:
    c1, c2, c3 = st.columns([1.15, 0.95, 1.2], gap="medium")

    # -- Zona 1: imagen y Grad-CAM
    with c1:
        st.subheader("Radiografía")
        with st.expander("Control de calidad de entrada", expanded=bool(reasons and not all(ok for _, ok, _ in qc))):
            for name, ok, detail in qc:
                st.markdown(f"{'✅' if ok else '❌'} {name}: {detail}")
        show_cam = st.toggle("Mostrar regiones influyentes del modelo", value=False,
                             disabled=not all(ok for _, ok, _ in qc))
        if show_cam:
            opts = sorted(outs, key=lambda o: -o["p"])
            sel = st.selectbox("Observación", opts, format_func=lambda o: f"{o['es']} ({o['p']:.2f})")
            alpha = st.slider("Opacidad del mapa", 0.0, 1.0, 0.5, 0.05)
            view = st.radio("Vista", ["Superposición", "Lado a lado"], horizontal=True)
            k = [en for en, _, _ in LABELS].index(sel["en"])
            cam = simulated_cam(h, sel["en"]) if model is None else densenet_gradcam(model, h + str(size), arr, size, k)
            ov = overlay(arr, cam, alpha)
            if view == "Superposición":
                st.image(ov, width="stretch")
            else:
                a, b = st.columns(2)
                a.image(arr, caption="Original", width="stretch")
                b.image(ov, caption="Mapa", width="stretch")
            st.caption("Grad-CAM es una localización discriminativa aproximada: indica qué regiones pesaron en la "
                       "salida, no la causa ni la ubicación diagnóstica. Puede resaltar artefactos (cables, marcas).")
        else:
            st.image(arr, width="stretch")

    # -- Zona 2: observaciones
    with c2:
        st.subheader("Observaciones (14)")
        if reasons:
            st.warning("Abstención activada:\n\n" + "\n".join(f"- {r}" for r in reasons))

        def row(o):
            hidden = hide_unc and o["state"] == "incierto"
            p = "—" if hidden else f"{o['p']:.2f}"
            return (f'<div class="obs"><div><span class="n">{o["es"]}</span>'
                    f'<span class="s s-{o["state"]}">{STATE_TXT[o["state"]]}</span></div>'
                    f'<span class="p">{p}</span></div>')

        paths = sorted([o for o in outs if o["kind"] == "patologia"], key=lambda o: -o["p"])
        st.markdown('<div class="grp">Observaciones radiológicas</div>' + "".join(row(o) for o in paths),
                    unsafe_allow_html=True)
        others = [o for o in outs if o["kind"] != "patologia"]
        st.markdown('<div class="grp">No son patología</div>' + "".join(row(o) for o in others),
                    unsafe_allow_html=True)
        with st.expander("Efecto de la calibración"):
            st.dataframe(pd.DataFrame([dict(Observación=o["es"], **{"p sin calibrar": round(o["p_raw"], 3),
                                            "p calibrada": round(o["p"], 3)}) for o in outs]),
                         hide_index=True, width="stretch")
            st.caption(f"p = σ(z / T), con T = {T}. Umbrales: bajo < {t_low} ≤ incierto < {t_high} ≤ alto.")

    # -- Zona 3: preinforme y decisión humana
    with c3:
        st.subheader("Borrador de preinforme")
        locked = ss.decision is not None
        if reasons:
            st.error("El preinforme automático está bloqueado. Puedes redactarlo manualmente o rechazar el caso.")
        st.text_area("Texto", key=f"txt_{case_key}", height=230,
                     disabled=locked or not ss.editing, label_visibility="collapsed")

        verif = verify(ss.text, outs, ss.original)
        machine_fail = [r for r in verif if r["origen"] == "plantilla" and r["veredicto"] != "Respaldada"]
        human_flags = [r for r in verif if r["origen"] == "revisor" and r["veredicto"] != "Respaldada"]
        n_ok = sum(r["veredicto"] == "Respaldada" for r in verif)
        st.caption(f"Trazabilidad: {n_ok} afirmaciones respaldadas · {len(machine_fail) + len(human_flags)} "
                   f"sin respaldo del modelo.")
        if machine_fail:
            st.error("El verificador detectó afirmaciones de la plantilla sin respaldo. Aprobación bloqueada.")
        if human_flags:
            st.warning("Afirmaciones añadidas por el revisor sin respaldo del modelo:\n\n"
                       + "\n".join(f"- «{r['frase']}»: {r['veredicto'].lower()}" for r in human_flags))
        if verif:
            with st.expander("Verificación y origen de cada afirmación"):
                st.dataframe(pd.DataFrame(verif), hide_index=True, width="stretch")
        if trace:
            with st.expander("Traza de generación"):
                st.dataframe(pd.DataFrame(trace), hide_index=True, width="stretch")

        if locked:
            (st.success if ss.decision == "Aprobado" else st.info)(
                f"{ss.decision}. La decisión quedó registrada en la auditoría.")
            st.button("Nuevo caso", on_click=on_new_case)
        else:
            if ss.editing:
                st.button("Guardar corrección", on_click=on_save, args=(case_key,), type="primary")
            else:
                st.button("Corregir", on_click=on_edit)
            understood = st.checkbox("Entiendo que es un apoyo y no un diagnóstico; la decisión final es mía.")
            attest = True
            if human_flags:
                attest = st.checkbox("Asumo como juicio clínico propio las afirmaciones sin respaldo del modelo.")
            a, r = st.columns(2)
            can_approve = understood and attest and not machine_fail and not ss.editing and ss.text.strip() != ""
            a.button("Aprobar", on_click=on_decide, args=("Aprobado", verif), disabled=not can_approve,
                     type="primary", width="stretch")
            st.text_input("Motivo del rechazo", key="reject_reason")
            r.button("Rechazar", on_click=on_decide, args=("Rechazado", verif),
                     disabled=ss.editing or not ss.get("reject_reason", "").strip(), width="stretch")
            if ss.editing:
                st.caption("Guarda la corrección antes de decidir: el registro conserva solo texto guardado.")
            st.caption("La aprobación registra hora, versión, texto original, texto final y cambios.")

with tab_log:
    st.subheader("Registro de auditoría")
    st.caption("Sin nombres ni identificadores clínicos: solo un hash parcial de la imagen.")
    if ss.audit:
        st.dataframe(pd.DataFrame([{k: v for k, v in e.items() if k in ("hora", "accion", "revisor", "motor", "imagen_sha256")}
                                   for e in ss.audit]), hide_index=True, width="stretch")
        st.download_button("Descargar registro (JSON)", json.dumps(ss.audit, ensure_ascii=False, indent=2),
                           "auditoria_radiology_explain.json", "application/json")
        with st.expander("Detalle completo"):
            st.json(ss.audit)

with tab_ver:
    st.subheader("Pruebas del verificador factual")
    st.caption("Fixtures controlados (US-14): cada caso fija estados del modelo y un texto, y compara el veredicto "
               "esperado con el obtenido.")

    def fake_outs(states):
        return [dict(en=en, es=es, kind=kd, p=0.5, p_raw=0.5, state=states.get(en, "bajo")) for en, es, kd in LABELS]

    fixtures = [
        ({"Pleural Effusion": "alto"}, "Se identifica derrame pleural.", "Respaldada"),
        ({"Pneumonia": "incierto"}, "Se identifica neumonía.", "No respaldada"),
        ({"Pleural Effusion": "alto"}, "No se observa derrame pleural.", "Contradicción"),
        ({"Pneumothorax": "incierto"}, "No se observa neumotórax.", "Certeza sobre incertidumbre"),
        ({}, "No se observa cardiomegalia.", "Respaldada"),
        ({"No Finding": "incierto"}, "Sin hallazgos radiológicos.", "No respaldada"),
        ({"Support Devices": "alto"}, "Catéter venoso central en posición.", "Respaldada"),
        ({"Pleural Effusion": "alto"}, "Sin derrame pleural.", "Contradicción"),
        ({"Pneumothorax": "incierto"}, "Sin neumotórax.", "Certeza sobre incertidumbre"),
        ({"Pleural Effusion": "alto"}, "No se observa neumotórax, se identifica derrame pleural.",
         "Respaldada | Respaldada"),
        ({"Cardiomegaly": "alto"}, "Cardiomegalia sin derrame pleural.", "Respaldada | Respaldada"),
        ({"Pneumonia": "alto"}, "Neumonía: descartada.", "Contradicción"),
    ]
    res = []
    for states, text, expected in fixtures:
        got = verify(text, fake_outs(states))
        v = " | ".join(r["veredicto"] for r in got) if got else "(sin afirmación)"
        res.append(dict(texto=text, esperado=expected, obtenido=v, resultado="✅" if v == expected else "❌"))
    st.dataframe(pd.DataFrame(res), hide_index=True, width="stretch")

    rng = np.random.default_rng(0)
    n_cases, n_bad = 300, 0
    for _ in range(n_cases):
        o = build_outputs(rng.normal(-1, 3, 14), 1.0, 0.2, 0.6)
        if abstention_reasons(o, [("", True, "")], 14):
            continue
        d, _ = build_draft(o)
        n_bad += sum(r["veredicto"] != "Respaldada" for r in verify(d, o, d))
    st.metric(f"Afirmaciones no respaldadas en borradores de {n_cases} casos aleatorios", n_bad)
