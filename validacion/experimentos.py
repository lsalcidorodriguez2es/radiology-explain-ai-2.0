"""
Experimentos de validación funcional de Radiology Explain AI 2.0 (capítulo 6).

Compara las versiones v0.1.0, v0.1.1 y v0.1.2 de la aplicación sobre el mismo material
de prueba. Todos los datos son sintéticos y deterministas: no se utilizan imágenes ni
informes de pacientes. Los resultados se escriben en validacion/resultados.json.

Uso:  python validacion/experimentos.py
"""
import itertools
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"v0.1.0": ROOT / "validacion/referencia/app_v0_1_0.py",
            "v0.1.1": ROOT / "validacion/referencia/app_v0_1_1.py",
            "v0.1.2": ROOT / "app.py"}
CURRENT = "v0.1.2"
MARK = "# Estado de sesión y acciones"
SEED = 20260929


def load_core(path):
    """Carga las funciones puras de la aplicación (todo lo anterior a la interfaz)."""
    import logging
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    src = Path(path).read_text(encoding="utf-8").split(MARK)[0]
    ns = {"__name__": "core"}
    exec(compile(src, str(path), "exec"), ns)
    return ns


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def upper_zero(n, alpha=0.05):
    """Cota superior unilateral exacta (Clopper-Pearson) cuando se observan 0 fallos."""
    return 1 - alpha ** (1 / n)


# --------------------------------------------------------------------------- #
# E1. Corpus controlado para el verificador factual
# --------------------------------------------------------------------------- #
FEM = {"Cardiomegaly", "Lung Opacity", "Lung Lesion", "Consolidation", "Pneumonia",
       "Atelectasis", "Pleural Other", "Fracture"}


def expected(positive, state):
    if positive:
        return "Respaldada" if state == "alto" else "No respaldada"
    return {"alto": "Contradicción", "incierto": "Certeza sobre incertidumbre", "bajo": "Respaldada"}[state]


def build_corpus(labels):
    obs = [(en, es.lower()) for en, es, kind in labels if en != "No Finding"]
    single_pos = ["Se identifica {t}.", "Se observa {t}.", "Hallazgo compatible con {t}."]
    single_neg = ["No se observa {t}.", "Sin {t}.", "No hay {t}.", "Se descarta {t}.",
                  "Ausencia de {t}.", "{T}: {desc}."]
    pair = [("No se observa {a}, se identifica {b}.", False, True),
            ("Se identifica {a} sin {b}.", True, False),
            ("No se observan {a} ni {b}.", False, False),
            ("Se identifican {a} y {b}.", True, True)]
    items = []  # (texto, estados, [(en, positivo)], familia)
    for (en, t), st in itertools.product(obs, ["alto", "incierto", "bajo"]):
        desc = "descartada" if en in FEM else "descartado"
        for tpl in single_pos:
            items.append((tpl.format(t=t), {en: st}, [(en, True)], "simple afirmativa"))
        for tpl in single_neg:
            items.append((tpl.format(t=t, T=t.capitalize(), desc=desc), {en: st}, [(en, False)], "simple negativa"))
    for (a, ta), (b, tb) in itertools.permutations(obs, 2):
        for sa, sb in itertools.product(["alto", "incierto", "bajo"], repeat=2):
            for tpl, pa, pb in pair:
                items.append((tpl.format(a=ta, b=tb), {a: sa, b: sb}, [(a, pa), (b, pb)], "compuesta"))
    for st in ["alto", "incierto", "bajo"]:
        items.append(("Sin hallazgos radiológicos significativos.", {"No Finding": st},
                      [("No Finding", True)], "sin hallazgos"))
    return items


def build_holdout(labels):
    """Corpus reservado: redacciones que no se usaron al corregir el verificador."""
    obs = [(en, es.lower()) for en, es, kind in labels if en != "No Finding"]
    pos = ["Se aprecia {t}.", "Imagen sugestiva de {t}.", "Persiste {t}."]
    neg = ["No se evidencia {t}.", "Negativo para {t}.", "Sin evidencia de {t}.", "{T} ausente.",
           "No se identifican signos de {t}."]
    pair = [("Se aprecia {a}; no se observa {b}.", True, False),
            ("{A} sin cambios, sin {b}.", True, False),
            ("No se identifica {a}; persiste {b}.", False, True),
            ("Sin {a} ni {b}.", False, False),
            ("Se observa {a} con {b}.", True, True)]
    items = []
    for (en, t), st in itertools.product(obs, ["alto", "incierto", "bajo"]):
        for tpl in pos:
            items.append((tpl.format(t=t), {en: st}, [(en, True)], "simple afirmativa"))
        for tpl in neg:
            items.append((tpl.format(t=t, T=t.capitalize()), {en: st}, [(en, False)], "simple negativa"))
    for (a, ta), (b, tb) in itertools.permutations(obs, 2):
        for sa, sb in itertools.product(["alto", "incierto", "bajo"], repeat=2):
            for tpl, pa, pb in pair:
                items.append((tpl.format(a=ta, A=ta.capitalize(), b=tb), {a: sa, b: sb}, [(a, pa), (b, pb)],
                              "compuesta"))
    return items


def run_e1(cores, corpus_fn=None):
    labels = cores[CURRENT]["LABELS"]
    corpus = (corpus_fn or build_corpus)(labels)
    es_of = {en: es for en, es, _ in labels}
    rows = {v: [] for v in cores}
    for v, ns in cores.items():
        for text, states, claims, fam in corpus:
            outs = [dict(en=en, es=es, kind=k, p=.5, p_raw=.5, state=states.get(en, "bajo")) for en, es, k in labels]
            got = {r["observacion"]: r["veredicto"] for r in ns["verify"](text, outs)}
            for en, pos in claims:
                exp = expected(pos, states[en])
                rows[v].append(dict(fam=fam, exp=exp, got=got.get(es_of[en], "(no detectada)")))
    res = {"n_frases": len(corpus), "n_afirmaciones": len(rows[CURRENT])}
    for v, r in rows.items():
        unsafe = [x for x in r if x["exp"] != "Respaldada"]
        safe = [x for x in r if x["exp"] == "Respaldada"]
        missed = sum(x["got"] == "Respaldada" or x["got"] == "(no detectada)" for x in unsafe)
        false_alarm = sum(x["got"] != "Respaldada" for x in safe)
        exact = sum(x["exp"] == x["got"] for x in r)
        fams = {}
        for fam in sorted({x["fam"] for x in r}):
            sub = [x for x in r if x["fam"] == fam]
            fams[fam] = dict(n=len(sub), exactas=sum(x["exp"] == x["got"] for x in sub),
                             inseguras=sum(x["exp"] != "Respaldada" for x in sub),
                             omitidas=sum(x["exp"] != "Respaldada" and x["got"] in ("Respaldada", "(no detectada)")
                                          for x in sub))
        res[v] = dict(n_inseguras=len(unsafe), omitidas=missed,
                      sensibilidad=wilson(len(unsafe) - missed, len(unsafe)),
                      n_seguras=len(safe), falsas_alarmas=false_alarm,
                      especificidad=wilson(len(safe) - false_alarm, len(safe)),
                      exactitud=wilson(exact, len(r)), por_familia=fams)
    # McNemar exacto sobre afirmaciones inseguras (detectada sí/no), emparejado por afirmación
    a, b = rows["v0.1.0"], rows[CURRENT]
    det = lambda x: x["got"] not in ("Respaldada", "(no detectada)")
    b01 = sum(1 for x, y in zip(a, b) if x["exp"] != "Respaldada" and not det(x) and det(y))
    b10 = sum(1 for x, y in zip(a, b) if x["exp"] != "Respaldada" and det(x) and not det(y))
    p = stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    res["mcnemar_v010_vs_actual"] = dict(solo_actual_detecta=b01, solo_v010_detecta=b10, p_exacto=p)
    for v in cores:
        if res[v]["omitidas"] == 0:
            res[v]["cota_sup_omision_95"] = upper_zero(res[v]["n_inseguras"])
    return res


# --------------------------------------------------------------------------- #
# E2. Seguridad de la plantilla determinista
# --------------------------------------------------------------------------- #
def run_e2(cores, n=10_000):
    res = {}
    for v, ns in cores.items():
        rng = np.random.default_rng(SEED)
        drafts = claims = bad = abst = r3_unc = 0
        for _ in range(n):
            T = rng.uniform(0.5, 4.0)
            lo = rng.uniform(0.05, 0.45)
            hi = rng.uniform(lo + 0.1, 0.95)
            outs = ns["build_outputs"](rng.normal(-1, 3, 14), T, lo, hi)
            if ns["abstention_reasons"](outs, [("", True, "")], 14):
                abst += 1
                continue
            d, trace = ns["build_draft"](outs)
            drafts += 1
            ver = ns["verify"](d, outs, d)
            claims += len(ver)
            bad += sum(r["veredicto"] != "Respaldada" for r in ver)
            unc = any(o["kind"] == "patologia" and o["state"] == "incierto" for o in outs)
            if unc and "sin hallazgos" in d.lower():
                r3_unc += 1
        res[v] = dict(casos=n, abstenciones=abst, borradores=drafts, afirmaciones=claims,
                      no_respaldadas=bad, cota_sup_95=upper_zero(claims) if bad == 0 else None,
                      sin_hallazgos_con_incertidumbre=r3_unc,
                      tasa_sin_hallazgos_con_incertidumbre=wilson(r3_unc, drafts))
    return res


# --------------------------------------------------------------------------- #
# E3. Puerta de calidad bajo perturbaciones y conversión de 16 bits
# --------------------------------------------------------------------------- #
def run_e3(cores, n=200):
    from PIL import Image, ImageFilter
    ns = cores[CURRENT]
    base = [ns["phantom"](s) for s in range(1000, 1000 + n)]
    rng = np.random.default_rng(SEED)

    def rate(imgs):
        return sum(not all(ok for _, ok, _ in ns["quality_checks"](a)) for a in imgs) / len(imgs)

    pert = {}
    pert["referencia"] = rate(base)
    for s in (10, 25, 50, 80):
        pert[f"ruido gaussiano sigma={s}"] = rate([np.clip(a + rng.normal(0, s, a.shape), 0, 255).astype(np.uint8)
                                                   for a in base])
    for f in (0.5, 0.35, 0.25, 0.15):
        pert[f"contraste x{f}"] = rate([(128 + (a.astype(float) - a.mean()) * f).clip(0, 255).astype(np.uint8)
                                        for a in base])
    for sz in (320, 256, 224, 128):
        pert[f"reescalado a {sz} px"] = rate([np.array(Image.fromarray(a).resize((sz, sz))) for a in base])
    for r in (2, 6, 12):
        pert[f"desenfoque radio {r}"] = rate([np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(r)))
                                              for a in base])
    pert["recorte a proporción 2:1"] = rate([a[:, :] if False else a[128:384, :] for a in base])

    # Conversión de PNG de 16 bits (rango completo y 12 bits útiles en contenedor de 16)
    import tempfile
    conv = {}
    tmp = Path(tempfile.mkdtemp())
    for name, scale in (("16 bits rango completo", 257), ("12 bits en contenedor de 16", 16)):
        paths = []
        for i, a in enumerate(base[:50]):
            p = tmp / f"{scale}_{i}.png"
            Image.fromarray((a.astype(np.uint32) * scale).astype(np.uint16)).save(p)
            paths.append(p)
        conv[name] = {"n": len(paths), "v0.1.0": rate([np.array(Image.open(p).convert("L")) for p in paths])}
        for v in ("v0.1.1", CURRENT):
            conv[name][v] = rate([cores[v]["load_upload"](p) for p in paths])
    return dict(n_imagenes=n, abstencion_por_perturbacion=pert, conversion_16_bits=conv)


# --------------------------------------------------------------------------- #
# E4. Reproducibilidad entre procesos
# --------------------------------------------------------------------------- #
CHILD = r"""
import sys, hashlib, warnings, logging; warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)
sys.path.insert(0, sys.argv[2])
from experimentos import load_core
ns = load_core(sys.argv[1]); out = []
for s in (11, 23, 42):
    a = ns["phantom"](s); h = ns["img_hash"](a)
    out.append(hashlib.sha256(ns["simulated_raw_logits"](h).tobytes()).hexdigest()[:10])
    for en, _, _ in ns["LABELS"]:
        out.append(hashlib.sha256(ns["simulated_cam"](h, en).tobytes()).hexdigest()[:10])
print(",".join(out))
"""


def run_e4(runs=5):
    res = {}
    for v, path in VERSIONS.items():
        outs = []
        for seed in range(runs):
            env = dict(os.environ, PYTHONHASHSEED=str(seed + 1))
            r = subprocess.run([sys.executable, "-c", CHILD, str(path), str(Path(__file__).parent)],
                               capture_output=True, text=True, env=env)
            outs.append(r.stdout.strip().split(","))
        cols = list(zip(*outs))
        logits = [c for i, c in enumerate(cols) if i % 15 == 0]
        cams = [c for i, c in enumerate(cols) if i % 15 != 0]
        res[v] = dict(procesos=runs, logits_estables=sum(len(set(c)) == 1 for c in logits), logits_total=len(logits),
                      mapas_estables=sum(len(set(c)) == 1 for c in cams), mapas_total=len(cams))
    return res


# --------------------------------------------------------------------------- #
# E5. Costo computacional en CPU
# --------------------------------------------------------------------------- #
def run_e5(cores, reps=15):
    import torch
    import torchvision
    torch.manual_seed(SEED)
    ns = cores[CURRENT]
    model = torchvision.models.densenet121(weights=None)
    model.classifier = torch.nn.Linear(1024, 14)
    model.eval()
    arr = ns["phantom"](11)
    res = {"cpu": platform.processor() or platform.machine(), "hilos_torch": torch.get_num_threads(),
           "torch": torch.__version__, "repeticiones": reps}
    gradcam = ns["densenet_gradcam"].__wrapped__ if hasattr(ns["densenet_gradcam"], "__wrapped__") else None
    for size in (224, 320):
        x = ns["_to_tensor"](arr, size)
        with torch.no_grad():
            model(x)
        t = []
        for _ in range(reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                model(x)
            t.append(time.perf_counter() - t0)
        g = []
        for _ in range(reps):
            t0 = time.perf_counter()
            gradcam(model, "k", arr, size, 5)
            g.append(time.perf_counter() - t0)
        res[f"inferencia_{size}_ms"] = [float(np.median(t) * 1e3), float(np.percentile(t, 90) * 1e3)]
        res[f"gradcam_{size}_ms"] = [float(np.median(g) * 1e3), float(np.percentile(g, 90) * 1e3)]
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        h = ns["img_hash"](arr)
        ns["build_outputs"](ns["simulated_raw_logits"](h), 1.8, .2, .6)
        t.append(time.perf_counter() - t0)
    res["simulado_ms"] = [float(np.median(t) * 1e3), float(np.percentile(t, 90) * 1e3)]
    res["parametros"] = int(sum(p.numel() for p in model.parameters()))
    return res


if __name__ == "__main__":
    cores = {v: load_core(p) for v, p in VERSIONS.items()}
    out = {"semilla": SEED, "python": platform.python_version(), "numpy": np.__version__}
    for name, fn in (("E1", lambda: run_e1(cores)), ("E1_reservado", lambda: run_e1(cores, build_holdout)), ("E2", lambda: run_e2(cores)), ("E3", lambda: run_e3(cores)),
                     ("E4", run_e4), ("E5", lambda: run_e5(cores))):
        t0 = time.time()
        out[name] = fn()
        print(f"{name} completado en {time.time() - t0:.1f} s", flush=True)
    dest = Path(__file__).parent / "resultados.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
