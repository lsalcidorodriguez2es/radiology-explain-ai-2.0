"""Pruebas de regresión extremo a extremo con streamlit.testing (se ejecutan en CI)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run(**radio):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_arranque_sin_excepciones():
    at = run()
    assert not at.exception
    assert at.text_area[0].value.startswith("Hallazgos:")


def test_fixtures_del_verificador():
    at = run()
    fx = at.dataframe[-1].value
    assert (fx["resultado"] == "✅").all(), fx[fx["resultado"] != "✅"]


def test_plantilla_sin_afirmaciones_no_respaldadas():
    at = run()
    assert at.metric[0].value == "0"


@pytest.mark.parametrize("fuente", ["Maniquí sintético B", "Maniquí sintético C"])
def test_maniquies(fuente):
    at = run()
    at.sidebar.radio[0].set_value(fuente).run()
    assert not at.exception


def test_abstencion_por_imagen_no_apta():
    at = run()
    at.sidebar.radio[0].set_value("Imagen no apta (prueba de abstención)").run()
    assert at.text_area[0].value == ""
    assert any("bloqueado" in e.value for e in at.error)


def test_flujo_corregir_aprobar_y_nuevo_caso():
    at = run()
    at.sidebar.text_input[0].set_value("R01").run()
    [b for b in at.button if b.label == "Corregir"][0].click().run()
    at.text_area[0].set_value(at.text_area[0].value + "\n- Sin neumotórax.").run()
    [b for b in at.button if b.label == "Guardar corrección"][0].click().run()
    for c in at.checkbox:
        if c.label.startswith(("Entiendo", "Asumo")):
            c.check()
    at.run()
    [b for b in at.button if b.label == "Aprobar"][0].click().run()
    assert at.success
    acciones = [e["accion"] for e in at.session_state["audit"]]
    assert acciones[-3:] == ["Caso cargado", "Corregido", "Aprobado"]
    assert at.session_state["audit"][-1]["revisor"] == "R01"
    [b for b in at.button if b.label == "Nuevo caso"][0].click().run()
    assert at.session_state["reject_reason"] == ""
