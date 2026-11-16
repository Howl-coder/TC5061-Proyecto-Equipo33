import pytest

from src import tracking


def test_validar_entorno_candidato_requiere_un_commit(monkeypatch):
    monkeypatch.setattr(tracking, "obtener_git_commit", lambda: "desconocido")

    with pytest.raises(RuntimeError, match="repositorio Git identificable"):
        tracking.validar_entorno_candidato()


def test_validar_entorno_candidato_rechaza_cambios_locales(monkeypatch):
    monkeypatch.setattr(tracking, "obtener_git_commit", lambda: "abc123")
    monkeypatch.setattr(tracking, "repo_con_cambios", lambda: True)

    with pytest.raises(RuntimeError, match="cambios locales"):
        tracking.validar_entorno_candidato()


def test_validar_entorno_candidato_acepta_arbol_limpio(monkeypatch):
    monkeypatch.setattr(tracking, "obtener_git_commit", lambda: "abc123")
    monkeypatch.setattr(tracking, "repo_con_cambios", lambda: False)

    tracking.validar_entorno_candidato()
