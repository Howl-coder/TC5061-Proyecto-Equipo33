"""Crea el informe DOCX de drift, comparación A/B y Continuous Training de P1."""

import argparse
import json
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
from config import DATA_DIR, REPORTS_DIR
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

DIRECTORIO_REPORTE = REPORTS_DIR / "monitoring"
RUTA_DRIFT = DIRECTORIO_REPORTE / "drift_p1.json"
RUTA_AB = DIRECTORIO_REPORTE / "ab_test_p1.json"
RUTA_METADATA = DATA_DIR / "monitoring" / "metadata.json"
RUTA_SALIDA_DEFAULT = DIRECTORIO_REPORTE / "TC5061_Drift_ABTest_Equipo33.docx"
UMBRAL_DRIFT_GLOBAL = 0.5


def _leer_json(ruta: Path) -> dict:
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Genera primero el reporte de monitoreo.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def _metricas_drift(datos: dict) -> tuple[list[dict], dict]:
    metricas = datos.get("metrics", [])
    variables = []
    resumen = None
    for metrica in metricas:
        nombre = metrica.get("metric_name", "")
        if nombre.startswith("ValueDrift("):
            variables.append(
                {
                    "variable": metrica["config"]["column"],
                    "score": float(metrica["value"]),
                    "umbral": float(metrica["config"]["threshold"]),
                    "metodo": metrica["config"]["method"],
                }
            )
        elif nombre.startswith("DriftedColumnsCount"):
            resumen = metrica["value"]
    if not variables or resumen is None:
        raise ValueError(f"{RUTA_DRIFT} no contiene métricas de drift por variable.")
    return variables, resumen


def _valor_metrica(datos: dict, nombre: str):
    for metrica in datos.get("metrics", []):
        if metrica.get("metric_name") == nombre:
            return metrica.get("value")
    return None


def _tabla(documento: Document, encabezados: list[str], filas: list[list[str]]) -> None:
    tabla = documento.add_table(rows=1, cols=len(encabezados))
    tabla.style = "Light Shading Accent 1"
    tabla.alignment = WD_TABLE_ALIGNMENT.CENTER
    for celda, texto in zip(tabla.rows[0].cells, encabezados):
        celda.text = texto
        celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        for run in celda.paragraphs[0].runs:
            run.bold = True
    for fila in filas:
        celdas = tabla.add_row().cells
        for celda, texto in zip(celdas, fila):
            celda.text = texto
            celda.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def _enlace(parrafo, texto: str, url: str) -> None:
    relacion = parrafo.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hipervinculo = OxmlElement("w:hyperlink")
    hipervinculo.set(qn("r:id"), relacion)
    run = OxmlElement("w:r")
    propiedades = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    propiedades.append(color)
    subrayado = OxmlElement("w:u")
    subrayado.set(qn("w:val"), "single")
    propiedades.append(subrayado)
    run.append(propiedades)
    texto_xml = OxmlElement("w:t")
    texto_xml.text = texto
    run.append(texto_xml)
    hipervinculo.append(run)
    parrafo._p.append(hipervinculo)


def _crear_grafico_drift(variables: list[dict], ruta: Path) -> None:
    ordenadas = sorted(variables, key=lambda item: item["score"])
    colores = [
        "#bd3b32" if item["score"] > item["umbral"] else "#377eb8"
        for item in ordenadas
    ]
    alto = max(4.0, len(ordenadas) * 0.32)
    fig, ax = plt.subplots(figsize=(9, alto))
    posiciones = range(len(ordenadas))
    scores = [item["score"] for item in ordenadas]
    ax.barh(posiciones, scores, color=colores)
    ax.set_yticks(list(posiciones), [item["variable"] for item in ordenadas])
    ax.set_xlabel("Score de drift (umbral por variable)")
    ax.set_title("Evidently: drift por variable, septiembre frente a abril–agosto")
    for y, item in enumerate(ordenadas):
        ax.plot([item["umbral"], item["umbral"]], [y - 0.4, y + 0.4], color="#222222", lw=1.5)
        ax.text(item["score"] + 0.004, y, f"{item['score']:.3f}", va="center", fontsize=8)
    limite = max(item["score"] for item in ordenadas)
    ax.set_xlim(0, limite * 1.18 if limite else 0.1)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(
        handles=[
            Patch(color="#bd3b32", label="Supera umbral"),
            Patch(color="#377eb8", label="No supera umbral"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _crear_grafico_ab(datos_ab: dict, ruta: Path) -> None:
    version_a, version_b = datos_ab["version_a"], datos_ab["version_b"]
    metricas = datos_ab["metricas"]
    nombres = [version_a, version_b]
    colores = ["#377eb8", "#f28e2b"]
    fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.4))
    for eje, clave, titulo, formato in (
        (ejes[0], "rmse", "RMSE ↓", ".4f"),
        (ejes[1], "mae", "MAE ↓", ".4f"),
        (ejes[2], "r2", "R² ↑", ".4f"),
    ):
        valores = [metricas[v][clave] for v in nombres]
        barras = eje.bar(["V" + version_a, "V" + version_b], valores, color=colores)
        eje.set_title(titulo)
        eje.grid(axis="y", alpha=0.2)
        eje.bar_label(barras, fmt="{:" + formato + "}", padding=2, fontsize=8)
    fig.suptitle("Evaluación sobre las mismas filas del periodo actual")
    fig.tight_layout()
    fig.savefig(ruta, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _estilo(documento: Document) -> None:
    section = documento.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    normal = documento.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    for estilo in ("Heading 1", "Heading 2"):
        documento.styles[estilo].font.name = "Aptos Display"
        documento.styles[estilo].font.color.rgb = RGBColor(31, 78, 121)


def generar_documento(ruta_salida: Path = RUTA_SALIDA_DEFAULT) -> Path:
    drift = _leer_json(RUTA_DRIFT)
    ab = _leer_json(RUTA_AB)
    metadata = _leer_json(RUTA_METADATA)
    variables, resumen_drift = _metricas_drift(drift)
    actual = metadata["actual"]
    referencia = metadata["referencia"]
    version_a, version_b = ab["version_a"], ab["version_b"]
    metricas = ab["metricas"]
    modelos = ab["modelos"]
    bootstrap = ab["bootstrap_rmse_b_menos_a"]
    drift_actual = {
        nombre: _valor_metrica(drift, nombre)
        for nombre in (
            "DatasetMissingValueCount()",
            "DuplicatedRowCount()",
            "EmptyRowsCount()",
            "EmptyColumnsCount()",
        )
    }

    if ruta_salida.exists():
        raise FileExistsError(
            f"No se sobrescribió el reporte existente: {ruta_salida}. Usa --output para otro nombre."
        )
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    documento = Document()
    _estilo(documento)
    titulo = documento.add_heading("Monitoreo del modelo MiBici P1", 0)
    titulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitulo = documento.add_paragraph(
        "Drift, evaluación A/B y recomendación de Continuous Training"
    )
    subtitulo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fecha = documento.add_paragraph("Corte del análisis: 14 de noviembre de 2026")
    fecha.alignment = WD_ALIGN_PARAGRAPH.CENTER

    documento.add_heading("Resumen ejecutivo", level=1)
    documento.add_paragraph(
        f"Evidently detectó drift en {int(resumen_drift['count'])} de "
        f"{len(variables)} variables ({resumen_drift['share']:.1%}), superando el criterio "
        f"global de {UMBRAL_DRIFT_GLOBAL:.0%}. La versión {version_b} no mejora a la "
        f"versión {version_a} en las métricas puntuales; por ello se recomienda conservar "
        "la versión 1 como champion y usar el drift como disparador de entrenamiento candidato, "
        "no de promoción automática."
    )

    documento.add_heading("1. Datos y método", level=1)
    documento.add_paragraph(
        f"Referencia: abril–agosto de 2026, periodo de entrenamiento "
        f"({referencia['filas']:,} filas efectivas, {referencia['estaciones']} estaciones; "
        f"{referencia['desde']} a {referencia['hasta']}). Actual: septiembre de 2026 "
        f"({actual['filas']:,} filas, {actual['estaciones']} estaciones; "
        f"{actual['desde']} a {actual['hasta']})."
    )
    documento.add_paragraph(
        "El drift se calculó con Evidently 0.7.23 sobre las 11 features de P1 y el target "
        "demanda_salidas. Para el A/B se usaron las versiones registradas del mismo modelo, "
        "las mismas filas de septiembre y las mismas métricas (RMSE, MAE y R²), con "
        "predicciones negativas acotadas a cero."
    )
    documento.add_paragraph(
        "La evaluación es offline, no un reparto aleatorio de tráfico real. Septiembre es el "
        "único periodo posterior disponible y también fue el conjunto de prueba original."
    )

    documento.add_heading("2. Drift y calidad de datos", level=1)
    documento.add_paragraph(
        "Las barras muestran los scores por variable; cada marca vertical representa el umbral "
        "de Evidently para esa variable. El umbral es un criterio de distancia, no un p-valor."
    )
    with tempfile.TemporaryDirectory() as temporal:
        ruta_drift_img = Path(temporal) / "drift.png"
        ruta_ab_img = Path(temporal) / "ab.png"
        _crear_grafico_drift(variables, ruta_drift_img)
        _crear_grafico_ab(ab, ruta_ab_img)
        documento.add_picture(str(ruta_drift_img), width=Inches(6.6))
        documento.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        documento.add_paragraph(
            "Superan el umbral: "
            + ", ".join(
                f"{item['variable']} ({item['score']:.3f})"
                for item in variables
                if item["score"] > item["umbral"]
            )
            + "."
        )
        documento.add_paragraph(
            "El mayor score es rolling_mean_6h (0.226). Las variables de calendario no "
            "superan el umbral; el cambio se concentra en demanda, rezagos y media móvil."
        )
        _tabla(
            documento,
            ["Control de calidad en actual", "Resultado"],
            [
                ["Valores faltantes", str(drift_actual["DatasetMissingValueCount()"]["count"])],
                ["Filas duplicadas", str(int(drift_actual["DuplicatedRowCount()"]))],
                ["Filas vacías", str(int(drift_actual["EmptyRowsCount()"]))],
                ["Columnas vacías", str(int(drift_actual["EmptyColumnsCount()"]))],
            ],
        )

        documento.add_heading("3. Resultado del A/B testing", level=1)
        documento.add_paragraph(
            f"V{version_a} (baseline) se compara contra V{version_b} (candidata). "
            f"El bootstrap pareado por día utilizó {bootstrap['bloques_dia']} bloques diarios "
            f"y {bootstrap['replicas']:,} réplicas."
        )
        documento.add_picture(str(ruta_ab_img), width=Inches(6.6))
        documento.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        _tabla(
            documento,
            ["Métrica", f"V{version_a}", f"V{version_b}", "Preferible"],
            [
                [
                    "RMSE ↓",
                    f"{metricas[version_a]['rmse']:.4f}",
                    f"{metricas[version_b]['rmse']:.4f}",
                    f"V{version_a}",
                ],
                [
                    "MAE ↓",
                    f"{metricas[version_a]['mae']:.4f}",
                    f"{metricas[version_b]['mae']:.4f}",
                    f"V{version_a}",
                ],
                [
                    "R² ↑",
                    f"{metricas[version_a]['r2']:.4f}",
                    f"{metricas[version_b]['r2']:.4f}",
                    f"V{version_a}",
                ],
            ],
        )
        documento.add_paragraph(
            f"Delta RMSE V{version_b} − V{version_a}: "
            f"{bootstrap['delta_observado_rmse_b_menos_a']:+.4f}; IC bootstrap 95% "
            f"[{bootstrap['ic_95_inferior']:+.4f}, {bootstrap['ic_95_superior']:+.4f}]. "
            "El intervalo incluye cero, así que la diferencia no es concluyente con este "
            "periodo. La candidata queda ligeramente peor en las métricas puntuales."
        )
        documento.add_paragraph(
            f"V{version_b}: hiperparámetros {modelos[version_b]['hiperparametros']}; "
            f"run {modelos[version_b]['run_id']}; git_dirty="
            f"{modelos[version_b]['git_dirty']}."
        )

    documento.add_heading("4. Recomendación de Continuous Training", level=1)
    documento.add_paragraph(
        "Sí conviene habilitar Continuous Training como flujo de generación de candidatos, "
        "porque hay drift en más de la mitad de las variables analizadas. No conviene activar "
        "promoción automática: este A/B no muestra mejora de la V2 y la incertidumbre abarca "
        "cero. Mantener V1 como champion."
    )
    for texto in (
        "Usar drift persistente como señal para entrenar una candidata y registrar su linaje.",
        (
            "Comparar candidata y champion con las mismas filas etiquetadas y un periodo posterior "
            "independiente; definir umbrales de mejora antes de la evaluación."
        ),
        (
            "Promover solo si la mejora es consistente, relevante para el negocio y no deteriora "
            "las métricas acordadas; mantener revisión humana en esta etapa."
        ),
    ):
        documento.add_paragraph(texto, style="List Bullet")
    documento.add_paragraph(
        "Limitaciones: septiembre es una muestra retrospectiva de un solo mes, ya usado como "
        "prueba de los baselines, y no son logs reales de la API. La V2 también conserva "
        "git_dirty=True en sus tags de Registry; debe repetirse el entrenamiento desde un "
        "estado limpio antes de cualquier promoción."
    )

    documento.add_heading("Anexo: reporte interactivo de Evidently", level=1)
    parrafo = documento.add_paragraph(
        "El reporte HTML completo, con distribuciones y gráficas interactivas, se entrega junto "
        "a este documento: "
    )
    _enlace(parrafo, "drift_p1.html", RUTA_DRIFT.with_suffix(".html").resolve().as_uri())

    documento.save(ruta_salida)
    return ruta_salida


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUTA_SALIDA_DEFAULT)
    args = parser.parse_args()
    resultado = generar_documento(args.output)
    print(f"Reporte DOCX generado: {resultado}")


if __name__ == "__main__":
    main()
