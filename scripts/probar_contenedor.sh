#!/usr/bin/env bash
# Construye la imagen de la API, levanta el contenedor y prueba /health, /modelo y /predict.
#
# Uso (desde cualquier carpeta, con el modelo ya exportado en models/servicio):
#   ./scripts/probar_contenedor.sh               # Podman, puerto 8000
#   ./scripts/probar_contenedor.sh docker 8080
#
# Argumentos:
#   $1  Motor de contenedores: podman (por defecto) | docker
#   $2  Puerto del host (por defecto: 8000)

set -euo pipefail

MOTOR="${1:-podman}"
PUERTO="${2:-8000}"
IMAGEN="mibici-api:latest"
CONTENEDOR="mibici-api-prueba"
BASE="http://127.0.0.1:${PUERTO}"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$MOTOR" != "podman" && "$MOTOR" != "docker" ]]; then
    echo "Error: motor debe ser 'podman' o 'docker' (recibido: '$MOTOR')." >&2
    exit 1
fi

if [[ ! -f "$RAIZ/models/servicio/registro.json" ]]; then
    echo "Falta models/servicio; ejecuta antes: python -m src.model_package --pregunta p1" >&2
    exit 1
fi

cleanup() {
    "$MOTOR" stop "$CONTENEDOR" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$MOTOR" build -f "$RAIZ/Containerfile" -t "$IMAGEN" "$RAIZ"

"$MOTOR" run -d --rm --name "$CONTENEDOR" -p "${PUERTO}:8000" "$IMAGEN" >/dev/null

# Esperar hasta 60 s a que /health responda
listo=0
for i in $(seq 1 60); do
    if python3 -c "
import urllib.request, json, sys
try:
    r = urllib.request.urlopen('${BASE}/health', timeout=2)
    sys.exit(0 if json.loads(r.read())['status'] == 'ok' else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        listo=1
        break
    fi
    sleep 1
done

if [[ $listo -eq 0 ]]; then
    "$MOTOR" logs "$CONTENEDOR" >&2
    echo "La API no responde en ${BASE}/health." >&2
    exit 1
fi
echo "GET /health -> ok"

# GET /modelo
INFO=$(python3 -c "
import urllib.request, json
r = urllib.request.urlopen('${BASE}/modelo')
d = json.loads(r.read())
print(d['nombre'], 'v' + d['version'], '(run', d['run_id'] + ')')
")
echo "GET /modelo -> $INFO"

# POST /predict con dos instancias válidas
PREDICCIONES=$(python3 -c "
import urllib.request, json
cuerpo = json.dumps({'instancias': [
    {'hora_sin': 0.866, 'hora_cos': -0.5, 'dia_sin': 0.0, 'dia_cos': 1.0,
     'es_fin_de_semana': 0, 'lag_salidas_1h': 3, 'lag_salidas_2h': 2,
     'lag_salidas_3h': 4, 'lag_salidas_24h': 5, 'lag_salidas_168h': 4,
     'rolling_mean_6h': 3.2},
    {'hora_sin': 0.0, 'hora_cos': 1.0, 'dia_sin': -0.7818, 'dia_cos': 0.6235,
     'es_fin_de_semana': 1, 'lag_salidas_1h': 0, 'lag_salidas_2h': 0,
     'lag_salidas_3h': 0, 'lag_salidas_24h': 1, 'lag_salidas_168h': 0,
     'rolling_mean_6h': 0.1}
]}).encode()
req = urllib.request.Request('${BASE}/predict', data=cuerpo,
      headers={'Content-Type': 'application/json'})
r = urllib.request.urlopen(req)
d = json.loads(r.read())
preds = d['predicciones']
if len(preds) != 2 or any(p < 0 for p in preds):
    raise ValueError(f'Respuesta inesperada: {d}')
print(json.dumps(d))
")
echo "POST /predict -> $PREDICCIONES"

# POST /predict con entrada incoherente — debe devolver 422
CODIGO=$(python3 -c "
import urllib.request, json
cuerpo = json.dumps({'instancias': [
    {'hora_sin': 0.866, 'hora_cos': -0.5, 'dia_sin': 0.0, 'dia_cos': 1.0,
     'es_fin_de_semana': 1, 'lag_salidas_1h': 3, 'lag_salidas_2h': 2,
     'lag_salidas_3h': 4, 'lag_salidas_24h': 5, 'lag_salidas_168h': 4,
     'rolling_mean_6h': 3.2}
]}).encode()
req = urllib.request.Request('${BASE}/predict', data=cuerpo,
      headers={'Content-Type': 'application/json'})
try:
    urllib.request.urlopen(req)
    print(200)
except urllib.error.HTTPError as e:
    print(e.code)
    import sys; sys.stderr.write(e.read().decode())
" 2>/tmp/detalle_422)
if [[ "$CODIGO" != "422" ]]; then
    echo "Se esperaba HTTP 422 para entrada incoherente y se obtuvo $CODIGO." >&2
    exit 1
fi
echo "POST /predict (entrada incoherente) -> 422 $(cat /tmp/detalle_422)"

echo "Prueba del contenedor superada."
