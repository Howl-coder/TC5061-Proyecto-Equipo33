<#
Construye la imagen de la API, levanta el contenedor y prueba /health, /modelo y /predict.

Uso (desde cualquier carpeta, con el modelo ya exportado en models/servicio):
    .\scripts\probar_contenedor.ps1                  # Podman, puerto 8000
    .\scripts\probar_contenedor.ps1 -Motor docker -Puerto 8080
Solo ASCII: Windows PowerShell 5.1 lee los .ps1 sin BOM como ANSI.
#>
param(
    [ValidateSet("podman", "docker")] [string]$Motor = "podman",
    [int]$Puerto = 8000,
    [string]$Imagen = "mibici-api:latest"
)

$raiz = Split-Path -Parent $PSScriptRoot
$contenedor = "mibici-api-prueba"
$base = "http://127.0.0.1:$Puerto"

if (-not (Test-Path (Join-Path $raiz "models/servicio/registro.json"))) {
    throw "Falta models/servicio; ejecuta antes: python -m src.model_package --pregunta p1"
}

& $Motor build -f (Join-Path $raiz "Containerfile") -t $Imagen $raiz
if ($LASTEXITCODE -ne 0) { throw "Error al construir la imagen." }

& $Motor run -d --rm --name $contenedor -p "${Puerto}:8000" $Imagen | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No se pudo iniciar el contenedor." }

try {
    $listo = $false
    for ($i = 0; $i -lt 60 -and -not $listo; $i++) {
        try { $listo = (Invoke-RestMethod "$base/health").status -eq "ok" }
        catch { Start-Sleep -Seconds 1 }
    }
    if (-not $listo) {
        & $Motor logs $contenedor
        throw "La API no responde en $base/health."
    }
    Write-Host "GET /health -> ok"

    $modelo = Invoke-RestMethod "$base/modelo"
    Write-Host "GET /modelo -> $($modelo.nombre) v$($modelo.version) (run $($modelo.run_id))"

    $laborable = @{
        hora_sin = 0.866; hora_cos = -0.5; dia_sin = 0.0; dia_cos = 1.0; es_fin_de_semana = 0
        lag_salidas_1h = 3; lag_salidas_2h = 2; lag_salidas_3h = 4
        lag_salidas_24h = 5; lag_salidas_168h = 4; rolling_mean_6h = 3.2
    }
    $domingo = @{
        hora_sin = 0.0; hora_cos = 1.0; dia_sin = -0.7818; dia_cos = 0.6235; es_fin_de_semana = 1
        lag_salidas_1h = 0; lag_salidas_2h = 0; lag_salidas_3h = 0
        lag_salidas_24h = 1; lag_salidas_168h = 0; rolling_mean_6h = 0.1
    }
    $cuerpo = @{ instancias = @($laborable, $domingo) } | ConvertTo-Json -Depth 4
    $respuesta = Invoke-RestMethod -Method Post -Uri "$base/predict" -ContentType "application/json" -Body $cuerpo
    if ($respuesta.predicciones.Count -ne 2 -or ($respuesta.predicciones | Where-Object { $_ -lt 0 })) {
        throw "Respuesta inesperada de /predict: $($respuesta | ConvertTo-Json -Compress)"
    }
    Write-Host "POST /predict -> $($respuesta | ConvertTo-Json -Compress)"

    $incoherente = $laborable.Clone()
    $incoherente.es_fin_de_semana = 1
    $codigo = 0
    try {
        Invoke-RestMethod -Method Post -Uri "$base/predict" -ContentType "application/json" `
            -Body (@{ instancias = @($incoherente) } | ConvertTo-Json -Depth 4) | Out-Null
        $codigo = 200
    }
    catch {
        if ($_.Exception.Response) { $codigo = [int]$_.Exception.Response.StatusCode }
        $detalle = $_.ErrorDetails.Message
    }
    if ($codigo -ne 422) { throw "Se esperaba HTTP 422 para una entrada incoherente y se obtuvo $codigo." }
    Write-Host "POST /predict (entrada incoherente) -> 422 $detalle"

    Write-Host "Prueba del contenedor superada." -ForegroundColor Green
}
finally {
    & $Motor stop $contenedor | Out-Null
}
