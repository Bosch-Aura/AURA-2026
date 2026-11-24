#!/bin/bash
# Reinicia el dashboard de AURA, conectado al ESP32, con log permanente.
# Corre en segundo plano y sobrevive a cerrar la terminal.
#   ./aura.sh          reinicia
#   ./aura.sh stop     detiene
#   ./aura.sh log      muestra el log en vivo (Ctrl+C para salir, no detiene AURA)

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$DIR/logs"

stop_aura() {
    pkill -f "dashboard_server.py" 2>/dev/null
    lsof -ti:5000 2>/dev/null | xargs -r kill -9 2>/dev/null
    sleep 1
}

case "$1" in
    stop)
        stop_aura
        echo "AURA detenido."
        exit 0
        ;;
    log)
        exec tail -f "$(ls -t "$LOG_DIR"/*.log | head -1)"
        ;;
esac

stop_aura

# Puerto del ESP32: cambia entre ttyCH341USB0 y ttyCH341USB1 al reconectar el USB.
PORT="$(ls /dev/ttyCH341USB* /dev/ttyUSB* 2>/dev/null | head -1)"
PORT_ARGS=()
if [[ -n "$PORT" ]]; then
    PORT_ARGS=(--port-name "$PORT")
else
    echo "AVISO: no encontre el ESP32 por USB. Conectalo desde el dashboard."
fi

mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/aura_$(date +%Y%m%d_%H%M%S).log"

cd "$DIR/brain" || exit 1
setsid nohup bash -c "source '$DIR/venv/bin/activate' && PYTHONUNBUFFERED=1 \
    python dashboard/dashboard_server.py --host 0.0.0.0 --port 5000 ${PORT_ARGS[*]} 2>&1 \
    | awk '{ print strftime(\"%H:%M:%S\"), \$0; fflush() }' > '$LOG'" \
    >/dev/null 2>&1 < /dev/null &

for _ in $(seq 1 20); do
    if curl -s -o /dev/null http://localhost:5000/health; then
        echo "AURA corriendo: http://192.168.1.10:5000"
        echo "ESP32: ${PORT:-no conectado}"
        echo "Log:   $LOG"
        exit 0
    fi
    sleep 1
done
echo "ERROR: AURA no respondio en 20 s. Mira el log: $LOG"
exit 1
