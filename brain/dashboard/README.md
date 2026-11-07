# ESP32 Car Control Dashboard

Dashboard web completo para controlar el ESP32 vía **UART** desde cualquier dispositivo en la red.

## 📁 Contenido

- `dashboard.html` - Interfaz web del dashboard
- `dashboard_server.py` - Servidor Flask que traduce comandos HTTP a UART
- `requirements.txt` - Dependencias de Python (Flask, pyserial)
- `start_dashboard.sh` - Script de inicio rápido
- `DASHBOARD_README.md` - Documentación del dashboard
- `DASHBOARD_SERVER_README.md` - Documentación del servidor

## 🚀 Inicio Rápido

### Opción 1: Conectar UART automáticamente al iniciar
```bash
cd dashboard
pip install -r requirements.txt
python dashboard_server.py --port-name /dev/ttyCH341USB0
# En Windows: python dashboard_server.py --uart-port COM3
```

### Opción 2: Conectar UART desde la interfaz web
```bash
cd dashboard
pip install -r requirements.txt
python dashboard_server.py
```
Luego en el navegador:
1. Abre el dashboard (http://localhost:5000)
2. Haz clic en "Refresh Ports"
3. Selecciona el puerto UART
4. Haz clic en "Connect UART"

### Opción 3: Usar el script de inicio
```bash
cd dashboard
python dashboard/dashboard_server.py --port-name /dev/ttyCH341USB0
```

## 🌐 Acceso

Una vez iniciado el servidor, verás algo como:

```
============================================================
ESP32 Car Control Dashboard Server (UART Mode)
============================================================
Dashboard available at:
  Local:   http://127.0.0.1:5000
  Network: http://192.168.1.100:5000
============================================================
UART: Connected to /dev/ttyCH341USB0
============================================================
```

- **Local**: Acceso desde la misma máquina
- **Network**: Acceso desde cualquier dispositivo en la misma red (usar esta IP)
- **UART**: Conexión serial al ESP32 por USB (115200 baud)

## ✨ Características

- ✅ **Comunicación UART**: Conecta directamente al ESP32 vía puerto serial
- ✅ Control completo del ESP32 (motor, dirección, sistema)
- ✅ Interfaz web moderna y responsive
- ✅ Selección de puerto UART desde la interfaz
- ✅ Log de eventos con comandos UART mostrados
- ✅ Accesible desde cualquier dispositivo en la red
- ✅ Fácil de migrar (todo en esta carpeta)

## 🔌 Configuración UART

- **Firmware**: [Bosch-Aura/embedded](https://github.com/Bosch-Aura/embedded) (no se modifica desde este repo)
- **Baud rate**: 115200 por USB (configurado automáticamente)
- **Puerto**: en la Jetson `/dev/ttyCH341USB0` (driver WCH); en una PC suele ser `/dev/ttyUSB0`
- **Protocolo**: `CHANNEL:COMMAND:VALUE\n` (ver `embedded/docs/BRAIN_TEAM_PROTOCOL.md`), implementado en `brain/command_sender.py`

Comportamiento del enlace:

- **Arranque**: abrir el puerto reinicia el ESP32. Al recibir `EVENT:SYSTEM_READY` (o a los 3 s si no llega) se manda `M:SYS_ARM:0`. Si la placa se reinicia en medio de la sesión se rearma sola, pero no se mueve hasta un `C:SET_SPEED` nuevo.
- **Watchdog**: el heartbeat manda `M:PING:0` cada 50 ms (el ESP32 pasa a `FAULT` sin tráfico durante 300 ms). `FAULT` no se rearma solo: usar el toggle *Armed*.
- **Velocidad**: la UI y las estrategias siguen usando 0-255. Se mapea a duty `140..255`. El umbral de arranque del firmware es 130 (medido en banco); 140 deja margen para el auto cargado en el piso. Por debajo del umbral el motor zumba sin girar y el ESP32 igual confirma `CMD_EXECUTED`.
- **Rechazos**: si el ESP32 ignora un `C:SET_SPEED` emite `EVENT:CMD_REJECTED:SET_SPEED:<motivo>`. Con `OBSTACLE` se conserva la velocidad para reenviarla al despejar; con `DISARMED`, `FAULT` o `SYS_STOP` se olvida y se cancelan las estrategias, para que el auto no arranque solo después. Marcha atrás con `C:SET_DIR:0`, enviado solo cuando cambia el sentido.
- **Dirección**: el valor de servo (50-160) se manda tal cual. Si el auto gira al revés, poner `STEER_INVERTED = True` en `command_sender.py`.
- **Ultrasonido**: lo maneja el ESP32. Con `EVENT:OBSTACLE_DETECTED` pasa a `STOPPED` y bloquea **solo el avance**: la marcha atrás y la dirección siguen funcionando. Al llegar `EVENT:OBSTACLE_CLEARED`, el dashboard reenvía la última velocidad pedida.
  - El ESP32 despeja solo con lecturas lejanas válidas. Si se saca el obstáculo y no queda nada dentro del alcance, el auto se queda esperando: para eso está el botón **RESET OBSTACLE** (`M:OBSTACLE_RESET`). Después del reset **no** se reenvía la velocidad, porque el obstáculo podría seguir ahí: hay que mandarla a mano.
  - Un obstáculo durante la espera o el avance del estacionamiento aborta la maniobra.
  - `EVENT:ULTRASONIC_SILENT` / `ULTRASONIC_OK` son informativos: 2 s sin lecturas válidas (espacio abierto o sensor desconectado) y su vuelta. Se ven en `/health` como `sensor_silent`.
- **Freno y paradas**: `E:BRAKE_NOW` frena una sola vez, sin bloqueo. Por eso EMERGENCY BRAKE, STOP, velocidad 0 y DISARM cancelan antes de frenar todo lo que las estrategias tengan programado (reanudaciones, estacionamiento). Lo mismo pasa con `FAULT`, `DISARMED` y un reinicio de la placa, que además olvidan la última velocidad: después de rearmar, el auto no se mueve hasta un comando nuevo.
- **Modo AUTO/MANUAL**: el ESP32 no tiene modos; el toggle solo cambia el estado del lado de la Jetson.

## 📖 Documentación

- Ver `DASHBOARD_README.md` para detalles sobre el uso del dashboard
- Ver `DASHBOARD_SERVER_README.md` para detalles sobre el servidor Flask

## 🔧 Migración

Para migrar el dashboard a otra máquina:

1. Copiar toda la carpeta `dashboard`
2. Instalar dependencias: `pip install -r requirements.txt`
3. Ejecutar: `python dashboard_server.py`

Todo está contenido en esta carpeta, por lo que es fácil de mover.

