// ESP32-S3 + 2 NEMA17 + TB6600
// Access Point com página HTML para:
// - Enviar angulos (t1, t2)
// - Enviar ponto (x,y,z)
// - Jog com step configuravel (0.5, 1, 5 graus)
// - Botao HOME (0°, 0°)
// - Controle de velocidade 0-100% (500us -> 60us)
// - Monitorar angulos atuais e pose FK (x,y,z)
//
// Cinemática:
//  - Junta 1: base, gira em Z, a 8 cm do chão
//  - L1 = 5 cm entre junta 1 e junta 2
//  - L2 = 10 cm após junta 2
//
// Mecânica:
//  - Motor 1: PUL = GPIO 4, DIR = GPIO 15
//  - Motor 2: PUL = GPIO 5, DIR = GPIO 16
//  - 3200 passos/volta no motor
//  - Caixa de redução aprox. 28,3:1 => ~90.560 passos/volta no eixo de saída

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <math.h>

// ---------- Config WiFi AP ----------
const char* AP_SSID = "Robo2DOF";
const char* AP_PASS = "12345678";

WebServer server(80);

// ---------- Config pinos ----------
const int PUL1_PIN = 4;
const int DIR1_PIN = 15;

const int PUL2_PIN = 5;
const int DIR2_PIN = 16;

// ---------- Parametros mecanicos ----------
const float BASE_HEIGHT = 8.0f;  // z fixo da ponta
const float L1 = 5.0f;           // cm
const float L2 = 10.0f;          // cm

// ---------- Passos ----------
const long MOTOR_STEPS_PER_REV = 3200;
const float GEAR_RATIO_ESTIMATED = 28.3f;
const long OUTPUT_STEPS_PER_REV = (long)(MOTOR_STEPS_PER_REV * GEAR_RATIO_ESTIMATED + 0.5f);
const float STEPS_PER_DEG = (float)OUTPUT_STEPS_PER_REV / 360.0f;

// ---------- Tempo de pulsos / velocidade ----------
const int MIN_INTERVAL_US = 60;   // 100%
const int MAX_INTERVAL_US = 500;  // 0%

int currentSpeedPct = 50;         // 0 a 100
int stepIntervalUs  = 500;        // usado de fato nos movimentos

// ---------- Convenção de direcao ----------
const bool DIR1_POSITIVE = HIGH;  // mude pra LOW se inverter
const bool DIR2_POSITIVE = HIGH;

// ---------- Estado atual (graus) ----------
float current_theta1_deg = 0.0f;
float current_theta2_deg = 0.0f;

// ---------- Função para mapear velocidade (%) -> intervalo em us ----------
int computeIntervalFromSpeed(int speedPct) {
  if (speedPct < 0) speedPct = 0;
  if (speedPct > 100) speedPct = 100;

  float diff = (float)(MAX_INTERVAL_US - MIN_INTERVAL_US); // 500 - 60 = 440
  float val  = (float)MAX_INTERVAL_US - ((float)speedPct / 100.0f) * diff;
  int interval = (int)(val + 0.5f);
  if (interval < MIN_INTERVAL_US) interval = MIN_INTERVAL_US;
  if (interval > MAX_INTERVAL_US) interval = MAX_INTERVAL_US;
  return interval;
}

// ---------- HTML da pagina ----------
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="UTF-8">
<title>Robo 2DOF - Controle</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  :root {
    color-scheme: dark;
  }
  body {
    font-family: Arial, sans-serif;
    background: #020617;
    color: #e5e7eb;
    margin: 0;
    padding: 0;
  }
  .container {
    max-width: 900px;
    margin: 0 auto;
    padding: 16px;
  }
  h1, h2 {
    text-align: center;
    color: #facc15;
    margin-top: 4px;
    margin-bottom: 12px;
  }
  .card {
    background: #0b1120;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  label {
    display: block;
    margin-bottom: 4px;
    font-size: 0.9rem;
  }
  input, select {
    width: 100%;
    box-sizing: border-box;
    padding: 6px 8px;
    margin-bottom: 8px;
    border-radius: 6px;
    border: 1px solid #374151;
    background: #020617;
    color: #e5e7eb;
  }
  button {
    background: #22c55e;
    border: none;
    color: #020617;
    padding: 8px 16px;
    border-radius: 999px;
    font-weight: bold;
    cursor: pointer;
    margin-top: 4px;
    margin-right: 4px;
    margin-bottom: 4px;
    white-space: nowrap;
  }
  button:hover {
    background: #16a34a;
  }
  .btn-secondary {
    background: #38bdf8;
  }
  .btn-secondary:hover {
    background: #0ea5e9;
  }
  .btn-danger {
    background: #f97373;
  }
  .btn-danger:hover {
    background: #ef4444;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
  }
  .col {
    flex: 1;
    min-width: 260px;
  }
  .status-item {
    margin-bottom: 4px;
  }
  .status-label {
    font-weight: bold;
    color: #93c5fd;
  }
  .status-value {
    font-family: "Courier New", monospace;
  }
  .msg {
    margin-top: 8px;
    font-size: 0.9rem;
  }
  .msg-ok {
    color: #4ade80;
  }
  .msg-err {
    color: #f97373;
  }
  .jog-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 8px;
  }
  .jog-group {
    flex: 1;
    min-width: 150px;
  }
  .jog-title {
    font-weight: bold;
    margin-bottom: 4px;
  }
  input[type=range] {
    width: 100%;
    margin: 4px 0;
  }
  @media (max-width: 600px) {
    h1 {
      font-size: 1.4rem;
    }
    h2 {
      font-size: 1.1rem;
    }
  }
</style>
</head>
<body>
<div class="container">
  <h1>Robo 2DOF - ESP32-S3</h1>

  <!-- STATUS -->
  <div class="card">
    <h2>Status atual</h2>
    <div class="status-item">
      <span class="status-label">t1 (base):</span>
      <span id="st_t1" class="status-value">0.00</span> °
    </div>
    <div class="status-item">
      <span class="status-label">t2 (cotovelo):</span>
      <span id="st_t2" class="status-value">0.00</span> °
    </div>
    <div class="status-item">
      <span class="status-label">x:</span>
      <span id="st_x" class="status-value">0.00</span> cm
    </div>
    <div class="status-item">
      <span class="status-label">y:</span>
      <span id="st_y" class="status-value">0.00</span> cm
    </div>
    <div class="status-item">
      <span class="status-label">z:</span>
      <span id="st_z" class="status-value">0.00</span> cm
    </div>
    <div class="status-item">
      <span class="status-label">Velocidade:</span>
      <span id="st_speed" class="status-value">50</span> %
    </div>
    <div id="status_msg" class="msg"></div>
  </div>

  <!-- JOG + HOME -->
  <div class="card">
    <h2>Jog &amp; Home</h2>
    <label for="jog_step">Passo de jog (graus):</label>
    <select id="jog_step">
      <option value="0.5">0.5°</option>
      <option value="1" selected>1°</option>
      <option value="5">5°</option>
    </select>

    <div class="jog-row">
      <div class="jog-group">
        <div class="jog-title">Junta 1 (base)</div>
        <button type="button" onclick="sendJog(1,-1)">J1 -</button>
        <button type="button" onclick="sendJog(1, 1)">J1 +</button>
      </div>
      <div class="jog-group">
        <div class="jog-title">Junta 2 (cotovelo)</div>
        <button type="button" onclick="sendJog(2,-1)">J2 -</button>
        <button type="button" onclick="sendJog(2, 1)">J2 +</button>
      </div>
    </div>

    <div style="margin-top: 12px;">
      <button type="button" class="btn-secondary" onclick="sendHome()">Home (0°, 0°)</button>
    </div>

    <div id="msg_jog" class="msg"></div>
  </div>

  <!-- ANGULOS & XYZ -->
  <div class="row">
    <div class="card col">
      <h2>Controle por angulos</h2>
      <form id="formAngles">
        <label for="in_t1">t1 (base, graus):</label>
        <input id="in_t1" type="number" step="0.1" value="0">
        <label for="in_t2">t2 (cotovelo, graus):</label>
        <input id="in_t2" type="number" step="0.1" value="0">
        <button type="submit">Enviar angulos</button>
      </form>
      <div id="msg_angles" class="msg"></div>
    </div>

    <div class="card col">
      <h2>Controle por XYZ</h2>
      <form id="formXYZ">
        <label for="in_x">X (cm):</label>
        <input id="in_x" type="number" step="0.1" value="15">
        <label for="in_y">Y (cm):</label>
        <input id="in_y" type="number" step="0.1" value="0">
        <label for="in_z">Z (cm) (fixo ~8):</label>
        <input id="in_z" type="number" step="0.1" value="8">
        <button type="submit">Enviar XYZ</button>
      </form>
      <div id="msg_xyz" class="msg"></div>
    </div>
  </div>

  <!-- VELOCIDADE -->
  <div class="card">
    <h2>Velocidade</h2>
    <label for="in_speed">Velocidade (0 a 100%):</label>
    <input id="in_speed" type="range" min="0" max="100" value="50">
    <div class="status-item">
      <span class="status-label">Atual:</span>
      <span id="speed_value" class="status-value">50</span> %
    </div>
    <div id="msg_speed" class="msg"></div>
  </div>

</div>

<script>
async function updateStatus() {
  try {
    const res = await fetch('/status');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const data = await res.json();
    document.getElementById('st_t1').textContent = data.t1.toFixed(2);
    document.getElementById('st_t2').textContent = data.t2.toFixed(2);
    document.getElementById('st_x').textContent  = data.x.toFixed(2);
    document.getElementById('st_y').textContent  = data.y.toFixed(2);
    document.getElementById('st_z').textContent  = data.z.toFixed(2);

    if (typeof data.speed === 'number') {
      const sp = data.speed;
      document.getElementById('st_speed').textContent = sp.toFixed(0);
      const slider = document.getElementById('in_speed');
      const label  = document.getElementById('speed_value');
      if (slider) slider.value = sp;
      if (label)  label.textContent = sp.toFixed(0);
    }

    const stMsg = document.getElementById('status_msg');
    stMsg.textContent = 'Status atualizado';
    stMsg.className = 'msg msg-ok';
  } catch (e) {
    const stMsg = document.getElementById('status_msg');
    stMsg.textContent = 'Erro ao atualizar status: ' + e.message;
    stMsg.className = 'msg msg-err';
  }
}

async function sendAngles(ev) {
  ev.preventDefault();
  const t1 = document.getElementById('in_t1').value;
  const t2 = document.getElementById('in_t2').value;
  const msgBox = document.getElementById('msg_angles');
  msgBox.textContent = 'Enviando...';
  msgBox.className = 'msg';

  try {
    const body = 't1=' + encodeURIComponent(t1) + '&t2=' + encodeURIComponent(t2);
    const res = await fetch('/set_angles', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body
    });
    const data = await res.json();
    if (data.success) {
      msgBox.textContent = 'Movimento concluido.';
      msgBox.className = 'msg msg-ok';
      updateStatus();
    } else {
      msgBox.textContent = 'Erro: ' + (data.msg || 'desconhecido');
      msgBox.className = 'msg msg-err';
    }
  } catch (e) {
    msgBox.textContent = 'Erro na requisicao: ' + e.message;
    msgBox.className = 'msg msg-err';
  }
}

async function sendXYZ(ev) {
  ev.preventDefault();
  const x = document.getElementById('in_x').value;
  const y = document.getElementById('in_y').value;
  const z = document.getElementById('in_z').value;
  const msgBox = document.getElementById('msg_xyz');
  msgBox.textContent = 'Enviando...';
  msgBox.className = 'msg';

  try {
    const body = 'x=' + encodeURIComponent(x) + '&y=' + encodeURIComponent(y) + '&z=' + encodeURIComponent(z);
    const res = await fetch('/set_xyz', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body
    });
    const data = await res.json();
    if (data.success) {
      msgBox.textContent = 'Movimento concluido.';
      msgBox.className = 'msg msg-ok';
      updateStatus();
    } else {
      msgBox.textContent = 'Erro: ' + (data.msg || 'desconhecido');
      msgBox.className = 'msg msg-err';
    }
  } catch (e) {
    msgBox.textContent = 'Erro na requisicao: ' + e.message;
    msgBox.className = 'msg msg-err';
  }
}

async function sendSpeed(ev) {
  const slider = document.getElementById('in_speed');
  const val = slider.value;
  const label = document.getElementById('speed_value');
  label.textContent = val;
  const msgBox = document.getElementById('msg_speed');
  msgBox.textContent = 'Atualizando velocidade...';
  msgBox.className = 'msg';

  try {
    const body = 'speed=' + encodeURIComponent(val);
    const res = await fetch('/set_speed', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body
    });
    const data = await res.json();
    if (data.success) {
      msgBox.textContent = 'Velocidade atualizada.';
      msgBox.className = 'msg msg-ok';
      updateStatus();
    } else {
      msgBox.textContent = 'Erro: ' + (data.msg || 'desconhecido');
      msgBox.className = 'msg msg-err';
    }
  } catch (e) {
    msgBox.textContent = 'Erro na requisicao: ' + e.message;
    msgBox.className = 'msg msg-err';
  }
}

async function sendJog(joint, dir) {
  const stepSel = document.getElementById('jog_step');
  const stepVal = parseFloat(stepSel.value || '1');
  const delta = stepVal * dir;
  const msgBox = document.getElementById('msg_jog');
  msgBox.textContent = 'Enviando jog...';
  msgBox.className = 'msg';

  try {
    const body = 'joint=' + encodeURIComponent(joint) + '&delta=' + encodeURIComponent(delta);
    const res = await fetch('/jog', {
      method: 'POST',
      headers: {'Content-Type': 'application/x-www-form-urlencoded'},
      body
    });
    const data = await res.json();
    if (data.success) {
      msgBox.textContent = 'Jog concluido.';
      msgBox.className = 'msg msg-ok';
      updateStatus();
    } else {
      msgBox.textContent = 'Erro: ' + (data.msg || 'desconhecido');
      msgBox.className = 'msg msg-err';
    }
  } catch (e) {
    msgBox.textContent = 'Erro na requisicao: ' + e.message;
    msgBox.className = 'msg msg-err';
  }
}

async function sendHome() {
  const msgBox = document.getElementById('msg_jog');
  msgBox.textContent = 'Indo para HOME (0°,0°)...';
  msgBox.className = 'msg';

  try {
    const res = await fetch('/home', {
      method: 'POST'
    });
    const data = await res.json();
    if (data.success) {
      msgBox.textContent = 'Home concluido.';
      msgBox.className = 'msg msg-ok';
      updateStatus();
    } else {
      msgBox.textContent = 'Erro: ' + (data.msg || 'desconhecido');
      msgBox.className = 'msg msg-err';
    }
  } catch (e) {
    msgBox.textContent = 'Erro na requisicao: ' + e.message;
    msgBox.className = 'msg msg-err';
  }
}

document.getElementById('formAngles').addEventListener('submit', sendAngles);
document.getElementById('formXYZ').addEventListener('submit', sendXYZ);
document.getElementById('in_speed').addEventListener('change', sendSpeed);
document.getElementById('in_speed').addEventListener('input', function(ev) {
  document.getElementById('speed_value').textContent = ev.target.value;
});

updateStatus();
setInterval(updateStatus, 1500);
</script>

</body>
</html>
)rawliteral";

// ======================================================================
// Funcoes de movimento / cinematica
// ======================================================================

long degToSteps(float degrees) {
  float steps = degrees * STEPS_PER_DEG;
  if (steps >= 0.0f) {
    return (long)(steps + 0.5f);
  } else {
    return (long)(steps - 0.5f);
  }
}

void moveMotorSteps(int pulPin, int dirPin, bool dirLevel, long steps) {
  if (steps <= 0) return;

  int interval = stepIntervalUs; // pega valor atual da velocidade

  digitalWrite(dirPin, dirLevel ? HIGH : LOW);
  delay(5);

  for (long i = 0; i < steps; i++) {
    digitalWrite(pulPin, HIGH);
    delayMicroseconds(5);           // largura do pulso
    digitalWrite(pulPin, LOW);
    delayMicroseconds(interval);    // intervalo entre passos
  }
}

void moveJointToAngle(int joint, float targetDeg) {
  float *currentAngle;
  int pulPin, dirPin;
  bool dirPositive;

  if (joint == 1) {
    currentAngle = &current_theta1_deg;
    pulPin = PUL1_PIN;
    dirPin = DIR1_PIN;
    dirPositive = DIR1_POSITIVE;
  } else {
    currentAngle = &current_theta2_deg;
    pulPin = PUL2_PIN;
    dirPin = DIR2_PIN;
    dirPositive = DIR2_POSITIVE;
  }

  float deltaDeg = targetDeg - (*currentAngle);
  long deltaSteps = degToSteps(deltaDeg);

  if (deltaSteps == 0) {
    Serial.printf("Junta %d ja esta em %.2f°\n", joint, *currentAngle);
    return;
  }

  bool dirLevel = (deltaSteps > 0) ? dirPositive : !dirPositive;
  long steps = labs(deltaSteps);

  Serial.printf("Junta %d: de %.2f° para %.2f° (delta=%.2f°, steps=%ld, interval=%dus)\n",
                joint, *currentAngle, targetDeg, deltaDeg, steps, stepIntervalUs);

  moveMotorSteps(pulPin, dirPin, dirLevel, steps);
  *currentAngle = targetDeg;
}

// FK: a partir de t1,t2 (graus) -> x,y,z (cm)
void forwardKinematics(float t1_deg, float t2_deg, float &x, float &y, float &z) {
  float t1 = t1_deg * PI / 180.0f;
  float t2 = t2_deg * PI / 180.0f;

  x = L1 * cosf(t1) + L2 * cosf(t1 + t2);
  y = L1 * sinf(t1) + L2 * sinf(t1 + t2);
  z = BASE_HEIGHT;
}

// IK: a partir de x,y -> t1,t2 (graus). Retorna true se tiver solucao.
bool inverseKinematics(float x, float y, float &t1_deg, float &t2_deg) {
  float r2 = x * x + y * y;
  float r = sqrtf(r2);

  float minReach = fabsf(L1 - L2);
  float maxReach = L1 + L2;

  if (r < minReach - 1e-3f || r > maxReach + 1e-3f) {
    return false;
  }

  float c2 = (r2 - L1 * L1 - L2 * L2) / (2.0f * L1 * L2);
  if (c2 < -1.0f) c2 = -1.0f;
  if (c2 > 1.0f)  c2 = 1.0f;

  float s2 = sqrtf(fmaxf(0.0f, 1.0f - c2 * c2));  // cotovelo "para frente"
  float theta2 = atan2f(s2, c2);

  float k1 = L1 + L2 * c2;
  float k2 = L2 * s2;
  float theta1 = atan2f(y, x) - atan2f(k2, k1);

  t1_deg = theta1 * 180.0f / PI;
  t2_deg = theta2 * 180.0f / PI;
  return true;
}

// ======================================================================
// Handlers HTTP
// ======================================================================

void handleRoot() {
  server.send_P(200, "text/html", INDEX_HTML);
}

void handleStatus() {
  float x, y, z;
  forwardKinematics(current_theta1_deg, current_theta2_deg, x, y, z);

  String json = "{";
  json += "\"success\":true,";
  json += "\"t1\":" + String(current_theta1_deg, 2) + ",";
  json += "\"t2\":" + String(current_theta2_deg, 2) + ",";
  json += "\"x\":" + String(x, 2) + ",";
  json += "\"y\":" + String(y, 2) + ",";
  json += "\"z\":" + String(z, 2) + ",";
  json += "\"speed\":" + String(currentSpeedPct);
  json += "}";

  server.send(200, "application/json", json);
}

void handleSetAngles() {
  if (!server.hasArg("t1") || !server.hasArg("t2")) {
    server.send(400, "application/json", "{\"success\":false,\"msg\":\"Parametros t1 e t2 obrigatorios\"}");
    return;
  }

  float t1 = server.arg("t1").toFloat();
  float t2 = server.arg("t2").toFloat();

  Serial.printf("HTTP: set_angles t1=%.2f t2=%.2f\n", t1, t2);

  moveJointToAngle(1, t1);
  moveJointToAngle(2, t2);

  float x, y, z;
  forwardKinematics(current_theta1_deg, current_theta2_deg, x, y, z);

  String json = "{";
  json += "\"success\":true,";
  json += "\"t1\":" + String(current_theta1_deg, 2) + ",";
  json += "\"t2\":" + String(current_theta2_deg, 2) + ",";
  json += "\"x\":" + String(x, 2) + ",";
  json += "\"y\":" + String(y, 2) + ",";
  json += "\"z\":" + String(z, 2) + ",";
  json += "\"speed\":" + String(currentSpeedPct);
  json += "}";

  server.send(200, "application/json", json);
}

void handleSetXYZ() {
  if (!server.hasArg("x") || !server.hasArg("y")) {
    server.send(400, "application/json", "{\"success\":false,\"msg\":\"Parametros x e y obrigatorios\"}");
    return;
  }

  float x = server.arg("x").toFloat();
  float y = server.arg("y").toFloat();
  float z = BASE_HEIGHT;
  if (server.hasArg("z")) {
    z = server.arg("z").toFloat();
  }

  if (fabsf(z - BASE_HEIGHT) > 0.5f) {
    Serial.printf("Aviso: Z solicitado=%.2f, mas robo opera em z=%.2f. Ignorando.\n", z, BASE_HEIGHT);
  }

  Serial.printf("HTTP: set_xyz x=%.2f y=%.2f z=%.2f\n", x, y, z);

  float t1_deg, t2_deg;
  if (!inverseKinematics(x, y, t1_deg, t2_deg)) {
    server.send(400, "application/json", "{\"success\":false,\"msg\":\"Ponto fora da area de alcance\"}");
    return;
  }

  moveJointToAngle(1, t1_deg);
  moveJointToAngle(2, t2_deg);

  float xf, yf, zf;
  forwardKinematics(current_theta1_deg, current_theta2_deg, xf, yf, zf);

  String json = "{";
  json += "\"success\":true,";
  json += "\"t1\":" + String(current_theta1_deg, 2) + ",";
  json += "\"t2\":" + String(current_theta2_deg, 2) + ",";
  json += "\"x\":" + String(xf, 2) + ",";
  json += "\"y\":" + String(yf, 2) + ",";
  json += "\"z\":" + String(zf, 2) + ",";
  json += "\"speed\":" + String(currentSpeedPct);
  json += "}";

  server.send(200, "application/json", json);
}

void handleSetSpeed() {
  if (!server.hasArg("speed")) {
    server.send(400, "application/json", "{\"success\":false,\"msg\":\"Parametro speed obrigatorio\"}");
    return;
  }

  int sp = server.arg("speed").toInt();
  if (sp < 0) sp = 0;
  if (sp > 100) sp = 100;

  currentSpeedPct = sp;
  stepIntervalUs = computeIntervalFromSpeed(currentSpeedPct);

  Serial.printf("HTTP: set_speed speed=%d%% -> interval=%dus\n", currentSpeedPct, stepIntervalUs);

  String json = "{";
  json += "\"success\":true,";
  json += "\"speed\":" + String(currentSpeedPct) + ",";
  json += "\"interval_us\":" + String(stepIntervalUs);
  json += "}";

  server.send(200, "application/json", json);
}

void handleJog() {
  if (!server.hasArg("joint") || !server.hasArg("delta")) {
    server.send(400, "application/json", "{\"success\":false,\"msg\":\"Parametros joint e delta obrigatorios\"}");
    return;
  }

  int joint = server.arg("joint").toInt();
  float delta = server.arg("delta").toFloat();

  if (joint != 1 && joint != 2) {
    server.send(400, "application/json", "{\"success\":false,\"msg\":\"Joint invalido\"}");
    return;
  }

  Serial.printf("HTTP: jog joint=%d delta=%.2f\n", joint, delta);

  float target;
  if (joint == 1) {
    target = current_theta1_deg + delta;
  } else {
    target = current_theta2_deg + delta;
  }

  moveJointToAngle(joint, target);

  float x, y, z;
  forwardKinematics(current_theta1_deg, current_theta2_deg, x, y, z);

  String json = "{";
  json += "\"success\":true,";
  json += "\"t1\":" + String(current_theta1_deg, 2) + ",";
  json += "\"t2\":" + String(current_theta2_deg, 2) + ",";
  json += "\"x\":" + String(x, 2) + ",";
  json += "\"y\":" + String(y, 2) + ",";
  json += "\"z\":" + String(z, 2) + ",";
  json += "\"speed\":" + String(currentSpeedPct);
  json += "}";

  server.send(200, "application/json", json);
}

void handleHome() {
  Serial.println("HTTP: home -> (0°,0°)");
  moveJointToAngle(1, 0.0f);
  moveJointToAngle(2, 0.0f);

  float x, y, z;
  forwardKinematics(current_theta1_deg, current_theta2_deg, x, y, z);

  String json = "{";
  json += "\"success\":true,";
  json += "\"t1\":" + String(current_theta1_deg, 2) + ",";
  json += "\"t2\":" + String(current_theta2_deg, 2) + ",";
  json += "\"x\":" + String(x, 2) + ",";
  json += "\"y\":" + String(y, 2) + ",";
  json += "\"z\":" + String(z, 2) + ",";
  json += "\"speed\":" + String(currentSpeedPct);
  json += "}";

  server.send(200, "application/json", json);
}

void handleNotFound() {
  server.send(404, "text/plain", "404 Not Found");
}

// ======================================================================
// SETUP / LOOP
// ======================================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  pinMode(PUL1_PIN, OUTPUT);
  pinMode(DIR1_PIN, OUTPUT);
  pinMode(PUL2_PIN, OUTPUT);
  pinMode(DIR2_PIN, OUTPUT);

  // velocidade inicial
  currentSpeedPct = 50;
  stepIntervalUs  = computeIntervalFromSpeed(currentSpeedPct);

  Serial.println();
  Serial.println("=== Robo 2DOF - AP + Web (Jog + Home + Velocidade) ===");
  Serial.print("OUTPUT_STEPS_PER_REV: ");
  Serial.println(OUTPUT_STEPS_PER_REV);
  Serial.print("STEPS_PER_DEG: ");
  Serial.println(STEPS_PER_DEG, 4);
  Serial.print("Velocidade inicial: ");
  Serial.print(currentSpeedPct);
  Serial.print("% -> intervalo ");
  Serial.print(stepIntervalUs);
  Serial.println(" us");

  // WiFi AP
  WiFi.mode(WIFI_AP);
  bool apOk = WiFi.softAP(AP_SSID, AP_PASS);
  if (apOk) {
    Serial.print("AP iniciado. SSID: ");
    Serial.println(AP_SSID);
    Serial.print("IP: ");
    Serial.println(WiFi.softAPIP());
  } else {
    Serial.println("Falha ao iniciar AP!");
  }

  // Rotas HTTP
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/set_angles", HTTP_POST, handleSetAngles);
  server.on("/set_xyz", HTTP_POST, handleSetXYZ);
  server.on("/set_speed", HTTP_POST, handleSetSpeed);
  server.on("/jog", HTTP_POST, handleJog);
  server.on("/home", HTTP_POST, handleHome);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("Servidor HTTP iniciado.");
}

void loop() {
  server.handleClient();
}
