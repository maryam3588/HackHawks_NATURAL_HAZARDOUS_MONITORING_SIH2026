/*
  NODE_02 - Wildfire + Extreme Heat
  ESP32-S3 | DHT11 + MQ2 smoke + flame sensor -> on-device TFLite Micro -> JSON

  Output: JSON only, no web page.
    - WiFi: open  http://<node-ip>/  (or /data)  on any phone/laptop on the same network.
      If the Pi's hotspot is unreachable the node starts its own hotspot (NODE02-JSON,
      http://192.168.4.1/) and keeps retrying the Pi; the hotspot turns off once it connects.
    - Raspberry Pi: the Pi replaces the ESP8266 gateway. The node joins the Pi's hotspot
      (bash hotspot.sh on -> HAZARD-NET) and POSTs the JSON to PI_URL every POST_MS, and at
      once when the alert level changes (e.g. instant flame alert). A slow or missing Pi
      doesn't block sampling or alerts for long. Watch it on http://192.168.4.1:3000/esp
    - Serial (115200): the same JSON, one line per sample (SERIAL_JSON 0 to disable).
    - Lines with "event" or "error" (Serial only) are status messages.

  Sketch folder must contain:  this .ino, heat_model.h, fire_model.h  (your own generated headers)
  Library needed:              Chirale_TensorFlowLite  (Library Manager), DHT sensor library
*/

#include <DHT.h>
#include <math.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "heat_model.h"
#include "flame_model.h"   // <<< CHECK >>> your project has flame_model.h, not fire_model.h

// Arduino auto-generates prototypes for every function below and inserts them right
// here, before struct MLModel is actually defined further down. modelSetup/pushWindow/
// predict all take MLModel& as a parameter, so without this forward declaration those
// auto-generated prototypes reference an unknown type and fail to compile.
struct MLModel;

// Same issue as MLModel above: setAlert() takes AlertState by value, so the enum must
// be fully defined before Arduino's auto-generated prototypes, not just declared.
enum AlertState { AL_WARMUP = 0, AL_LOW, AL_MEDIUM, AL_HIGH, AL_CRITICAL, AL_FAULT };

// ---------------- Config ----------------
#define NODE_ID     "NODE_02"
#define SAMPLE_MS   2000        // <<< CHECK >>> must match the time step used to train the models
#define ARENA_SIZE  (32 * 1024)
#define SERIAL_JSON 1

// ---------------- WiFi ----------------
const char* WIFI_SSID = "HAZARD-NET";        // the Pi's hotspot (bash hotspot.sh on), 2.4 GHz
const char* WIFI_PASS = "hazard1234";        // must match HOTSPOT_PASSWORD on the Pi
const char* AP_SSID   = "NODE02-JSON";           // fallback hotspot (used only if the gateway is unreachable)
const char* AP_PASS   = "12345678";

// ---------------- Raspberry Pi ----------------
const char* PI_URL = "http://192.168.4.1:3000/api/sensor-data";   // the Pi on its own hotspot
#define POST_MS        5000     // send the latest reading to the Pi every 5 s (sampling stays SAMPLE_MS)
#define PI_TIMEOUT_MS  400      // the Pi saves the reading even if its answer comes later than this

// ---------------- Pins ----------------
#define DHTPIN    4
#define DHTTYPE   DHT11         // DHT11 only reads 0-50 C. Use DHT22 for real extreme-heat sensing.
#define SMOKE_PIN 1             // MQ2 AO  (ADC1). AO can reach 5V: use a divider (e.g. 10k/20k) to stay <= 3.3V
#define FLAME_PIN 2             // flame sensor AO (ADC1)
#define FLAME_DO_PIN 5          // flame sensor DO (digital)
#define FLAME_DO_ACTIVE_LOW 1   // most modules pull DO LOW when a flame is seen

// ---------------- LED + buzzer alerts (same wiring as Node 1) ----------------
#define LED_GREEN_PIN   10      // each LED: GPIO -> 220-330 ohm -> LED(+), LED(-) -> GND
#define LED_YELLOW_PIN  11
#define LED_RED_PIN     12
#define BUZZER_PIN      13      // active buzzer (+) -> GPIO, (-) -> GND
#define BUZZER_ACTIVE_HIGH 1    // 1 = buzzer sounds when the pin is HIGH
#define BLINK_SLOW_MS   500     // warm-up blink half-period
#define BLINK_FAST_MS   125     // critical / fault blink half-period

// Instant alert: the flame sensor's DO pin triggers CRITICAL at once, without waiting for the models
#define INSTANT_FLAME_ALERT 1
#define FLAME_CONFIRM_MS    200     // flame must be seen this long (filters flicker / electrical noise)
#define FLAME_HOLD_MS       10000   // keep the alarm this long after the flame disappears

// ---------------- Model constants ----------------
#define MAX_WIN  15
#define MAX_FEAT 6

// heat_model.h / fire_model.h, as generated, are raw TFLite byte arrays only - they
// don't define any of the names below. The window/feature sizes come straight from
// the shape comments at the top of each header ([1,4,2] for heat, [1,15,6] for fire)
// and match hf[2]/ff[6] further down, so they're safe to hardcode.
#define HEAT_MODEL_DATA    heat_model_data      // <<< CHECK >>> must match the array name in heat_model.h
#define HEAT_WINDOW_LEN    4
#define HEAT_NUM_FEATURES  2

#define FIRE_MODEL_DATA    flame_model_data     // <<< CHECK >>> must match the array name in fire_model.h
#define FIRE_WINDOW_LEN    15
#define FIRE_NUM_FEATURES  6

// Computed from disaster_training_dataset.xlsx (node_id == "node2", 14400 rows),
// NOT from node2_fire_heat_dataset.csv (the file train_heat.py/train_fire.py
// actually read) -- that file was never supplied. Column mapping used:
//   temperature_C  <- temperature_c   (direct match)
//   humidity_pct   <- humidity_pct    (direct match)
//   mq2_smoke      <- smoke_raw       (closest available column, unconfirmed same scale)
//   mq2_gas        <- gas_raw         (closest available column, unconfirmed same scale)
//   flame_detected <- flame           (binary 0/1, direct match)
//   flame_value    <- NO COLUMN EXISTS. This file only has a binary flame flag, never
//                      a continuous 0..1023 reading. The numbers below are NOT measured;
//                      they're a bimodal estimate (0 when flame=0, 1023 when flame=1)
//                      using the real flame positive rate (5.88%) from this data:
//                        mean = 1023 * p, std = 1023 * sqrt(p*(1-p)), p = 0.058819
//                      This feature may contribute little real signal until it's
//                      replaced with stats from actual continuous flame readings.
const float HEAT_FEAT_MEAN[HEAT_NUM_FEATURES] = {34.265178f, 52.520447f};
const float HEAT_FEAT_STD[HEAT_NUM_FEATURES]  = {8.409767f, 15.691554f};
const float FIRE_FEAT_MEAN[FIRE_NUM_FEATURES] = {158.789592f, 281.435486f, 0.058819f, 60.172292f, 34.265178f, 52.520447f};
const float FIRE_FEAT_STD[FIRE_NUM_FEATURES]  = {150.867322f, 206.706549f, 0.235295f, 240.698047f, 8.409767f, 15.691554f};

// ---------------- Fire model inputs ----------------
// Feature order (from fire_model.h): mq2_smoke, mq2_gas, flame_detected, flame_value, temperature_C, humidity_pct
//
// <<< CHECK >>> Calibration. The training data used these scales; your hardware must be mapped to them:
//  - One MQ2 can't separate "smoke" from "gas", so the same reading is fed to both inputs (approximation).
//  - MQ2_SCALE multiplies the 12-bit ADC value. If you use a 10k/20k divider on AO (reads x0.667), use 1.5.
//    Adjust so clean air reads low (the model treats ~200 as calm and ~1500+ as smoky).
//  - flame_value: training data is high when a flame is present (roughly 0..1023). Many ESP32 flame
//    modules read LOW on flame, so the AO reading is inverted and scaled between these two values.
const float MQ2_SCALE          = 1.0f;
const int   FLAME_AO_NO_FLAME  = 4000;   // raw ADC with no flame
const int   FLAME_AO_STRONG    = 500;    // raw ADC with a strong flame close by

float flameValueFromRaw(int raw) {
  float f = (float)(FLAME_AO_NO_FLAME - raw) / (FLAME_AO_NO_FLAME - FLAME_AO_STRONG) * 1023.0f;
  return constrain(f, 0.0f, 1023.0f);
}

// Uniform labels so all nodes share one schema.
// (heat: Normal/Caution/Danger/Extreme Danger. fire: Normal/Watch/Warning/Danger. Same 0-3 order.)
const char* LABELS[4] = {"LOW", "MEDIUM", "HIGH", "CRITICAL"};

// ---------------- Globals ----------------
DHT dht(DHTPIN, DHTTYPE);
static tflite::AllOpsResolver resolver;
WebServer server(80);
char latestJson[1024] = "{\"status\":\"starting\"}";

struct MLModel {
  const unsigned char* data;
  int nFeat, winLen;
  const float* mean;
  const float* sd;
  uint8_t* arena;
  tflite::MicroInterpreter* interp;
  TfLiteTensor* in;
  TfLiteTensor* out;
  float win[MAX_WIN][MAX_FEAT];
  int count;
  bool ok;
};

alignas(16) static uint8_t heatArena[ARENA_SIZE];
alignas(16) static uint8_t fireArena[ARENA_SIZE];
MLModel heat, fire;

float lastTemp = NAN, lastHum = NAN;
unsigned long nextTick = 0;
bool apOn = false;                 // fallback hotspot running
unsigned long lastPost = 0, lastWifiTry = 0;
int lastPostedAlert = -1;          // alert level in the last packet sent to the Pi

// ---------------- Output ----------------
void sendLine(const char* s) { Serial.println(s); }   // add LoRa.print(s) here later if needed

void sendError(const char* who, const char* msg) {
  char b[160];
  snprintf(b, sizeof(b), "{\"error\":\"%s: %s\"}", who, msg);
  sendLine(b);
}

void num(char* out, size_t n, float v, int dec) {
  if (isnan(v)) snprintf(out, n, "null");
  else snprintf(out, n, "%.*f", dec, v);
}

// POST to the Raspberry Pi. Short timeout so a slow/unreachable Pi can't stall sensor
// sampling or the LED/buzzer alerts for long; the Pi still saves a reading it answers late.
// Prints the HTTP result on Serial only when it changes
// (200 = saved, 400 = rejected - the Pi's ESP Live page shows why, -11 = slow answer, still saved).
void postToPi(const char* json) {
  if (WiFi.status() != WL_CONNECTED) return;   // not on the Pi's network right now
  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(PI_TIMEOUT_MS);
  http.setTimeout(PI_TIMEOUT_MS);
  if (!http.begin(client, PI_URL)) return;
  http.addHeader("Content-Type", "application/json");
  int code = http.POST((uint8_t*)json, strlen(json));
  http.end();

  static int lastCode = 0;
  if (code != lastCode) {
    lastCode = code;
    char b[96];
    snprintf(b, sizeof(b), "{\"event\":\"pi_post\",\"http\":%d}", code);
    sendLine(b);
  }
}

// ---------------- ML ----------------
const char* modelSetup(MLModel& m, const unsigned char* data, int nFeat, int winLen,
                       const float* mean, const float* sd, uint8_t* arena) {
  m.data = data; m.nFeat = nFeat; m.winLen = winLen;
  m.mean = mean; m.sd = sd; m.arena = arena; m.count = 0; m.ok = false;

  const tflite::Model* model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return "schema version mismatch";

  m.interp = new tflite::MicroInterpreter(model, resolver, arena, ARENA_SIZE);
  if (m.interp->AllocateTensors() != kTfLiteOk) return "AllocateTensors failed (raise ARENA_SIZE)";

  m.in  = m.interp->input(0);
  m.out = m.interp->output(0);
  if (m.in->type != kTfLiteInt8 || m.out->type != kTfLiteInt8) return "expected int8 input/output";
  if ((int)m.in->bytes != winLen * nFeat) return "input size != window x features";

  m.ok = true;
  return nullptr;
}

void pushWindow(MLModel& m, const float* feat) {
  if (!m.ok) return;
  if (m.count == 0) {
    // Pad the whole window with this first reading so predict() can run right away,
    // instead of waiting winLen real samples to fill it. Trade-off: predictions made
    // before the window is full of real data reflect a flat/constant history, not an
    // actual trend -- they firm up as real samples slide in over the next winLen ticks.
    for (int t = 0; t < m.winLen; t++) memcpy(m.win[t], feat, m.nFeat * sizeof(float));
    m.count = m.winLen;
  } else {
    memmove(m.win[0], m.win[1], (m.winLen - 1) * sizeof(m.win[0]));
    memcpy(m.win[m.winLen - 1], feat, m.nFeat * sizeof(float));
  }
}

bool predict(MLModel& m, int& cls, float& conf, float* probs) {
  if (!m.ok || m.count < m.winLen) return false;

  int8_t* dst = m.in->data.int8;
  const float sc = m.in->params.scale;
  const int zp = m.in->params.zero_point;

  for (int t = 0; t < m.winLen; t++) {
    for (int f = 0; f < m.nFeat; f++) {
      float z = (m.win[t][f] - m.mean[f]) / m.sd[f];
      int q = (int)lroundf(z / sc) + zp;
      dst[t * m.nFeat + f] = (int8_t)constrain(q, -128, 127);
    }
  }

  if (m.interp->Invoke() != kTfLiteOk) return false;

  float best = -1e9f;
  cls = 0;
  for (int i = 0; i < 4; i++) {
    float p = (m.out->data.int8[i] - m.out->params.zero_point) * m.out->params.scale;
    probs[i] = constrain(p, 0.0f, 1.0f);
    if (p > best) { best = p; cls = i; }
  }
  conf = constrain(best, 0.0f, 1.0f);
  return true;
}

// ---------------- LED + buzzer alerts ----------------
//  Alert level = the worse of the fire and heat risks (whichever have a prediction so far)
//  Warm-up (windows filling) : green slow blink, buzzer off
//  LOW                       : green solid, buzzer off
//  MEDIUM                    : yellow solid, one short beep when it starts
//  HIGH                      : red solid, 200 ms beep every second
//  CRITICAL                  : red fast blink, buzzer beeps in step with the LED
//  Fault (model failed)      : yellow fast blink, buzzer off
AlertState alertState = AL_WARMUP;
unsigned long beepUntil = 0;

void setLeds(bool g, bool y, bool r) {
  digitalWrite(LED_GREEN_PIN, g);
  digitalWrite(LED_YELLOW_PIN, y);
  digitalWrite(LED_RED_PIN, r);
}

void buzz(bool on) {
  digitalWrite(BUZZER_PIN, (on == (bool)BUZZER_ACTIVE_HIGH) ? HIGH : LOW);
}

void alertInit() {
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_YELLOW_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  setLeds(false, false, false);
  buzz(false);
}

// Wiring check at boot: green, yellow, red in turn, then one short beep
void bootSelfTest() {
  setLeds(true, false, false);  delay(300);
  setLeds(false, true, false);  delay(300);
  setLeds(false, false, true);  delay(300);
  setLeds(false, false, false);
  buzz(true); delay(150); buzz(false);
}

void setAlert(AlertState a) {
  if (a == alertState) return;
  alertState = a;
  if (a == AL_MEDIUM) beepUntil = millis() + 150;     // single short beep on entry
}

const char* ALERT_NAMES[6] = {"WARMUP", "LOW", "MEDIUM", "HIGH", "CRITICAL", "FAULT"};

bool instantActive = false;
unsigned long flameSince = 0, flameLastSeen = 0;

// Checked on every loop pass (works even if the models failed to load)
void updateInstantAlert() {
#if INSTANT_FLAME_ALERT
  unsigned long now = millis();
  bool seen = (digitalRead(FLAME_DO_PIN) == (FLAME_DO_ACTIVE_LOW ? LOW : HIGH));

  if (seen) {
    if (flameSince == 0) flameSince = now;
    flameLastSeen = now;
    if (!instantActive && now - flameSince >= FLAME_CONFIRM_MS) {
      instantActive = true;
      char b[128];
      snprintf(b, sizeof(b),
        "{\"event\":\"instant_alert\",\"source\":\"flame\",\"nodeId\":\"%s\",\"timestamp\":%lu}",
        NODE_ID, now / 1000UL);
      sendLine(b);
      nextTick = now;               // publish a fresh JSON reading immediately
    }
  } else {
    flameSince = 0;
    if (instantActive && now - flameLastSeen >= FLAME_HOLD_MS) instantActive = false;
  }
#endif
}

// Call every loop pass: non-blocking, driven by millis()
void updateAlerts() {
  updateInstantAlert();
  unsigned long now = millis();
  bool blinkSlow = ((now / BLINK_SLOW_MS) % 2) == 0;
  bool blinkFast = ((now / BLINK_FAST_MS) % 2) == 0;

  switch (instantActive ? AL_CRITICAL : alertState) {     // instant alert overrides the model result
    case AL_WARMUP:   setLeds(blinkSlow, false, false); buzz(false); break;
    case AL_LOW:      setLeds(true, false, false);      buzz(false); break;
    case AL_MEDIUM:   setLeds(false, true, false);      buzz(now < beepUntil); break;
    case AL_HIGH:     setLeds(false, false, true);      buzz((now % 1000) < 200); break;
    case AL_CRITICAL: setLeds(false, false, blinkFast); buzz(blinkFast); break;
    case AL_FAULT:    setLeds(false, blinkFast, false); buzz(false); break;
  }
}

// ---------------- WiFi + JSON endpoint ----------------
void handleJson() {
  server.sendHeader("Cache-Control", "no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.sendHeader("Refresh", "2");   // browsers reload every 2 s; apps/scripts ignore it
  server.send(200, "application/json", latestJson);
}

void connectWiFi() {
  WiFi.mode(WIFI_AP_STA);            // station (to the Pi) + room for the fallback hotspot
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) {
    updateAlerts();
    delay(20);
  }

  char b[200];
  if (WiFi.status() == WL_CONNECTED) {
    WiFi.mode(WIFI_STA);
    snprintf(b, sizeof(b), "{\"event\":\"wifi\",\"mode\":\"STA\",\"ip\":\"%s\"}",
             WiFi.localIP().toString().c_str());
  } else {
    // Pi not reachable yet (e.g. it is still booting): own hotspot now, keep retrying the Pi
    WiFi.softAP(AP_SSID, AP_PASS);
    apOn = true;
    snprintf(b, sizeof(b), "{\"event\":\"wifi\",\"mode\":\"AP\",\"ssid\":\"%s\",\"ip\":\"%s\",\"retrying\":\"%s\"}",
             AP_SSID, WiFi.softAPIP().toString().c_str(), WIFI_SSID);
  }
  sendLine(b);

  MDNS.begin("node2");                 // http://node2.local
  server.on("/", handleJson);
  server.on("/data", handleJson);
  server.begin();
}

// Call every loop pass: reconnects to the Pi's hotspot and drops the fallback hotspot once connected
void keepWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (apOn) {
      WiFi.softAPdisconnect(true);
      WiFi.mode(WIFI_STA);
      apOn = false;
      char b[160];
      snprintf(b, sizeof(b), "{\"event\":\"wifi\",\"mode\":\"STA\",\"ip\":\"%s\"}",
               WiFi.localIP().toString().c_str());
      sendLine(b);
    }
    return;
  }
  if (millis() - lastWifiTry >= 10000) {
    lastWifiTry = millis();
    WiFi.reconnect();
  }
}

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);

  analogReadResolution(12);
  pinMode(FLAME_DO_PIN, INPUT);
  dht.begin();
  alertInit();
  bootSelfTest();

  const char* e;
  e = modelSetup(heat, HEAT_MODEL_DATA, HEAT_NUM_FEATURES, HEAT_WINDOW_LEN,
                 HEAT_FEAT_MEAN, HEAT_FEAT_STD, heatArena);
  if (e) sendError("heat model", e);

  e = modelSetup(fire, FIRE_MODEL_DATA, FIRE_NUM_FEATURES, FIRE_WINDOW_LEN,
                 FIRE_FEAT_MEAN, FIRE_FEAT_STD, fireArena);
  if (e) sendError("fire model", e);

  char b[160];
  snprintf(b, sizeof(b), "{\"event\":\"ready\",\"nodeId\":\"%s\",\"heat\":%s,\"fire\":%s}",
           NODE_ID, heat.ok ? "true" : "false", fire.ok ? "true" : "false");
  sendLine(b);

  alertState = (!heat.ok || !fire.ok) ? AL_FAULT : AL_WARMUP;

  connectWiFi();
  nextTick = millis();
}

// ---------------- Loop ----------------
void loop() {
  server.handleClient();
  keepWiFi();
  updateAlerts();
  if ((long)(millis() - nextTick) < 0) return;
  nextTick += SAMPLE_MS;

  // --- read sensors (keep last good DHT value if a read fails) ---
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h)) lastHum = h;
  if (!isnan(t)) lastTemp = t;

  int smokeRaw = analogRead(SMOKE_PIN);
  int flameRaw = analogRead(FLAME_PIN);
  int flameDet = (digitalRead(FLAME_DO_PIN) == (FLAME_DO_ACTIVE_LOW ? LOW : HIGH)) ? 1 : 0;
  float mq2 = smokeRaw * MQ2_SCALE;
  float flameValue = flameValueFromRaw(flameRaw);

  // --- ML ---
  if (!isnan(lastTemp) && !isnan(lastHum)) {
    float hf[2] = {lastTemp, lastHum};
    pushWindow(heat, hf);
    float ff[6] = {mq2, mq2, (float)flameDet, flameValue, lastTemp, lastHum};
    pushWindow(fire, ff);
  }

  int hCls = 0, fCls = 0; float hConf = 0, fConf = 0;
  float hP[4] = {0}, fP[4] = {0};
  bool hOk = predict(heat, hCls, hConf, hP);
  bool fOk = predict(fire, fCls, fConf, fP);

  // --- alert state ---
  if (!heat.ok || !fire.ok) {
    setAlert(AL_FAULT);
  } else {
    int worst = -1;
    if (hOk && hCls > worst) worst = hCls;
    if (fOk && fCls > worst) worst = fCls;
    setAlert(worst < 0 ? AL_WARMUP : (AlertState)(AL_LOW + worst));
  }

  // --- build JSON ---
  // "levels" = probability of LOW, MEDIUM, HIGH, CRITICAL (the Pi turns it into a 0..1 hazard score)
  char heatJ[160], fireJ[160];
  if (hOk) snprintf(heatJ, sizeof(heatJ), "{\"percentage\":%d,\"label\":\"%s\",\"levels\":[%.3f,%.3f,%.3f,%.3f]}",
                    (int)lroundf(hConf * 100), LABELS[hCls], hP[0], hP[1], hP[2], hP[3]);
  else     snprintf(heatJ, sizeof(heatJ), "{\"percentage\":null,\"label\":null}");

  if (fOk) snprintf(fireJ, sizeof(fireJ), "{\"percentage\":%d,\"label\":\"%s\",\"levels\":[%.3f,%.3f,%.3f,%.3f]}",
                    (int)lroundf(fConf * 100), LABELS[fCls], fP[0], fP[1], fP[2], fP[3]);
  else     snprintf(fireJ, sizeof(fireJ), "{\"percentage\":null,\"label\":null}");

  char sTemp[16], sHum[16];
  num(sTemp, sizeof(sTemp), lastTemp, 0);
  num(sHum,  sizeof(sHum),  lastHum,  0);

  char buf[1024];
  snprintf(buf, sizeof(buf),
    "{\"nodeId\":\"%s\",\"timestamp\":%lu,\"rssi\":%d,\"sensors\":{"
      "\"temperature\":{\"value\":%s,\"unit\":\"\\u00b0C\"},"
      "\"humidity\":{\"value\":%s,\"unit\":\"%%\"},"
      "\"smoke\":{\"value\":%d,\"unit\":\"raw_adc\"},"
      "\"flame\":{\"value\":%d,\"unit\":\"raw_adc\"},"
      "\"flameDetected\":{\"value\":%d,\"unit\":\"bool\"}},"
    "\"alert\":{\"level\":\"%s\",\"instant\":%s},"
    "\"risk\":{\"fire\":%s,\"heat\":%s}}",
    NODE_ID, millis() / 1000UL, (int)(WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0),
    sTemp, sHum, smokeRaw, flameRaw, flameDet,
    ALERT_NAMES[instantActive ? AL_CRITICAL : alertState], instantActive ? "true" : "false",
    fireJ, heatJ);

  snprintf(latestJson, sizeof(latestJson), "%s", buf);   // served to WiFi clients

  // send to the Raspberry Pi every POST_MS, and at once when the alert level changes
  int alertNow = instantActive ? AL_CRITICAL : alertState;
  if (millis() - lastPost >= POST_MS || alertNow != lastPostedAlert) {
    lastPost = millis();
    lastPostedAlert = alertNow;
    postToPi(buf);
  }
#if SERIAL_JSON
  sendLine(buf);
#endif
}
