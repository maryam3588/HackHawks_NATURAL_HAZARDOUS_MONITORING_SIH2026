/*
  NODE_01 - Flood + Landslide
  ESP32-S3 | sensors -> on-device TFLite Micro models -> one JSON line per sample

  Output: JSON only, no web page.
    - Raspberry Pi: joins the Pi's hotspot (bash hotspot.sh on -> HAZARD-NET) and POSTs the
      JSON to http://192.168.4.1:3000/api/sensor-data every POST_MS. Watch it on the Pi's
      ESP Live page: http://192.168.4.1:3000/esp
    - WiFi: any phone/laptop on the same network opens  http://<node-ip>/  (or /data)
      and gets the latest reading as JSON. CORS is enabled so other apps can fetch it.
      If the Pi's hotspot is unreachable the node starts its own hotspot (see AP_SSID below)
      and keeps retrying the Pi in the background; the hotspot turns off once it connects.
    - Serial (115200): the same JSON, one line per sample (set SERIAL_JSON 0 to disable).
    - Lines with "event" or "error" (Serial only) are status messages; skip them in a parser.

  BEFORE YOU UPLOAD - check the items marked  <<< CHECK >>>
*/

#include <DHT.h>
#include <Wire.h>
#include <math.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ESPmDNS.h>
#include <HTTPClient.h>
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#if __has_include("flood_model.h")
  #include "flood_model.h"
  #define HAVE_FLOOD_MODEL 1
#else
  #define HAVE_FLOOD_MODEL 0      // flood_model.h not in the sketch folder -> flood risk stays null
#endif
#if __has_include("landslide_model.h")
  #include "landslide_model.h"
  #define HAVE_SLIDE_MODEL 1
#else
  #define HAVE_SLIDE_MODEL 0      // landslide_model.h not in the sketch folder -> landslide risk stays null
#endif

// Array names used by the generated flood_model.h / landslide_model.h
#define FLOOD_MODEL_DATA      flood_model_data
#define LANDSLIDE_MODEL_DATA  landslide_model_data

// ---------------- Config ----------------
#define NODE_ID        "NODE_01"
#define SAMPLE_MS      2000     // <<< CHECK >>> must match the time step used to train the models
#define WINDOW         20       // model window (steps)
#define ARENA_SIZE     (48 * 1024)

#define USE_MPU        0        // set to 1 once a working MPU6050 (GY-521) is wired
#define USE_ULTRASONIC 0        // set to 1 if water level comes from a JSN-SR04T
#define SERIAL_JSON    1        // also print each JSON line on Serial

// ---------------- WiFi ----------------
const char* WIFI_SSID = "HAZARD-NET";        // the Pi's hotspot (bash hotspot.sh on), 2.4 GHz
const char* WIFI_PASS = "hazard1234";        // must match HOTSPOT_PASSWORD on the Pi
const char* AP_SSID   = "NODE01-JSON";           // fallback hotspot
const char* AP_PASS   = "12345678";              // min 8 characters

// ---------------- Raspberry Pi ----------------
const char* PI_URL   = "http://192.168.4.1:3000/api/sensor-data";   // the Pi on its own hotspot
#define POST_MS         5000    // send the latest reading to the Pi every 5 s (sampling stays SAMPLE_MS)
#define PI_TIMEOUT_MS   800     // the Pi saves the reading even if its answer comes later than this

// ---------------- Pins ----------------
#define DHTPIN 4
#define DHTTYPE DHT11
#define SOIL_PIN  1
#define WATER_PIN 2
#define RAIN_PIN  3
#define TRIG_PIN  5             // JSN-SR04T (only if USE_ULTRASONIC)
#define ECHO_PIN  6             // ECHO is 5V: use a voltage divider to 3.3V!
#define SDA_PIN   8
#define SCL_PIN   9
#define MPU_ADDR  0x68

// ---------------- Calibration ----------------
// <<< CHECK >>> these are placeholders. They MUST reproduce the units used in the training data.
const int   SOIL_DRY_ADC = 3000, SOIL_WET_ADC = 1200;               // -> 0..100 %
const int   RAIN_DRY_ADC = 4095, RAIN_WET_ADC = 1000;               // lower ADC = wetter
const float RAIN_MAX_MM  = 50.0f;                                    // mm at fully wet
const int   WATER_EMPTY_ADC = 0, WATER_FULL_ADC = 2500;              // analog level sensor
const float WATER_MAX_CM = 100.0f;                                   // cm at full
const float SENSOR_HEIGHT_CM = 100.0f;   // JSN-SR04T only: sensor height above the empty riverbed/tank
// NOTE: the flood model reads water level as a level that RISES with flooding
// (training mean 139 cm, std 211 cm). Capping at 100 cm limits how high the risk can go.

// ---------------- Training statistics (from your spec) ----------------
const float FLOOD_MEAN[2] = {139.12f, 5.01f};
const float FLOOD_STD[2]  = {210.99f, 14.35f};

const float SLIDE_MEAN[10] = {5.01f, 57.09f, 0.00007f, 0.00026f, 1.0002f,
                              0.0033f, -0.0054f, 0.0107f, 0.997f, 0.597f};
const float SLIDE_STD[10]  = {14.35f, 34.59f, 0.0886f, 0.0862f, 0.0843f,
                              2.241f, 2.245f, 2.236f, 1.373f, 0.855f};
// feature order: rain_mm, soil_pct, ax, ay, az, gx, gy, gz, tilt_x, tilt_y

const char* LABELS[4] = {"LOW", "MEDIUM", "HIGH", "CRITICAL"};

// ---------------- Globals ----------------
DHT dht(DHTPIN, DHTTYPE);
static tflite::AllOpsResolver resolver;
WebServer server(80);
char latestJson[1024] = "{\"status\":\"starting\"}";

struct MLModel {
  const char* name;
  const unsigned char* data;
  int nFeat;
  const float* mean;
  const float* sd;
  uint8_t* arena;
  tflite::MicroInterpreter* interp;
  TfLiteTensor* in;
  TfLiteTensor* out;
  float win[WINDOW][10];
  int count;
  bool ok;
};

alignas(16) static uint8_t floodArena[ARENA_SIZE];
alignas(16) static uint8_t slideArena[ARENA_SIZE];
MLModel flood, slide;

bool mpuOk = false;
bool apOn = false;                 // fallback hotspot running
unsigned long lastPost = 0, lastWifiTry = 0;
float lastTemp = NAN, lastHum = NAN, lastWater = NAN;
unsigned long nextTick = 0;

// ---------------- Output ----------------
void sendLine(const char* s) {
  Serial.println(s);          // add LoRa.print(s) here later if you need radio output
}

void sendError(const char* who, const char* msg) {
  char b[160];
  snprintf(b, sizeof(b), "{\"error\":\"%s: %s\"}", who, msg);
  sendLine(b);
}

void num(char* out, size_t n, float v, int dec) {
  if (isnan(v)) snprintf(out, n, "null");
  else snprintf(out, n, "%.*f", dec, v);
}

// ---------------- Sensors ----------------
float readWaterCm() {
#if USE_ULTRASONIC
  digitalWrite(TRIG_PIN, LOW);  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  unsigned long d = pulseIn(ECHO_PIN, HIGH, 30000);
  if (d == 0) return NAN;
  float distance = d * 0.0343f / 2.0f;              // sensor -> water surface
  return constrain(SENSOR_HEIGHT_CM - distance, 0.0f, WATER_MAX_CM);   // water level = height - distance
#else
  int raw = analogRead(WATER_PIN);
  float cm = (float)(raw - WATER_EMPTY_ADC) * WATER_MAX_CM / (WATER_FULL_ADC - WATER_EMPTY_ADC);
  return constrain(cm, 0.0f, WATER_MAX_CM);
#endif
}

float rainMmFromRaw(int raw) {
  float f = (float)(RAIN_DRY_ADC - raw) / (RAIN_DRY_ADC - RAIN_WET_ADC);
  return constrain(f, 0.0f, 1.0f) * RAIN_MAX_MM;
}

float soilPctFromRaw(int raw) {
  float f = (float)(SOIL_DRY_ADC - raw) / (SOIL_DRY_ADC - SOIL_WET_ADC) * 100.0f;
  return constrain(f, 0.0f, 100.0f);
}

#if USE_MPU
bool mpuInit() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);            // PWR_MGMT_1: wake up
  Wire.write(0x00);
  return Wire.endTransmission() == 0;
}

bool mpuRead(float& ax, float& ay, float& az, float& gx, float& gy, float& gz) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);            // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    v[i] = (int16_t)((hi << 8) | lo);
  }
  ax = v[0] / 16384.0f; ay = v[1] / 16384.0f; az = v[2] / 16384.0f;   // +-2 g
  gx = v[4] / 131.0f;   gy = v[5] / 131.0f;   gz = v[6] / 131.0f;     // +-250 deg/s
  return true;
}
#endif

// ---------------- ML ----------------
const char* modelSetup(MLModel& m, const char* name, const unsigned char* data,
                       int nFeat, const float* mean, const float* sd, uint8_t* arena) {
  m.name = name; m.data = data; m.nFeat = nFeat; m.mean = mean; m.sd = sd;
  m.arena = arena; m.count = 0; m.ok = false;

  const tflite::Model* model = tflite::GetModel(data);
  if (model->version() != TFLITE_SCHEMA_VERSION) return "schema version mismatch";

  m.interp = new tflite::MicroInterpreter(model, resolver, arena, ARENA_SIZE);
  if (m.interp->AllocateTensors() != kTfLiteOk) return "AllocateTensors failed (raise ARENA_SIZE)";

  m.in  = m.interp->input(0);
  m.out = m.interp->output(0);
  if (m.in->type != kTfLiteInt8 || m.out->type != kTfLiteInt8) return "expected int8 input/output";
  if ((int)m.in->bytes != WINDOW * nFeat) return "input size != WINDOW x features";

  m.ok = true;
  return nullptr;
}

void pushWindow(MLModel& m, const float* feat) {
  if (!m.ok) return;
  if (m.count < WINDOW) {
    memcpy(m.win[m.count++], feat, m.nFeat * sizeof(float));
  } else {
    memmove(m.win[0], m.win[1], (WINDOW - 1) * sizeof(m.win[0]));
    memcpy(m.win[WINDOW - 1], feat, m.nFeat * sizeof(float));
  }
}

// returns true and fills cls/conf (and the 4 class probabilities) when the window is full and inference succeeded
bool predict(MLModel& m, int& cls, float& conf, float* probs) {
  if (!m.ok || m.count < WINDOW) return false;

  int8_t* dst = m.in->data.int8;
  const float sc = m.in->params.scale;
  const int zp = m.in->params.zero_point;

  for (int t = 0; t < WINDOW; t++) {
    for (int f = 0; f < m.nFeat; f++) {
      float z = (m.win[t][f] - m.mean[f]) / m.sd[f];
      int q = (int)lroundf(z / sc) + zp;
      dst[t * m.nFeat + f] = (int8_t)constrain(q, -128, 127);
    }
  }

  if (m.interp->Invoke() != kTfLiteOk) return false;

  const int nCls = 4;
  float best = -1e9f;
  cls = 0;
  for (int i = 0; i < nCls; i++) {
    float p = (m.out->data.int8[i] - m.out->params.zero_point) * m.out->params.scale;
    probs[i] = constrain(p, 0.0f, 1.0f);
    if (p > best) { best = p; cls = i; }
  }
  conf = constrain(best, 0.0f, 1.0f);
  return true;
}

// ---------------- WiFi + JSON endpoint ----------------
void handleJson() {
  server.sendHeader("Cache-Control", "no-store");
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(200, "application/json", latestJson);
}

void connectWiFi() {
  WiFi.mode(WIFI_AP_STA);            // station (to the Pi) + room for the fallback hotspot
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 15000) delay(250);

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

  MDNS.begin("node1");                 // http://node1.local
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

// POST one JSON reading to the Pi. Prints the HTTP result on Serial only when it changes
// (200 = saved, 400 = rejected - the Pi's ESP Live page shows why, -11 = slow answer, still saved).
void postToPi(const char* json) {
  if (WiFi.status() != WL_CONNECTED) return;
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

// ---------------- Setup ----------------
void setup() {
  Serial.begin(115200);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 5000) delay(10);

  analogReadResolution(12);
  dht.begin();

#if USE_ULTRASONIC
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
#endif
#if USE_MPU
  mpuOk = mpuInit();
  if (!mpuOk) sendError("mpu", "not found, landslide model running degraded");
#endif

  const char* e;
#if HAVE_FLOOD_MODEL
  e = modelSetup(flood, "flood", FLOOD_MODEL_DATA, 2, FLOOD_MEAN, FLOOD_STD, floodArena);
  if (e) sendError("flood model", e);
#else
  sendError("flood model", "flood_model.h not found, model disabled");
#endif
#if HAVE_SLIDE_MODEL
  e = modelSetup(slide, "landslide", LANDSLIDE_MODEL_DATA, 10, SLIDE_MEAN, SLIDE_STD, slideArena);
  if (e) sendError("landslide model", e);
#else
  sendError("landslide model", "landslide_model.h not found, model disabled");
#endif

  char b[160];
  snprintf(b, sizeof(b),
    "{\"event\":\"ready\",\"nodeId\":\"%s\",\"flood\":%s,\"landslide\":%s,\"mpu\":%s}",
    NODE_ID, flood.ok ? "true" : "false", slide.ok ? "true" : "false", mpuOk ? "true" : "false");
  sendLine(b);

  connectWiFi();
  nextTick = millis();
}

// ---------------- Loop ----------------
void loop() {
  server.handleClient();
  keepWiFi();
  if ((long)(millis() - nextTick) < 0) return;
  nextTick += SAMPLE_MS;

  // --- read sensors (keep last good value if a read fails) ---
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h)) lastHum = h;
  if (!isnan(t)) lastTemp = t;

  float w = readWaterCm();
  if (!isnan(w)) lastWater = w;

  int rainRaw = analogRead(RAIN_PIN);
  int soilRaw = analogRead(SOIL_PIN);
  float rainMm  = rainMmFromRaw(rainRaw);
  float soilPct = soilPctFromRaw(soilRaw);

  // --- motion. Without a live IMU the landslide model is NOT run (see below) ---
  float ax = SLIDE_MEAN[2], ay = SLIDE_MEAN[3], az = SLIDE_MEAN[4];
  float gx = SLIDE_MEAN[5], gy = SLIDE_MEAN[6], gz = SLIDE_MEAN[7];
  float tiltX = SLIDE_MEAN[8], tiltY = SLIDE_MEAN[9];
  float accMag = NAN, tiltXo = NAN, tiltYo = NAN;   // what gets reported
  bool imuLive = false;

#if USE_MPU
  if (mpuOk && mpuRead(ax, ay, az, gx, gy, gz)) {
    tiltX = atan2f(ay, az) * 180.0f / PI;
    tiltY = atan2f(-ax, sqrtf(ay * ay + az * az)) * 180.0f / PI;
    accMag = sqrtf(ax * ax + ay * ay + az * az);
    tiltXo = tiltX; tiltYo = tiltY;
    imuLive = true;
  }
#endif

  // --- ML ---
  float ff[2]  = {isnan(lastWater) ? FLOOD_MEAN[0] : lastWater, rainMm};
  float sf[10] = {rainMm, soilPct, ax, ay, az, gx, gy, gz, tiltX, tiltY};
  pushWindow(flood, ff);
  // The landslide model is driven mainly by tilt/accel/gyro. With no live IMU its output
  // is meaningless (it sits at a fixed MEDIUM), so skip it and restart the window.
  if (imuLive) pushWindow(slide, sf); else slide.count = 0;

  int fCls = 0, sCls = 0; float fConf = 0, sConf = 0;
  float fP[4] = {0}, sP[4] = {0};
  bool fOk = predict(flood, fCls, fConf, fP);
  bool sOk = predict(slide, sCls, sConf, sP);

  // --- build JSON ---
  // "levels" = probability of LOW, MEDIUM, HIGH, CRITICAL (the Pi turns it into a 0..1 hazard score)
  char floodJ[160], slideJ[160];
  if (fOk) snprintf(floodJ, sizeof(floodJ), "{\"percentage\":%d,\"label\":\"%s\",\"levels\":[%.3f,%.3f,%.3f,%.3f]}",
                    (int)lroundf(fConf * 100), LABELS[fCls], fP[0], fP[1], fP[2], fP[3]);
  else     snprintf(floodJ, sizeof(floodJ), "{\"percentage\":null,\"label\":null}");

  if (sOk) snprintf(slideJ, sizeof(slideJ), "{\"percentage\":%d,\"label\":\"%s\",\"levels\":[%.3f,%.3f,%.3f,%.3f]}",
                    (int)lroundf(sConf * 100), LABELS[sCls], sP[0], sP[1], sP[2], sP[3]);
  else     snprintf(slideJ, sizeof(slideJ), "{\"percentage\":null,\"label\":null%s}",
                    imuLive ? "" : ",\"reason\":\"no_imu\"");

  char sWater[16], sSoil[16], sTemp[16], sHum[16], sTx[16], sTy[16], sAcc[16], sRain[16];
  num(sWater, sizeof(sWater), lastWater, 1);
  num(sSoil,  sizeof(sSoil),  soilPct,   0);
  num(sTemp,  sizeof(sTemp),  lastTemp,  0);
  num(sHum,   sizeof(sHum),   lastHum,   0);
  num(sTx,    sizeof(sTx),    tiltXo,    1);
  num(sTy,    sizeof(sTy),    tiltYo,    1);
  num(sAcc,   sizeof(sAcc),   accMag,    2);
  num(sRain,  sizeof(sRain),  rainMm,    1);

  char buf[1024];
  snprintf(buf, sizeof(buf),
    "{\"nodeId\":\"%s\",\"timestamp\":%lu,\"rssi\":%d,\"sensors\":{"
      "\"waterLevel\":{\"value\":%s,\"unit\":\"cm\"},"
      "\"rain\":{\"value\":%d,\"unit\":\"raw_adc\"},"
      "\"rainfall\":{\"value\":%s,\"unit\":\"mm\"},"
      "\"soilMoisture\":{\"value\":%s,\"unit\":\"%%\"},"
      "\"temperature\":{\"value\":%s,\"unit\":\"\\u00b0C\"},"
      "\"humidity\":{\"value\":%s,\"unit\":\"%%\"},"
      "\"tiltX\":{\"value\":%s,\"unit\":\"\\u00b0\"},"
      "\"tiltY\":{\"value\":%s,\"unit\":\"\\u00b0\"},"
      "\"acceleration\":{\"value\":%s,\"unit\":\"g\"}},"
    "\"risk\":{\"flood\":%s,\"landslide\":%s}}",
    NODE_ID, millis() / 1000UL, (int)(WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0),
    sWater, rainRaw, sRain, sSoil, sTemp, sHum, sTx, sTy, sAcc,
    floodJ, slideJ);

  snprintf(latestJson, sizeof(latestJson), "%s", buf);   // served to WiFi clients
  if (millis() - lastPost >= POST_MS) {                  // send to the Raspberry Pi
    lastPost = millis();
    postToPi(buf);
  }
#if SERIAL_JSON
  sendLine(buf);
#endif
}
