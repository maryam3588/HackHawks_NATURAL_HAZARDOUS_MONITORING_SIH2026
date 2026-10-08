/*
  HackHawks Hazard Monitor - ESP32-S3 sensor node

  Reads the node's sensors and POSTs them as JSON to the Raspberry Pi:
      http://<PI_HOST>:<PI_PORT>/api/sensor-data
  Watch the packets arrive on the Pi's "ESP Live" page: http://<PI_HOST>:3000/esp

  Board:    ESP32S3 Dev Module (Arduino IDE, "esp32" boards package by Espressif)
  Library:  "DHT sensor library" by Adafruit (Library Manager) - only if USE_DHT22 is 1
            (it also asks for "Adafruit Unified Sensor"; install both)

  ONE sketch for both nodes: set NODE_ID below.
    NODE_01  flood + landslide: water level, soil moisture, rain, tilt/acceleration, temperature, humidity
    NODE_02  wildfire + heat:   smoke (MQ-2), flame, temperature, humidity

  !! The pin numbers and calibration values are EXAMPLES. Change them to match your wiring and
  !! calibrate each sensor. A sensor that is not connected reads as "missing" and is not sent.
*/

#include <WiFi.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <math.h>

// ============================== SETTINGS ===================================
const char *WIFI_SSID     = "YOUR_WIFI_NAME";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char *PI_HOST       = "192.168.1.42";   // the Pi's IP (setup.sh prints it), or "raspberrypi.local"
const uint16_t PI_PORT    = 3000;
const char *NODE_ID       = "NODE_01";        // "NODE_01" or "NODE_02"

// How often to send. The dashboard updates on every packet; the Pi itself
// passes one reading per node to the ML model every 2 minutes.
const unsigned long SEND_INTERVAL_MS = 5000;

#define USE_DHT22   1   // temperature + humidity (both nodes)
#define USE_MPU6050 1   // tilt + acceleration (NODE_01), I2C, no library needed

// ------------------------------ PINS (examples) ----------------------------
// NODE_01
const int PIN_ULTRASONIC_TRIG = 4;    // JSN-SR04T / HC-SR04 water level
const int PIN_ULTRASONIC_ECHO = 5;    // use a voltage divider if the module runs at 5 V
const int PIN_SOIL_ANALOG     = 6;    // capacitive soil moisture (ADC1 pin)
const int PIN_RAIN_DIGITAL    = 7;    // rain module DO, LOW = wet
const int PIN_I2C_SDA         = 8;    // MPU6050
const int PIN_I2C_SCL         = 9;
// NODE_02
const int PIN_SMOKE_ANALOG    = 1;    // MQ-2 AO through a divider to <= 3.3 V (ADC1 pin)
const int PIN_FLAME_DIGITAL   = 2;    // IR flame module DO, LOW = flame
// both
const int PIN_DHT             = 15;   // DHT22 data

// ------------------------------ CALIBRATION --------------------------------
// Water level = sensor height above the channel bed minus measured distance.
const float SENSOR_HEIGHT_CM  = 100.0;   // the website accepts water_level 0-100 cm
// Soil sensor ADC readings in dry air and in water (measure yours)
const int SOIL_ADC_DRY = 3200;
const int SOIL_ADC_WET = 1300;
// =========================================================================

#if USE_DHT22
#include <DHT.h>
DHT dht(PIN_DHT, DHT22);
#endif

const uint8_t MPU_ADDR = 0x68;
bool mpuOk = false;
bool isNode1() { return strcmp(NODE_ID, "NODE_01") == 0; }

// ------------------------------ SENSORS ------------------------------------
float readWaterLevelCm() {
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  unsigned long echoUs = pulseIn(PIN_ULTRASONIC_ECHO, HIGH, 30000UL);  // 30 ms timeout (~5 m)
  if (echoUs == 0) return NAN;                                         // no echo
  float distanceCm = echoUs * 0.0343f / 2.0f;
  float level = SENSOR_HEIGHT_CM - distanceCm;
  if (level < 0) level = 0;
  if (level > 100) level = 100;
  return level;
}

float readSoilMoisturePct() {
  int raw = analogRead(PIN_SOIL_ANALOG);
  if (raw <= 0 || raw >= 4095) return NAN;                             // floating / shorted pin
  float pct = 100.0f * (SOIL_ADC_DRY - raw) / (float)(SOIL_ADC_DRY - SOIL_ADC_WET);
  return constrain(pct, 0.0f, 100.0f);
}

bool readRain() { return digitalRead(PIN_RAIN_DIGITAL) == LOW; }
bool readFlame() { return digitalRead(PIN_FLAME_DIGITAL) == LOW; }
int readSmokeRaw() { return analogRead(PIN_SMOKE_ANALOG); }            // 0-4095

bool mpuBegin() {
  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);  // PWR_MGMT_1
  Wire.write(0x00);  // wake up
  return Wire.endTransmission() == 0;
}

// Tilt from the accelerometer (degrees) and total acceleration (g)
bool readMpu(float &tiltX, float &tiltY, float &accelG) {
  if (!mpuOk) return false;
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);  // ACCEL_XOUT_H
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, (uint8_t)6) != 6) return false;
  // read bytes one at a time: in "(read() << 8) | read()" C++ may call either read() first
  uint8_t b[6];
  for (int i = 0; i < 6; i++) b[i] = Wire.read();
  int16_t ax = (int16_t)((b[0] << 8) | b[1]);
  int16_t ay = (int16_t)((b[2] << 8) | b[3]);
  int16_t az = (int16_t)((b[4] << 8) | b[5]);
  float x = ax / 16384.0f, y = ay / 16384.0f, z = az / 16384.0f;  // +-2 g range
  tiltX = atan2f(x, sqrtf(y * y + z * z)) * 180.0f / PI;
  tiltY = atan2f(y, sqrtf(x * x + z * z)) * 180.0f / PI;
  accelG = sqrtf(x * x + y * y + z * z);
  return true;
}

// ------------------------------ JSON ---------------------------------------
// Adds "key":value, skipping missing (NaN) values so the Pi treats them as missing
void addNumber(String &json, const char *key, float value, int decimals) {
  if (isnan(value)) return;
  json += ",\"";
  json += key;
  json += "\":";
  json += String(value, decimals);
}

void addBool(String &json, const char *key, bool value) {
  json += ",\"";
  json += key;
  json += "\":";
  json += value ? "true" : "false";
}

String buildPayload() {
  String json = "{\"node_id\":\"";
  json += NODE_ID;
  json += "\"";

  float temperature = NAN, humidity = NAN;
#if USE_DHT22
  temperature = dht.readTemperature();
  humidity = dht.readHumidity();
#endif
  addNumber(json, "temperature", temperature, 1);
  addNumber(json, "humidity", humidity, 0);

  if (isNode1()) {
    addNumber(json, "water_level", readWaterLevelCm(), 1);
    addNumber(json, "soil_moisture", readSoilMoisturePct(), 0);
    addBool(json, "rain", readRain());
    float tiltX, tiltY, accelG;
    if (readMpu(tiltX, tiltY, accelG)) {
      float tilt = constrain(sqrtf(tiltX * tiltX + tiltY * tiltY), -10.0f, 10.0f);  // website accepts -10..10
      addNumber(json, "tilt", tilt, 2);
      addNumber(json, "tilt_x_deg", tiltX, 2);       // extra ML fields
      addNumber(json, "tilt_y_deg", tiltY, 2);
      addNumber(json, "acceleration_g", accelG, 3);
    }
    // A tipping-bucket rain gauge would add: addNumber(json, "rainfall_mm_h", mmPerHour, 1);
  } else {
    int smokeRaw = readSmokeRaw();
    addNumber(json, "smoke", smokeRaw * 1000.0f / 4095.0f, 0);  // website accepts smoke 0-1000
    addNumber(json, "smoke_raw", smokeRaw, 0);                  // raw ADC value for the ML model
    addBool(json, "flame", readFlame());
    // An MQ-135 would add: addNumber(json, "gas_raw", analogRead(PIN_GAS_ANALOG), 0);
  }

  addNumber(json, "signal_strength", WiFi.RSSI(), 0);           // Wi-Fi RSSI in dBm
  json += "}";
  return json;
}

// ------------------------------ NETWORK ------------------------------------
void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.printf("Connecting to Wi-Fi \"%s\"", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWi-Fi connected, ESP32 IP %s, RSSI %d dBm\n", WiFi.localIP().toString().c_str(), WiFi.RSSI());
  } else {
    Serial.println("\nWi-Fi not connected - will retry");
  }
}

void sendReading() {
  connectWiFi();
  if (WiFi.status() != WL_CONNECTED) return;

  String payload = buildPayload();
  String url = String("http://") + PI_HOST + ":" + PI_PORT + "/api/sensor-data";

  HTTPClient http;
  http.setTimeout(5000);
  if (!http.begin(url)) {
    Serial.println("Bad URL: " + url);
    return;
  }
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(payload);
  if (code == 200) {
    Serial.println("Sent OK: " + payload);
  } else if (code > 0) {
    // 400 = the Pi rejected the data; the reason is in the body and on the ESP Live page
    Serial.printf("Pi answered %d: %s\n  sent: %s\n", code, http.getString().c_str(), payload.c_str());
  } else {
    Serial.printf("Could not reach the Pi at %s (%s)\n", url.c_str(), http.errorToString(code).c_str());
  }
  http.end();
}

// ------------------------------ MAIN ---------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.printf("\nHackHawks node %s starting\n", NODE_ID);

  analogReadResolution(12);   // 0-4095
  if (isNode1()) {
    pinMode(PIN_ULTRASONIC_TRIG, OUTPUT);
    pinMode(PIN_ULTRASONIC_ECHO, INPUT);
    pinMode(PIN_RAIN_DIGITAL, INPUT_PULLUP);
#if USE_MPU6050
    mpuOk = mpuBegin();
    Serial.println(mpuOk ? "MPU6050 found" : "MPU6050 not found - tilt will not be sent");
#endif
  } else {
    pinMode(PIN_FLAME_DIGITAL, INPUT_PULLUP);
  }
#if USE_DHT22
  dht.begin();
#endif
  connectWiFi();
}

void loop() {
  static unsigned long lastSend = 0;
  if (lastSend == 0 || millis() - lastSend >= SEND_INTERVAL_MS) {
    lastSend = millis();
    sendReading();
  }
  delay(10);
}
