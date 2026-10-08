// Translates the JSON the team's ESP32-S3 sketches send (SIH.ino = NODE_01,
// NODE_2.ino = NODE_02) into the flat format the rest of the server uses.
//
// Sketch format (values are {value, unit} objects, or plain numbers):
//   {"nodeId":"NODE_01","timestamp":123,"rssi":-60,
//    "sensors":{"waterLevel":{"value":45.2,"unit":"cm"},"rain":{"value":3900,"unit":"raw_adc"},
//               "rainfall":{"value":1.6,"unit":"mm"},"soilMoisture":{"value":60}, ...},
//    "risk":{"flood":{"percentage":91,"label":"LOW","levels":[0.91,0.06,0.02,0.01]}, ...}}
//
// Flat format: {"node_id":"NODE_01","water_level":45.2,"rain":false,"rainfall_mm_h":1.6, ...}
// A flat packet (like the curl test command) passes through unchanged.

const RAIN_ON_MM = 2.5;                       // rainfall at/above this counts as "raining" (light rain)
const RAIN_DRY_ADC = 4095, RAIN_WET_ADC = 1000, RAIN_MAX_MM = 50;  // same constants as SIH.ino
const ADC_MAX = 4095;

// on-node TinyML outputs -> the ML model's TinyML input columns
const RISK_COLUMNS = {
  flood: 'node1_flood',
  landslide: 'node1_landslide',
  fire: 'node2_wildfire',
  wildfire: 'node2_wildfire',
  heat: 'node2_extreme_heat',
  extreme_heat: 'node2_extreme_heat',
};
const LEVELS = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];

function isSketchFormat(body) {
  return Boolean(body) && typeof body === 'object' && !Array.isArray(body)
    && body.node_id === undefined
    && (body.nodeId !== undefined || (body.sensors && typeof body.sensors === 'object'));
}

// {value: x} -> x, plain number -> number, null/missing -> undefined
function val(entry) {
  const v = entry && typeof entry === 'object' && !Array.isArray(entry) ? entry.value : entry;
  if (v === null || v === undefined || v === '') return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : v;          // non-numbers are kept so validation reports them
}

function setIf(out, key, value) {
  if (value !== undefined) out[key] = value;
}

// The node runs a 4-class model (LOW..CRITICAL). The Pi model was trained on a
// 0..1 "chance of hazard" score with label = score >= 0.6, so the score is
// P(HIGH) + P(CRITICAL). Without the full probabilities, it is estimated from
// the top class and its confidence.
function tinymlScore(risk) {
  if (!risk || typeof risk !== 'object') return undefined;
  if (Array.isArray(risk.levels) && risk.levels.length === 4 && risk.levels.every((p) => Number.isFinite(Number(p)))) {
    return Math.min(Math.max(Number(risk.levels[2]) + Number(risk.levels[3]), 0), 1);
  }
  const level = LEVELS.indexOf(String(risk.label || '').toUpperCase());
  const conf = Number(risk.percentage) / 100;
  if (level < 0 || !Number.isFinite(conf)) return undefined;
  return level >= 2 ? conf : Math.max(0, 1 - conf);
}

function fromSketch(body) {
  const s = (body.sensors && typeof body.sensors === 'object') ? body.sensors : {};
  const out = { node_id: typeof body.nodeId === 'string' ? body.nodeId : body.nodeId };

  // NODE_01
  setIf(out, 'water_level', val(s.waterLevel));
  setIf(out, 'soil_moisture', val(s.soilMoisture));
  const rainRaw = val(s.rain);
  let rainfall = val(s.rainfall);
  if (rainfall === undefined && typeof rainRaw === 'number') {
    const wet = (RAIN_DRY_ADC - rainRaw) / (RAIN_DRY_ADC - RAIN_WET_ADC);
    rainfall = Math.round(Math.min(Math.max(wet, 0), 1) * RAIN_MAX_MM * 10) / 10;
  }
  setIf(out, 'rain_raw', rainRaw);
  setIf(out, 'rainfall_mm_h', rainfall);
  if (typeof rainfall === 'number') out.rain = rainfall >= RAIN_ON_MM;
  const tiltX = val(s.tiltX);
  const tiltY = val(s.tiltY);
  setIf(out, 'tilt_x_deg', tiltX);
  setIf(out, 'tilt_y_deg', tiltY);
  if (typeof tiltX === 'number' || typeof tiltY === 'number') {
    // the dashboard's single tilt = the axis leaning the most (sign kept)
    const x = typeof tiltX === 'number' ? tiltX : 0;
    const y = typeof tiltY === 'number' ? tiltY : 0;
    out.tilt = Math.abs(x) >= Math.abs(y) ? x : y;
  }
  setIf(out, 'acceleration_g', val(s.acceleration));

  // both nodes
  setIf(out, 'temperature', val(s.temperature));
  setIf(out, 'humidity', val(s.humidity));

  // NODE_02: smoke is the raw 12-bit ADC value (0..4095); the dashboard rules use 0..1000
  const smokeRaw = val(s.smoke);
  if (typeof smokeRaw === 'number') {
    out.smoke_raw = smokeRaw;
    out.smoke = Math.round(Math.min(Math.max(smokeRaw, 0), ADC_MAX) * 1000 / ADC_MAX);
  } else setIf(out, 'smoke', smokeRaw);
  setIf(out, 'gas_raw', val(s.gas));
  setIf(out, 'flame_raw', val(s.flame));
  const flameSeen = val(s.flameDetected);
  const instant = body.alert && typeof body.alert === 'object' && body.alert.instant === true;
  if (flameSeen !== undefined || instant) out.flame = instant || flameSeen === 1 || flameSeen === true;

  setIf(out, 'signal_strength', val(body.rssi));
  if (typeof body.timestamp === 'number') out.device_uptime_s = body.timestamp;  // node uptime, not a date

  // on-node TinyML results
  if (body.risk && typeof body.risk === 'object') {
    Object.entries(body.risk).forEach(([name, risk]) => {
      const column = RISK_COLUMNS[name.toLowerCase()];
      const score = column ? tinymlScore(risk) : undefined;
      if (score === undefined) return;
      out[`${column}_score`] = Math.round(score * 1000) / 1000;
      out[`${column}_label`] = score >= 0.6 ? 1 : 0;
    });
  }
  return out;
}

// Returns the flat payload; flat payloads are returned unchanged.
function toFlat(body) {
  return isSketchFormat(body) ? fromSketch(body) : body;
}

module.exports = { toFlat, isSketchFormat, tinymlScore };
