// ML dataset: every reading used by a "Run ML now" press is appended to
// ml/data/ml_dataset.csv, together with the ML result for its window.
// Columns use the training names (water_level_cm, ...), so the file can be
// uploaded to the Colab notebook (Block 4) after you add label columns such
// as flood_within_15m for the times a disaster really happened.
const fs = require('fs');
const path = require('path');

const DATASET_PATH = path.join(__dirname, '..', 'ml', 'data', 'ml_dataset.csv');

// dashboard field -> training column name
const SENSOR_COLUMNS = [
  ['water_level', 'water_level_cm'],
  ['rainfall_mm_h', 'rainfall_mm_h'],
  ['soil_moisture', 'soil_moisture_pct'],
  ['temperature', 'temperature_c'],
  ['humidity', 'humidity_pct'],
  ['tilt', 'tilt_x_deg'],
  ['tilt_y_deg', 'tilt_y_deg'],
  ['acceleration_g', 'acceleration_g'],
  ['smoke', 'smoke_raw'],
  ['gas_raw', 'gas_raw'],
  ['flame', 'flame'],
  ['rain', 'rain'],
  ['signal_strength', 'signal_strength'],
  ['node1_flood_score', 'node1_flood_score'],
  ['node1_landslide_score', 'node1_landslide_score'],
  ['node1_flood_label', 'node1_flood_label'],
  ['node1_landslide_label', 'node1_landslide_label'],
  ['node2_wildfire_score', 'node2_wildfire_score'],
  ['node2_extreme_heat_score', 'node2_extreme_heat_score'],
  ['node2_wildfire_label', 'node2_wildfire_label'],
  ['node2_extreme_heat_label', 'node2_extreme_heat_label'],
];
const HAZARDS = ['flood', 'landslide', 'wildfire', 'extreme_heat'];

const HEADER = [
  'timestamp', 'node_id', 'site_id', 'reading_id',
  ...SENSOR_COLUMNS.map(([, column]) => column),
  'ml_run_id', 'ml_window_start', 'ml_window_end', 'ml_window_readings',
  ...HAZARDS.flatMap((h) => [`ml_${h}_probability`, `ml_${h}_level`]),
];

function csvValue(value) {
  if (value === null || value === undefined || value === '') return '';
  if (typeof value === 'boolean') return value ? '1' : '0';
  if (value instanceof Date) return value.toISOString();
  const text = String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function readingValue(reading, field) {
  // smoke_raw column: the node's raw MQ-2 ADC value when it sends one, else the dashboard smoke value
  if (field === 'smoke' && reading.extra && reading.extra.smoke_raw !== undefined) return reading.extra.smoke_raw;
  // tilt_x_deg column: the node's real x-axis tilt when it sends one, else the single tilt angle
  if (field === 'tilt' && reading.extra && reading.extra.tilt_x_deg !== undefined) return reading.extra.tilt_x_deg;
  if (reading[field] !== undefined && reading[field] !== null) return reading[field];
  if (reading.extra && reading.extra[field] !== undefined) return reading.extra[field];
  return null;
}

// Writes are chained so two presses at the same time cannot interleave lines
let writeChain = Promise.resolve();
let savedIds = null; // reading_ids already in the file (loaded on first use)

async function loadSavedIds() {
  savedIds = new Set();
  try {
    const lines = (await fs.promises.readFile(DATASET_PATH, 'utf8')).split('\n').filter(Boolean);
    const column = parseCsvLine(lines[0]).indexOf('reading_id');
    if (column >= 0) lines.slice(1).forEach((line) => savedIds.add(parseCsvLine(line)[column]));
  } catch (err) {
    // no file yet
  }
}

// Appends the readings that are not in the file yet (pressing the button twice
// within 2 minutes must not store the same reading twice). Resolves to
// { added, skipped }.
function append(readings, ml, run) {
  const hazards = (ml && ml.hazards) || {};
  const result = writeChain.then(async () => {
    if (!savedIds) await loadSavedIds();
    const fresh = readings.filter((r) => !savedIds.has(String(r._id)));
    if (!fresh.length) return { added: 0, skipped: readings.length };
    const rows = fresh.map((reading) => [
      reading.timestamp instanceof Date ? reading.timestamp.toISOString() : reading.timestamp,
      reading.node_id,
      run.site_id,
      String(reading._id),
      ...SENSOR_COLUMNS.map(([field]) => readingValue(reading, field)),
      run.run_id, run.window_start, run.window_end, readings.length,
      ...HAZARDS.flatMap((h) => [hazards[h] ? hazards[h].probability : null, hazards[h] ? hazards[h].level : null]),
    ].map(csvValue).join(','));

    await fs.promises.mkdir(path.dirname(DATASET_PATH), { recursive: true });
    let exists = true;
    try { await fs.promises.access(DATASET_PATH); } catch (err) { exists = false; }
    const text = (exists ? '' : HEADER.join(',') + '\n') + rows.map((r) => r + '\n').join('');
    await fs.promises.appendFile(DATASET_PATH, text);
    fresh.forEach((r) => savedIds.add(String(r._id)));
    return { added: fresh.length, skipped: readings.length - fresh.length };
  });
  writeChain = result.catch(() => {}); // a failed write must not block later ones
  return result;
}

async function info() {
  try {
    const stat = await fs.promises.stat(DATASET_PATH);
    const text = await fs.promises.readFile(DATASET_PATH, 'utf8');
    const lines = text.split('\n').filter(Boolean);
    return { exists: true, rows: Math.max(lines.length - 1, 0), bytes: stat.size, updated_at: stat.mtime.toISOString() };
  } catch (err) {
    return { exists: false, rows: 0, bytes: 0, updated_at: null };
  }
}

// One CSV line -> cells, honouring "quoted, values" written by csvValue()
function parseCsvLine(line) {
  const cells = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cell += '"'; i += 1; }
      else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { cells.push(cell); cell = ''; }
    else cell += ch;
  }
  cells.push(cell);
  return cells;
}

// Last `limit` data rows as objects, newest first (Database page preview)
async function preview(limit = 50) {
  try {
    const text = await fs.promises.readFile(DATASET_PATH, 'utf8');
    const lines = text.split('\n').filter(Boolean);
    const header = lines[0].split(',');
    return lines.slice(1).slice(-limit).reverse().map((line) => {
      const cells = parseCsvLine(line);
      return Object.fromEntries(header.map((name, i) => [name, cells[i] ?? '']));
    });
  } catch (err) {
    return [];
  }
}

module.exports = { append, info, preview, DATASET_PATH, HEADER };
