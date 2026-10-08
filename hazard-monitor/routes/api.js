const express = require('express');
const router = express.Router();
const SensorReading = require('../models/SensorReading');
const Alert = require('../models/Alert');
const Node = require('../models/Node');
const Prediction = require('../models/Prediction');
const riskEngine = require('../services/riskEngine');
const mlClient = require('../services/mlClient');
const receiveLog = require('../services/receiveLog');
const mlDataset = require('../services/mlDataset');
const espFormat = require('../services/espFormat');

// Extra numeric fields an ESP32 node may send for the ML model. They are not
// needed by the dashboard; they are stored in SensorReading.extra and passed on.
const EXTRA_NUMERIC_FIELDS = [
  'rainfall_mm_h', 'tilt_x_deg', 'tilt_y_deg', 'acceleration_g', 'smoke_raw', 'gas_raw',
  'node1_flood_score', 'node1_landslide_score', 'node1_flood_label', 'node1_landslide_label',
  'node2_wildfire_score', 'node2_extreme_heat_score', 'node2_wildfire_label', 'node2_extreme_heat_label',
  'rain_raw', 'flame_raw', 'device_uptime_s',
];
const NUMERIC_FIELDS = ['soil_moisture', 'water_level', 'temperature', 'humidity', 'tilt', 'smoke', 'signal_strength',
  ...EXTRA_NUMERIC_FIELDS];
const BOOLEAN_FIELDS = ['rain', 'flame'];

// ESP32 firmware often sends 1/0 or "true" for booleans and numbers as text.
// Convert those to proper types before validating. Anything that cannot be
// converted is left as it is, so validation reports it.
// The team sketches' nested format ({"nodeId":..,"sensors":{..}}) is first
// translated to the flat one by services/espFormat.js.
const normalizePayload = (body) => {
  if (!body || typeof body !== 'object' || Array.isArray(body)) return body;
  const data = { ...espFormat.toFlat(body) };
  if (typeof data.node_id === 'string') data.node_id = data.node_id.trim().toUpperCase();
  BOOLEAN_FIELDS.forEach((field) => {
    const value = data[field];
    if (value === 1 || value === '1' || (typeof value === 'string' && value.trim().toLowerCase() === 'true')) data[field] = true;
    if (value === 0 || value === '0' || (typeof value === 'string' && value.trim().toLowerCase() === 'false')) data[field] = false;
  });
  NUMERIC_FIELDS.forEach((field) => {
    const value = data[field];
    if (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value))) data[field] = Number(value);
  });
  return data;
};

// VALIDATION HELPER
const validateSensorData = (data) => {
  const errors = [];

  if (!data || typeof data !== 'object' || Array.isArray(data)) return ['body must be a JSON object'];
  if (!data.node_id) errors.push('node_id required');
  
  // Validate numeric fields
  const numericFields = {
    soil_moisture: [0, 100],
    water_level: [0, 100],
    temperature: [-50, 60],
    humidity: [0, 100],
    tilt: [-90, 90],
    smoke: [0, 1000],
  };

  Object.entries(numericFields).forEach(([field, [min, max]]) => {
    if (data[field] !== undefined && data[field] !== null) {
      const val = parseFloat(data[field]);
      if (isNaN(val)) {
        errors.push(`${field} must be numeric`);
      } else if (val < min || val > max) {
        errors.push(`${field} out of range [${min}, ${max}]`);
      }
    }
  });

  EXTRA_NUMERIC_FIELDS.forEach((field) => {
    if (data[field] !== undefined && data[field] !== null && !Number.isFinite(Number(data[field]))) {
      errors.push(`${field} must be numeric`);
    }
  });

  // Validate boolean fields
  if (data.rain !== undefined && typeof data.rain !== 'boolean') {
    errors.push('rain must be boolean');
  }
  if (data.flame !== undefined && typeof data.flame !== 'boolean') {
    errors.push('flame must be boolean');
  }

  return errors;
};

// POST sensor data
router.post('/sensor-data', async (req, res) => {
  try {
    const data = normalizePayload(req.body);
    const io = req.app.get('io');

    // VALIDATION
    const validationErrors = validateSensorData(data);
    if (validationErrors.length > 0) {
      const packet = receiveLog.record(req, data, { accepted: false, errors: validationErrors });
      if (io) io.emit('esp-packet', packet);
      return res.status(400).json({
        error: 'Validation failed',
        details: validationErrors
      });
    }

    // Extra ML fields the node sent (kept so "Run ML now" can use them later)
    const extra = {};
    EXTRA_NUMERIC_FIELDS.forEach((field) => {
      if (data[field] !== undefined && data[field] !== null) extra[field] = Number(data[field]);
    });

    // Save sensor reading (?? keeps a real 0 reading; || used to turn 0 into null)
    const reading = new SensorReading({
      node_id: data.node_id,
      soil_moisture: data.soil_moisture ?? null,
      rain: data.rain ?? false,
      water_level: data.water_level ?? null,
      temperature: data.temperature ?? null,
      humidity: data.humidity ?? null,
      tilt: data.tilt ?? null,
      smoke: data.smoke ?? null,
      flame: data.flame ?? false,
      signal_strength: data.signal_strength ?? null,
      extra: Object.keys(extra).length ? extra : undefined,
    });

    const savedReading = await reading.save();

    // Calculate risks
    const risks = riskEngine.calculateAllRisks(data.node_id, data);

    // PREVENT ALERT SPAM - Check for existing active alerts
    const alertPromises = Object.entries(risks).map(async ([hazard, score]) => {
      if (score >= 40) {
        const severity = riskEngine.getSeverity(score);
        
        // Check if alert already exists for this node/hazard
        const existingAlert = await Alert.findOne({
          node_id: data.node_id,
          hazard: hazard,
          status: 'ACTIVE'
        });

        if (!existingAlert) {
          // Create new alert only if one doesn't exist
          const alert = new Alert({
            node_id: data.node_id,
            hazard: hazard,
            risk_score: score,
            severity: severity,
            message: `${hazard} risk at ${severity} level (${Math.round(score)}%)`,
          });
          return alert.save();
        } else {
          // Update existing alert with new risk score
          return Alert.findByIdAndUpdate(
            existingAlert._id,
            { 
              risk_score: score,
              timestamp: new Date() // Update timestamp
            },
            { new: true }
          );
        }
      }
      return null;
    });

    await Promise.all(alertPromises);

    // Update node last_seen
    await Node.findOneAndUpdate(
      { node_id: data.node_id },
      { last_seen: new Date() }
    );

    // ML MODEL: chance of disaster per hazard (null when not due or ML is down)
    const ml = await mlClient.predict({
      ...data,
      timestamp: data.timestamp || savedReading.timestamp,
    });
    if (ml) {
      // keep a history of ML predictions (best effort, never blocks the reading)
      Prediction.create({
        node_id: data.node_id,
        timestamp: ml.timestamp || savedReading.timestamp,
        model_type: 'hazard_ml_v4',
        predictions: ml.hazards,
        source: 'pi_ml_service',
      }).catch((err) => console.error('Prediction save error:', err.message));
    }

    // EMIT TO SOCKET.IO
    if (io) {
      io.emit('sensor-update', {
        ...data,
        // real ESP32 nodes usually have no clock -> fall back to server time
        timestamp: data.timestamp || savedReading.timestamp,
        risks: risks
      });
      if (ml) io.emit('ml-update', ml);
      io.emit('esp-packet', receiveLog.record(req, data, { accepted: true, ml_sent: Boolean(ml) }));
    } else {
      receiveLog.record(req, data, { accepted: true, ml_sent: Boolean(ml) });
    }

    res.json({
      success: true,
      recorded_id: savedReading._id,
      risks: risks,
      ml: ml,
    });
  } catch (err) {
    // e.g. database down: show it on the ESP Live page too
    const packet = receiveLog.record(req, req.body, { accepted: false, errors: [`server error: ${err.message}`] });
    const io = req.app.get('io');
    if (io) io.emit('esp-packet', packet);
    res.status(500).json({ error: err.message });
  }
});

// GET latest reading
router.get('/sensor-data/latest', async (req, res) => {
  try {
    const nodeId = req.query.node || 'NODE_01';
    
    const reading = await SensorReading.findOne({ node_id: nodeId })
      .sort({ timestamp: -1 })
      .lean();

    res.json(reading || {});
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET historical data
router.get('/sensor-data/history', async (req, res) => {
  try {
    const nodeId = req.query.node || 'NODE_01';
    const limit = Math.min(parseInt(req.query.limit) || 100, 5000);

    const readings = await SensorReading.find({ node_id: nodeId })
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();

    res.json(readings);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET all nodes
router.get('/nodes', async (req, res) => {
  try {
    const nodes = await Node.find().lean();
    res.json(nodes);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET alerts
router.get('/alerts', async (req, res) => {
  try {
    const alerts = await Alert.find({ status: 'ACTIVE' })
      .sort({ timestamp: -1 })
      .limit(50)
      .lean();

    res.json(alerts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET alerts by node
router.get('/alerts/:nodeId', async (req, res) => {
  try {
    const alerts = await Alert.find({ 
      node_id: req.params.nodeId,
      status: 'ACTIVE' 
    })
      .sort({ timestamp: -1 })
      .lean();

    res.json(alerts);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// Resolve alert
router.put('/alerts/:alertId/resolve', async (req, res) => {
  try {
    const alert = await Alert.findByIdAndUpdate(
      req.params.alertId,
      { status: 'RESOLVED' },
      { new: true }
    );
    res.json(alert);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET database statistics
router.get('/db-status', async (req, res) => {
  try {
    const totalReadings = await SensorReading.countDocuments();
    const totalAlerts = await Alert.countDocuments({ status: 'ACTIVE' });
    const node1Readings = await SensorReading.countDocuments({ node_id: 'NODE_01' });
    const node2Readings = await SensorReading.countDocuments({ node_id: 'NODE_02' });
    const totalPredictions = await Prediction.countDocuments();

    res.json({
      totalReadings,
      totalAlerts,
      node1Readings,
      node2Readings,
      totalPredictions
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET training data for ML (last N readings)
router.get('/ml/training-data', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit) || 1000, 50000);
    const nodeId = req.query.node;

    let query = {};
    if (nodeId) query.node_id = nodeId;

    const readings = await SensorReading.find(query)
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();

    res.json({
      count: readings.length,
      data: readings
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET risk predictions (for ML model testing)
router.get('/ml/predictions', async (req, res) => {
  try {
    const nodeId = req.query.node || 'NODE_01';
    const limit = Math.min(parseInt(req.query.limit) || 50, 1000);

    const readings = await SensorReading.find({ node_id: nodeId })
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();

    const predictions = readings.map(reading => {
      const risks = riskEngine.calculateAllRisks(nodeId, reading);
      return {
        timestamp: reading.timestamp,
        ...risks
      };
    });

    res.json(predictions);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST ML model predictions (receive from ML teammate)
router.post('/ml/predictions/save', async (req, res) => {
  try {
    const { node_id, timestamp, model_type, predictions, confidence, source } = req.body;

    if (!node_id || !predictions) {
      return res.status(400).json({ error: 'node_id and predictions required' });
    }

    // Save prediction to database
    const prediction = new Prediction({
      node_id: node_id,
      timestamp: timestamp || new Date(),
      model_type: model_type || 'custom_model',
      predictions: predictions,
      confidence: confidence || null,
      source: source || 'ml_team',
    });

    const savedPrediction = await prediction.save();

    res.json({
      success: true,
      message: 'Prediction saved',
      prediction_id: savedPrediction._id,
      data: { node_id, timestamp, predictions }
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET ML predictions history
router.get('/ml/predictions/history', async (req, res) => {
  try {
    const nodeId = req.query.node;
    const modelType = req.query.model;
    const limit = Math.min(parseInt(req.query.limit) || 100, 1000);

    let query = {};
    if (nodeId) query.node_id = nodeId;
    if (modelType) query.model_type = modelType;

    const predictions = await Prediction.find(query)
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();

    res.json({
      count: predictions.length,
      data: predictions
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---------------------------------------------------------------------------
// RUN ML NOW: combine each node's readings from the last 2 minutes into one
// reading, send it to the ML model straight away, and add every reading of
// the window (with the ML result) to the ML dataset CSV.
// ---------------------------------------------------------------------------
const RUN_NOW_WINDOW_SECONDS = 120;
const COMBINE_NUMERIC = ['soil_moisture', 'water_level', 'temperature', 'humidity', 'tilt', 'smoke', 'signal_strength'];

// Averages numeric values, "any true" for rain/flame, the newest timestamp.
// One combined reading per 2 minutes matches what the model receives
// automatically; 60 raw readings 2 s apart would distort its time windows.
function combineReadings(nodeId, readings) {
  const mean = (values) => {
    const valid = values.filter((v) => typeof v === 'number' && Number.isFinite(v));
    return valid.length ? valid.reduce((a, b) => a + b, 0) / valid.length : undefined;
  };
  const combined = { node_id: nodeId, timestamp: readings[readings.length - 1].timestamp };
  COMBINE_NUMERIC.forEach((field) => {
    const value = mean(readings.map((r) => r[field]));
    if (value !== undefined) combined[field] = value;
  });
  combined.rain = readings.some((r) => r.rain === true);
  combined.flame = readings.some((r) => r.flame === true);
  EXTRA_NUMERIC_FIELDS.forEach((field) => {
    const value = mean(readings.map((r) => (r.extra ? r.extra[field] : undefined)));
    if (value !== undefined) combined[field] = value;
  });
  return combined;
}

router.post('/ml/run-now', async (req, res) => {
  try {
    const io = req.app.get('io');
    const windowEnd = new Date();
    const windowStart = new Date(windowEnd.getTime() - RUN_NOW_WINDOW_SECONDS * 1000);
    const runId = `run_${windowEnd.toISOString()}`;

    const readings = await SensorReading.find({ timestamp: { $gte: windowStart } })
      .sort({ timestamp: 1 })
      .lean();
    const byNode = {};
    readings.forEach((r) => { (byNode[r.node_id] = byNode[r.node_id] || []).push(r); });
    ['NODE_01', 'NODE_02'].forEach((nodeId) => { byNode[nodeId] = byNode[nodeId] || []; });

    const results = {};
    for (const [nodeId, nodeReadings] of Object.entries(byNode)) {
      if (!nodeReadings.length) {
        results[nodeId] = { readings: 0, ml: null, dataset_rows: 0, message: 'no readings in the last 2 minutes' };
        continue;
      }
      const ml = await mlClient.predictNow(combineReadings(nodeId, nodeReadings));
      if (ml) {
        Prediction.create({
          node_id: nodeId,
          timestamp: ml.timestamp || windowEnd,
          model_type: 'hazard_ml_v4_run_now',
          predictions: ml.hazards,
          source: 'run_ml_now_button',
        }).catch((err) => console.error('Prediction save error:', err.message));
        if (io) io.emit('ml-update', ml);
      }
      const dataset = await mlDataset.append(nodeReadings, ml, {
        run_id: runId,
        site_id: process.env.SITE_ID || 'pi_site_01',
        window_start: windowStart.toISOString(),
        window_end: windowEnd.toISOString(),
      });
      results[nodeId] = {
        readings: nodeReadings.length,
        ml,
        dataset_rows: dataset.added,
        dataset_already_saved: dataset.skipped,
        message: ml ? 'sent to ML' : `ML not reached: ${(await mlClient.status()).last_error || 'unknown error'}`,
      };
    }

    res.json({
      run_id: runId,
      window_seconds: RUN_NOW_WINDOW_SECONDS,
      window_start: windowStart.toISOString(),
      window_end: windowEnd.toISOString(),
      results,
      dataset: await mlDataset.info(),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET ML dataset: size + newest rows (Database page)
router.get('/ml/dataset', async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit, 10) || 50, 500);
  const info = await mlDataset.info();
  res.json({
    exists: info.exists,
    total_rows: info.rows,
    bytes: info.bytes,
    updated_at: info.updated_at,
    columns: mlDataset.HEADER,
    rows: await mlDataset.preview(limit),
  });
});

// GET ML dataset as a CSV file
router.get('/ml/dataset.csv', (req, res) => {
  res.download(mlDataset.DATASET_PATH, 'ml_dataset.csv', (err) => {
    if (err && !res.headersSent) res.status(404).json({ error: 'No ML dataset yet - press "Run ML now" on the dashboard first.' });
  });
});

// GET database overview (Database page)
router.get('/database/overview', async (req, res) => {
  try {
    const nodeId = req.query.node;
    const limit = Math.min(parseInt(req.query.limit, 10) || 100, 1000);
    const readingQuery = nodeId ? { node_id: nodeId } : {};
    const [totalReadings, node1Readings, node2Readings, activeAlerts, totalPredictions, readings, alerts, predictions] = await Promise.all([
      SensorReading.countDocuments(),
      SensorReading.countDocuments({ node_id: 'NODE_01' }),
      SensorReading.countDocuments({ node_id: 'NODE_02' }),
      Alert.countDocuments({ status: 'ACTIVE' }),
      Prediction.countDocuments(),
      SensorReading.find(readingQuery).sort({ timestamp: -1 }).limit(limit).lean(),
      Alert.find().sort({ timestamp: -1 }).limit(50).lean(),
      Prediction.find().sort({ timestamp: -1 }).limit(50).lean(),
    ]);
    res.json({
      counts: { totalReadings, node1Readings, node2Readings, activeAlerts, totalPredictions },
      readings,
      alerts,
      predictions,
      dataset: await mlDataset.info(),
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET all sensor readings as CSV (newest first, up to ?limit=, default 10000)
router.get('/sensor-data/export.csv', async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit, 10) || 10000, 100000);
    const query = req.query.node ? { node_id: req.query.node } : {};
    const readings = await SensorReading.find(query).sort({ timestamp: -1 }).limit(limit).lean();
    const columns = ['timestamp', 'node_id', 'water_level', 'soil_moisture', 'rain', 'temperature', 'humidity',
      'tilt', 'smoke', 'flame', 'signal_strength', ...EXTRA_NUMERIC_FIELDS];
    const cell = (v) => {
      if (v === null || v === undefined) return '';
      if (v instanceof Date) return v.toISOString();
      const text = String(v);
      return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    const lines = [columns.join(',')].concat(readings.map((r) => columns.map((c) => cell(
      r[c] !== undefined ? r[c] : (r.extra ? r.extra[c] : undefined))).join(',')));
    res.setHeader('Content-Type', 'text/csv');
    res.setHeader('Content-Disposition', 'attachment; filename="sensor_readings.csv"');
    res.send(lines.join('\n') + '\n');
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET live receiver log (ESP Live page)
router.get('/esp/recent', (req, res) => {
  res.json(receiveLog.summary());
});

// GET ML model status (is the ML service connected?)
router.get('/ml/status', async (req, res) => {
  res.json(await mlClient.status());
});

// GET latest ML result per node (dashboard uses this on page load)
router.get('/ml/latest', (req, res) => {
  res.json(mlClient.latest());
});

// GET anomaly detection data
router.get('/ml/anomalies', async (req, res) => {
  try {
    const nodeId = req.query.node || 'NODE_01';
    
    const readings = await SensorReading.find({ node_id: nodeId })
      .sort({ timestamp: -1 })
      .limit(100)
      .lean();

    // Calculate mean and std dev for each field
    const fields = ['temperature', 'humidity', 'water_level', 'smoke'];
    const stats = {};

    fields.forEach(field => {
      const values = readings
        .map(r => r[field])
        .filter(v => v !== null && v !== undefined);
      
      if (values.length === 0) return;

      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, v) => a + Math.pow(v - mean, 2), 0) / values.length;
      const stdDev = Math.sqrt(variance);

      stats[field] = { 
        mean: parseFloat(mean.toFixed(2)), 
        stdDev: parseFloat(stdDev.toFixed(2)), 
        min: Math.min(...values), 
        max: Math.max(...values) 
      };
    });

    res.json({
      node_id: nodeId,
      sample_size: readings.length,
      statistics: stats,
      readings: readings.slice(0, 10)
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;