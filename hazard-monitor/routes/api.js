const express = require('express');
const router = express.Router();
const SensorReading = require('../models/SensorReading');
const Alert = require('../models/Alert');
const Node = require('../models/Node');
const Prediction = require('../models/Prediction');
const riskEngine = require('../services/riskEngine');
const mlClient = require('../services/mlClient');

// VALIDATION HELPER
const validateSensorData = (data) => {
  const errors = [];

  if (!data.node_id) errors.push('node_id required');
  
  // Validate numeric fields
  const numericFields = {
    soil_moisture: [0, 100],
    water_level: [0, 100],
    temperature: [-50, 60],
    humidity: [0, 100],
    tilt: [-10, 10],
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
    const data = req.body;
    const io = req.app.get('io');

    // VALIDATION
    const validationErrors = validateSensorData(data);
    if (validationErrors.length > 0) {
      return res.status(400).json({ 
        error: 'Validation failed', 
        details: validationErrors 
      });
    }

    // Save sensor reading
    const reading = new SensorReading({
      node_id: data.node_id,
      soil_moisture: data.soil_moisture || null,
      rain: data.rain || false,
      water_level: data.water_level || null,
      temperature: data.temperature || null,
      humidity: data.humidity || null,
      tilt: data.tilt || null,
      smoke: data.smoke || null,
      flame: data.flame || false,
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
    }

    res.json({
      success: true,
      recorded_id: savedReading._id,
      risks: risks,
      ml: ml,
    });
  } catch (err) {
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
    const limit = parseInt(req.query.limit) || 100;

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
    const limit = parseInt(req.query.limit) || 1000;
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
    const limit = parseInt(req.query.limit) || 50;

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
    const limit = parseInt(req.query.limit) || 100;

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