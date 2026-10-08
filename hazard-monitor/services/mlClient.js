// Connects the dashboard to the ML service (ml/live_inference.py) that runs
// on the same Pi at ML_URL. The ML service keeps its own rolling history per
// node and returns, for every hazard, a calibrated chance of disaster plus a
// SAFE / WATCH / WARNING level.
const axios = require('axios');

const ML_URL = (process.env.ML_URL || 'http://127.0.0.1:5001').replace(/\/$/, '');
const SITE_ID = process.env.SITE_ID || 'pi_site_01';

// The models were trained on one reading per node every 5 minutes. Readings
// arriving faster (the simulator sends every 2 s) are still saved and shown,
// but only one per node per ML_SAMPLE_SECONDS is sent to the ML model.
const parsedSample = parseFloat(process.env.ML_SAMPLE_SECONDS);
const SAMPLE_SECONDS = Number.isFinite(parsedSample) && parsedSample >= 0 ? parsedSample : 300;

const state = {
  lastSentAt: {},       // node_id -> ms timestamp of last reading sent to ML
  latest: {},           // node_id -> last ML result
  lastSuccessAt: null,
  lastError: null,
};

async function predict(reading) {
  const nodeId = reading.node_id;
  const now = Date.now();
  if (state.lastSentAt[nodeId] && now - state.lastSentAt[nodeId] < SAMPLE_SECONDS * 1000) {
    return null; // not due yet for this node
  }
  state.lastSentAt[nodeId] = now;

  try {
    const response = await axios.post(`${ML_URL}/predict`, {
      ...reading,
      site_id: reading.site_id || SITE_ID,
      timestamp: reading.timestamp || new Date(now).toISOString(),
    }, { timeout: 3000 });

    const result = response.data;
    if (!result || result.error || !result.hazards) {
      state.lastError = (result && result.error) || 'ML service returned no prediction';
      return null;
    }
    const entry = { ...result, node_id: nodeId, received_at: new Date(now).toISOString() };
    state.latest[nodeId] = entry;
    state.lastSuccessAt = entry.received_at;
    state.lastError = null;
    return entry;
  } catch (err) {
    // Let the next reading retry instead of waiting a full sample period
    delete state.lastSentAt[nodeId];
    state.lastError = err.code === 'ECONNREFUSED' ? 'ML service is not running' : err.message;
    return null;
  }
}

async function status() {
  try {
    const response = await axios.get(`${ML_URL}/health`, { timeout: 2000 });
    const health = response.data || {};
    // The service is reachable again: an old "not running" error is stale
    if (state.lastError === 'ML service is not running') state.lastError = null;
    return {
      working: health.status === 'ok',
      models: health.models || {},
      unavailable: health.unavailable || [],
      last_prediction_at: state.lastSuccessAt,
      last_error: state.lastError,
      sample_seconds: SAMPLE_SECONDS,
    };
  } catch (err) {
    return {
      working: false,
      models: {},
      unavailable: [],
      last_prediction_at: state.lastSuccessAt,
      last_error: err.code === 'ECONNREFUSED' ? 'ML service is not running' : err.message,
      sample_seconds: SAMPLE_SECONDS,
    };
  }
}

function latest() {
  return state.latest;
}

module.exports = { predict, status, latest, SAMPLE_SECONDS };
