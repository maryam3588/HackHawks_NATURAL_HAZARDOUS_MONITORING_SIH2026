const mongoose = require('mongoose');

const alertSchema = new mongoose.Schema({
  node_id: {
    type: String,
    required: true,
    index: true,
  },
  hazard: {
    type: String,
    required: true,
  },
  risk_score: {
    type: Number,
    required: true,
  },
  severity: {
    type: String,
    enum: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
    required: true,
  },
  message: String,
  timestamp: {
    type: Date,
    default: Date.now,
    index: true,
  },
  status: {
    type: String,
    default: 'ACTIVE',
  },
});

module.exports = mongoose.model('Alert', alertSchema);