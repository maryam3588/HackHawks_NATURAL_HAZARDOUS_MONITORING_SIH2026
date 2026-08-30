const mongoose = require('mongoose');

const predictionSchema = new mongoose.Schema({
  node_id: {
    type: String,
    required: true,
    index: true,
  },
  timestamp: {
    type: Date,
    default: Date.now,
    index: true,
  },
  model_type: String,
  predictions: {
    type: mongoose.Schema.Types.Mixed,
  },
  confidence: Number,
  source: String,
});

module.exports = mongoose.model('Prediction', predictionSchema);