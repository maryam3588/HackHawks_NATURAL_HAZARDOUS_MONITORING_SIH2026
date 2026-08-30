const mongoose = require('mongoose');

const sensorReadingSchema = new mongoose.Schema({
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
  soil_moisture: Number,
  rain: Boolean,
  water_level: Number,
  temperature: Number,
  humidity: Number,
  tilt: Number,
  smoke: Number,
  flame: Boolean,
  signal_strength: Number,
});

sensorReadingSchema.index({ node_id: 1, timestamp: -1 });

module.exports = mongoose.model('SensorReading', sensorReadingSchema);