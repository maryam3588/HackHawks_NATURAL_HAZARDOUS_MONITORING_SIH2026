const mongoose = require('mongoose');

const nodeSchema = new mongoose.Schema({
  node_id: {
    type: String,
    unique: true,
    required: true,
  },
  name: {
    type: String,
    required: true,
  },
  location: String,
  hazards: [String],
  status: {
    type: String,
    default: 'online',
  },
  last_seen: {
    type: Date,
    default: Date.now,
  },
  created_at: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('Node', nodeSchema);