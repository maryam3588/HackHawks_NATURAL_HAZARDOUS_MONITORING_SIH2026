const Node = require('../models/Node');

const initializeDefaultNodes = async () => {
  try {
    const existingNodes = await Node.countDocuments();
    
    if (existingNodes === 0) {
      await Node.create([
        {
          node_id: 'NODE_01',
          name: 'Flood & Landslide Monitor',
          location: 'Zone A',
          hazards: ['FLOOD', 'LANDSLIDE'],
          status: 'online',
        },
        {
          node_id: 'NODE_02',
          name: 'Fire & Heat Monitor',
          location: 'Zone B',
          hazards: ['FIRE', 'HEAT'],
          status: 'online',
        },
      ]);
      console.log('✓ Default nodes initialized');
    }
  } catch (err) {
    console.error('Initialization error:', err);
  }
};

module.exports = { initializeDefaultNodes };