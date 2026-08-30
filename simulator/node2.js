class Node2Simulator {
  constructor() {
    this.state = {
      smoke: 100,
      temperature: 32,
      humidity: 50,
      flame: false
    };
  }

  generateReading() {
    this.state.smoke += (Math.random() - 0.5) * 15;
    this.state.smoke = Math.max(50, Math.min(500, this.state.smoke));

    this.state.temperature += (Math.random() - 0.5) * 1.2;
    this.state.temperature = Math.max(20, Math.min(50, this.state.temperature));

    this.state.humidity += (Math.random() - 0.5) * 2;
    this.state.humidity = Math.max(20, Math.min(80, this.state.humidity));

    if (Math.random() < 0.02) {
      this.state.flame = true;
    } else if (Math.random() < 0.05) {
      this.state.flame = false;
    }

    return {
      node_id: 'NODE_02',
      timestamp: new Date().toISOString(),
      ...this.state
    };
  }

  triggerFire() {
    this.state.smoke = 450;
    this.state.flame = true;
    this.state.temperature = 42;
    this.state.humidity = 25;
  }

  triggerHeat() {
    this.state.temperature = 46;
    this.state.humidity = 15;
    this.state.smoke = 150;
  }

  reset() {
    this.state = {
      smoke: 100,
      temperature: 32,
      humidity: 50,
      flame: false
    };
  }
}

module.exports = new Node2Simulator();