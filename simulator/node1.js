class Node1Simulator {
  constructor() {
    this.state = {
      soil_moisture: 60,
      rain: false,
      water_level: 45,
      temperature: 28,
      humidity: 65,
      tilt: 0.5
    };
  }

  generateReading() {
    this.state.water_level += (Math.random() - 0.5) * 2;
    this.state.water_level = Math.max(0, Math.min(100, this.state.water_level));

    this.state.soil_moisture += (Math.random() - 0.5) * 1.5;
    this.state.soil_moisture = Math.max(30, Math.min(95, this.state.soil_moisture));

    if (Math.random() < 0.05) {
      this.state.rain = true;
    } else if (Math.random() < 0.03) {
      this.state.rain = false;
    }

    this.state.temperature += (Math.random() - 0.5) * 0.8;
    this.state.temperature = Math.max(15, Math.min(40, this.state.temperature));

    this.state.humidity += (Math.random() - 0.5) * 2;
    this.state.humidity = Math.max(40, Math.min(95, this.state.humidity));

    this.state.tilt += (Math.random() - 0.5) * 0.1;
    this.state.tilt = Math.max(-5, Math.min(5, this.state.tilt));

    return {
      node_id: 'NODE_01',
      timestamp: new Date().toISOString(),
      ...this.state
    };
  }

  triggerFlood() {
    this.state.water_level = 75;
    this.state.rain = true;
    this.state.humidity = 90;
    this.state.soil_moisture = 85;
  }

  triggerLandslide() {
    this.state.tilt = 3.5;
    this.state.soil_moisture = 88;
    this.state.rain = true;
  }

  reset() {
    this.state = {
      soil_moisture: 60,
      rain: false,
      water_level: 45,
      temperature: 28,
      humidity: 65,
      tilt: 0.5
    };
  }
}

module.exports = new Node1Simulator();