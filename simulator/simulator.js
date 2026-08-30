const node1 = require('./node1');
const node2 = require('./node2');
const axios = require('axios');

class Simulator {
  constructor(apiUrl = 'http://localhost:3000') {
    this.apiUrl = apiUrl;
    this.intervalId = null;
    this.running = false;
  }

  async sendReading(reading) {
    try {
      await axios.post(`${this.apiUrl}/api/sensor-data`, reading);
    } catch (err) {
      console.error('Simulator send error:', err.message);
    }
  }

  start() {
    if (this.running) return;
    this.running = true;
    console.log('🎯 Simulator started');

    this.intervalId = setInterval(async () => {
      const reading1 = node1.generateReading();
      const reading2 = node2.generateReading();

      await this.sendReading(reading1);
      await this.sendReading(reading2);
    }, 2000);
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.running = false;
      console.log('⏹ Simulator stopped');
    }
  }

  triggerFloodNode1() {
    node1.triggerFlood();
  }

  triggerLandslideNode1() {
    node1.triggerLandslide();
  }

  triggerFireNode2() {
    node2.triggerFire();
  }

  triggerHeatNode2() {
    node2.triggerHeat();
  }

  resetAll() {
    node1.reset();
    node2.reset();
  }
}

module.exports = new Simulator();