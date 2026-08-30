class RiskEngine {
  calculateFloodRisk(data) {
    let score = 0;
    
    if (data.water_level > 80) score += 40;
    else if (data.water_level > 60) score += 25;
    else if (data.water_level > 40) score += 10;
    
    if (data.rain === true) score += 30;
    
    if (data.humidity > 85) score += 15;
    else if (data.humidity > 70) score += 8;
    
    if (data.soil_moisture > 80) score += 10;
    
    return Math.min(score, 100);
  }

  calculateLandslideRisk(data) {
    let score = 0;
    
    if (data.tilt > 3) score += 50;
    else if (data.tilt > 2) score += 35;
    else if (data.tilt > 1) score += 15;
    
    if (data.soil_moisture > 85) score += 20;
    else if (data.soil_moisture > 70) score += 10;
    
    if (data.rain === true) score += 20;
    
    if (data.humidity > 80 && data.soil_moisture > 75) score += 15;
    
    return Math.min(score, 100);
  }

  calculateFireRisk(data) {
    let score = 0;
    
    if (data.smoke > 400) score += 40;
    else if (data.smoke > 250) score += 25;
    else if (data.smoke > 150) score += 10;
    
    if (data.flame === true) score += 50;
    
    if (data.temperature > 40) score += 20;
    else if (data.temperature > 35) score += 10;
    
    if (data.humidity < 30) score += 15;
    else if (data.humidity < 50) score += 8;
    
    return Math.min(score, 100);
  }

  calculateHeatRisk(data) {
    let score = 0;
    
    if (data.temperature > 45) score += 50;
    else if (data.temperature > 40) score += 35;
    else if (data.temperature > 35) score += 15;
    
    if (data.humidity < 20) score += 25;
    else if (data.humidity < 40) score += 15;
    
    if (data.temperature > 40 && data.humidity < 30) score += 10;
    
    return Math.min(score, 100);
  }

  getSeverity(riskScore) {
    if (riskScore >= 80) return 'CRITICAL';
    if (riskScore >= 60) return 'HIGH';
    if (riskScore >= 40) return 'MEDIUM';
    return 'LOW';
  }

  calculateAllRisks(nodeId, data) {
    if (nodeId === 'NODE_01') {
      return {
        FLOOD: this.calculateFloodRisk(data),
        LANDSLIDE: this.calculateLandslideRisk(data)
      };
    } else if (nodeId === 'NODE_02') {
      return {
        FIRE: this.calculateFireRisk(data),
        HEAT: this.calculateHeatRisk(data)
      };
    }
    return {};
  }
}

module.exports = new RiskEngine();