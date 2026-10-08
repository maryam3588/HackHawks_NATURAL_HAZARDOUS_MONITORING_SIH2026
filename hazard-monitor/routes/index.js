const express = require('express');
const router = express.Router();

router.get('/', (req, res) => {
  res.redirect('/dashboard');
});

router.get('/dashboard', (req, res) => {
  res.render('dashboard');
});

// Live view of packets arriving from the ESP32 nodes
router.get('/esp', (req, res) => {
  res.render('esp');
});

// Stored data: readings, alerts, ML predictions, ML dataset
router.get('/database', (req, res) => {
  res.render('database');
});

module.exports = router;