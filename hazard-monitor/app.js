require('dotenv').config();
const express = require('express');
const path = require('path');
const cookieParser = require('cookie-parser');
const logger = require('morgan');

const app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));
// Serve Chart.js locally so the dashboard works with no internet (Pi offline / hotspot)
app.use('/vendor/chart.js', express.static(path.join(__dirname, 'node_modules', 'chart.js', 'dist')));

// Routes
const indexRouter = require('./routes/index');
const apiRouter = require('./routes/api');

app.use('/', indexRouter);
app.use('/api', apiRouter);



// Malformed JSON sent to the receiver: log it for the ESP Live page and answer
// with JSON (an ESP32 cannot read the HTML error page)
const receiveLog = require('./services/receiveLog');
app.use((err, req, res, next) => {
  if (err.type === 'entity.parse.failed' && req.path === '/api/sensor-data') {
    const raw = typeof err.body === 'string' ? err.body.slice(0, 300) : '';
    const packet = receiveLog.record(req, { raw_body: raw }, { accepted: false, errors: ['body is not valid JSON'] });
    const io = req.app.get('io');
    if (io) io.emit('esp-packet', packet);
    return res.status(400).json({ error: 'Validation failed', details: ['body is not valid JSON'] });
  }
  return next(err);
});

// error handler
app.use((err, req, res, next) => {
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;