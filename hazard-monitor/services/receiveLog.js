// In-memory log of every packet that reaches POST /api/sensor-data, accepted
// or rejected, for the ESP Live page. Kept in memory only (last MAX_PACKETS),
// so it resets when the server restarts; accepted readings are also saved in
// the database as usual.
const MAX_PACKETS = 300;

const packets = [];
const nodes = {}; // node_id -> { count, accepted, rejected, last_seen, last_ip, recent: [ms timestamps] }
let sequence = 0;

function clientIp(req) {
  const ip = (req.headers['x-forwarded-for'] || '').split(',')[0].trim() || req.socket.remoteAddress || '';
  return ip.replace(/^::ffff:/, '');
}

function record(req, payload, outcome) {
  const now = Date.now();
  const nodeId = payload && typeof payload.node_id === 'string' ? payload.node_id : '(missing node_id)';
  const packet = {
    id: ++sequence,
    received_at: new Date(now).toISOString(),
    node_id: nodeId,
    ip: clientIp(req),
    user_agent: String(req.headers['user-agent'] || '').slice(0, 80),
    accepted: outcome.accepted,
    errors: outcome.errors || [],
    ml_sent: Boolean(outcome.ml_sent),
    payload: payload,
  };
  packets.unshift(packet);
  if (packets.length > MAX_PACKETS) packets.length = MAX_PACKETS;

  // a packet without a node_id (e.g. broken JSON) is listed, but is not a node
  if (!payload || typeof payload.node_id !== 'string') return packet;
  const stats = nodes[nodeId] || (nodes[nodeId] = { count: 0, accepted: 0, rejected: 0, recent: [] });
  stats.count += 1;
  stats[outcome.accepted ? 'accepted' : 'rejected'] += 1;
  stats.last_seen = packet.received_at;
  stats.last_ip = packet.ip;
  stats.recent.push(now);
  while (stats.recent.length && now - stats.recent[0] > 60000) stats.recent.shift();
  return packet;
}

function summary() {
  const now = Date.now();
  const nodeSummary = {};
  Object.entries(nodes).forEach(([nodeId, s]) => {
    nodeSummary[nodeId] = {
      count: s.count,
      accepted: s.accepted,
      rejected: s.rejected,
      last_seen: s.last_seen,
      last_ip: s.last_ip,
      packets_last_minute: s.recent.filter((t) => now - t <= 60000).length,
    };
  });
  return { nodes: nodeSummary, packets: packets.slice(0, 100), server_time: new Date(now).toISOString() };
}

module.exports = { record, summary };
