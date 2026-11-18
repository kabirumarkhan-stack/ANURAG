import app from './app';

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`====================================================`);
  console.log(`🌿 Climate AI Node.js Backend Server running`);
  console.log(`🚀 API active at: http://localhost:${PORT}/api/`);
  console.log(`   - POST /api/prediction`);
  console.log(`   - GET  /api/trends/30days`);
  console.log(`   - GET  /api/forecast/7days`);
  console.log(`   - GET  /api/model/metrics`);
  console.log(`====================================================`);
});
