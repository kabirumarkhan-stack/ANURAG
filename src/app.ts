import express from 'express';
import cors from 'cors';
import path from 'path';
import apiRoutes from './routes/api';

const app = express();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve API Routes
app.use('/api', apiRoutes);

// Serve static frontend files
const clientDist = path.join(__dirname, '../client/dist');
const templatesDir = path.join(__dirname, '../templates');

app.use(express.static(clientDist));
app.use(express.static(templatesDir));

app.get('*', (req, res) => {
  const indexHtml = path.join(templatesDir, 'index.html');
  res.sendFile(indexHtml);
});

export default app;
