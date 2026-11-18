import { Router } from 'express';
import { handlePrediction } from '../controllers/predictionController';
import { get30DayTrends, get7DayForecast, getModelMetrics } from '../controllers/weatherController';

const router = Router();

router.post('/prediction', handlePrediction);
router.get('/trends/30days', get30DayTrends);
router.get('/forecast/7days', get7DayForecast);
router.get('/model/metrics', getModelMetrics);

export default router;
