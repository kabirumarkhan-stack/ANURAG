import { Request, Response } from 'express';
import { MLService } from '../services/mlService';

export const get30DayTrends = async (req: Request, res: Response) => {
  const city = (req.query.city as string) || 'delhi';
  const result = MLService.get30DayTrends(city);
  return res.json(result);
};

export const get7DayForecast = async (req: Request, res: Response) => {
  const city = (req.query.city as string) || 'delhi';
  const result = await MLService.get7DayForecast(city);
  return res.json(result);
};

export const getModelMetrics = async (req: Request, res: Response) => {
  const metrics = MLService.getMetrics();
  return res.json({ status: 'success', metrics });
};
