import { Request, Response } from 'express';
import { MLService } from '../services/mlService';

export const handlePrediction = async (req: Request, res: Response) => {
  try {
    const { temperature, humidity, windSpeed, rainfall, pressure } = req.body;

    const input = {
      temperature: Number(temperature) || 28.0,
      humidity: Number(humidity) || 65.0,
      windSpeed: Number(windSpeed) || 12.0,
      rainfall: Number(rainfall) || 0.0,
      pressure: Number(pressure) || 1013.0
    };

    const result = await MLService.predictTemperature(input);
    return res.status(200).json(result);
  } catch (error: any) {
    return res.status(500).json({
      status: 'error',
      message: error.message || 'Internal Prediction Engine Failure'
    });
  }
};
