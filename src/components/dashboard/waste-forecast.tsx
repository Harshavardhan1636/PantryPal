'use client';

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { PantryItem } from '@/lib/types';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { addDays, format } from 'date-fns';

type WasteForecastProps = {
  items: PantryItem[];
};

export function WasteForecast({ items }: WasteForecastProps) {
  // Generate forecast data for the next 7 days
  const generateForecastData = () => {
    const today = new Date();
    const forecast = [];

    for (let i = 0; i < 7; i++) {
      const date = addDays(today, i);
      const dateStr = format(date, 'EEE'); // Mon, Tue, Wed, etc.

      // Count items expiring on this day
      const expiringItems = items.filter(item => {
        if (!item.expiryDate) return false;
        const expiryDate = new Date(item.expiryDate);
        return format(expiryDate, 'yyyy-MM-dd') === format(date, 'yyyy-MM-dd');
      });

      // Calculate estimated waste value (rough estimate)
      const estimatedWaste = expiringItems.reduce((sum, item) => {
        const riskValue = item.riskScore || 0.5;
        return sum + riskValue;
      }, 0);

      forecast.push({
        day: dateStr,
        highRisk: items.filter(i => i.riskClass === 'High' && new Date(i.expiryDate || '') >= date).length,
        mediumRisk: items.filter(i => i.riskClass === 'Medium' && new Date(i.expiryDate || '') >= date).length,
        lowRisk: items.filter(i => i.riskClass === 'Low' && new Date(i.expiryDate || '') >= date).length,
        estimatedWaste: Math.round(estimatedWaste * 10) / 10,
        expiring: expiringItems.length,
      });
    }

    return forecast;
  };

  const forecastData = generateForecastData();
  
  // Calculate trend
  const totalHighRiskToday = items.filter(i => i.riskClass === 'High').length;
  const totalHighRiskTomorrow = forecastData[1]?.highRisk || 0;
  const trend = totalHighRiskTomorrow > totalHighRiskToday ? 'up' : 'down';

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle>Weekly Waste Forecast</CardTitle>
            <CardDescription>Predicted food waste risk over the next 7 days</CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {trend === 'up' ? (
              <TrendingUp className="h-5 w-5 text-destructive" />
            ) : (
              <TrendingDown className="h-5 w-5 text-green-500" />
            )}
            <span className={`text-sm font-medium ${trend === 'up' ? 'text-destructive' : 'text-green-500'}`}>
              {trend === 'up' ? 'Increasing' : 'Decreasing'}
            </span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Risk Items Chart */}
          <div>
            <h4 className="text-sm font-medium mb-3">Risk Level Distribution</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="highRisk" fill="#ef4444" name="High Risk" />
                <Bar dataKey="mediumRisk" fill="#f59e0b" name="Medium Risk" />
                <Bar dataKey="lowRisk" fill="#10b981" name="Low Risk" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Items Expiring Chart */}
          <div>
            <h4 className="text-sm font-medium mb-3">Items Expiring Each Day</h4>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={forecastData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="expiring" 
                  stroke="#E5BE8D" 
                  strokeWidth={2}
                  name="Items Expiring"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4 pt-4 border-t">
            <div className="text-center">
              <p className="text-2xl font-bold text-destructive">{items.filter(i => i.riskClass === 'High').length}</p>
              <p className="text-xs text-muted-foreground">High Risk Now</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-orange-500">{forecastData.reduce((sum, d) => sum + d.expiring, 0)}</p>
              <p className="text-xs text-muted-foreground">Expiring This Week</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-500">{items.filter(i => i.riskClass === 'Low').length}</p>
              <p className="text-xs text-muted-foreground">Safe Items</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
