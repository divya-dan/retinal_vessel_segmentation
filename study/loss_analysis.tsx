import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import _ from 'lodash';

const LossAnalysis = () => {
  const [data, setData] = useState([]);
  const [selectedMetrics, setSelectedMetrics] = useState([
    'segres-dice_ce-full_train_loss',
    'segres-dice_ce-full_val_loss',
    'unet-dice_ce-full_train_loss',
    'unet-dice_ce-full_val_loss'
  ]);
  const [viewMode, setViewMode] = useState('comparison');

  useEffect(() => {
    const loadData = async () => {
      try {
        const csvData = await window.fs.readFile('losses_wide.csv', { encoding: 'utf8' });
        const lines = csvData.trim().split('\n');
        const headers = lines[0].split(',');
        
        const parsedData = lines.slice(1).map(line => {
          const values = line.split(',');
          const row = { epoch: parseInt(values[0]) };
          
          for (let i = 1; i < headers.length; i++) {
            const value = values[i];
            row[headers[i]] = value && value.trim() !== '' ? parseFloat(value) : null;
          }
          
          return row;
        });
        
        setData(parsedData);
      } catch (error) {
        console.error('Error loading data:', error);
      }
    };
    
    loadData();
  }, []);

  const colors = {
    'segres-dice_ce-full_train_loss': '#2563eb',
    'segres-dice_ce-full_val_loss': '#3b82f6',
    'segres-dice_ce-patch_train_loss': '#1e40af',
    'segres-dice_ce-patch_val_loss': '#60a5fa',
    'segres-dice_focal-full_train_loss': '#dc2626',
    'segres-dice_focal-full_val_loss': '#ef4444',
    'segres-dice_focal-patch_train_loss': '#991b1b',
    'segres-dice_focal-patch_val_loss': '#f87171',
    'unet-dice_ce-full_train_loss': '#16a34a',
    'unet-dice_ce-full_val_loss': '#22c55e',
    'unet-dice_ce-patch_train_loss': '#15803d',
    'unet-dice_ce-patch_val_loss': '#4ade80',
    'unet-dice_focal-full_train_loss': '#ca8a04',
    'unet-dice_focal-full_val_loss': '#eab308',
    'unet-dice_focal-patch_train_loss': '#a16207',
    'unet-dice_focal-patch_val_loss': '#facc15'
  };

  const getMetricsByView = () => {
    switch (viewMode) {
      case 'architectures':
        return [
          'segres-dice_ce-full_val_loss',
          'segres-dice_focal-full_val_loss',
          'unet-dice_ce-full_val_loss',
          'unet-dice_focal-full_val_loss'
        ];
      case 'loss_functions':
        return [
          'segres-dice_ce-full_val_loss',
          'segres-dice_focal-full_val_loss',
          'unet-dice_ce-full_val_loss',
          'unet-dice_focal-full_val_loss'
        ];
      case 'training_strategies':
        return [
          'segres-dice_ce-full_val_loss',
          'segres-dice_ce-patch_val_loss',
          'unet-dice_ce-full_val_loss',
          'unet-dice_ce-patch_val_loss'
        ];
      default:
        return selectedMetrics;
    }
  };

  const calculateFinalPerformance = () => {
    if (data.length === 0) return [];
    
    const lastEpoch = data[data.length - 1];
    const metrics = Object.keys(lastEpoch).filter(key => key.includes('val_loss'));
    
    return metrics.map(metric => ({
      metric: metric.replace('_val_loss', ''),
      finalLoss: lastEpoch[metric],
      architecture: metric.includes('segres') ? 'SegResNet' : 'U-Net',
      lossFunction: metric.includes('dice_ce') ? 'Dice+CE' : 'Dice+Focal',
      strategy: metric.includes('full') ? 'Full Image' : 'Patch-based'
    })).filter(item => item.finalLoss !== null);
  };

  const calculateOverfitting = () => {
    if (data.length === 0) return [];
    
    const lastEpoch = data[data.length - 1];
    const overfittingMetrics = [];
    
    const pairs = [
      ['segres-dice_ce-full_train_loss', 'segres-dice_ce-full_val_loss'],
      ['segres-dice_ce-patch_train_loss', 'segres-dice_ce-patch_val_loss'],
      ['segres-dice_focal-full_train_loss', 'segres-dice_focal-full_val_loss'],
      ['segres-dice_focal-patch_train_loss', 'segres-dice_focal-patch_val_loss'],
      ['unet-dice_ce-full_train_loss', 'unet-dice_ce-full_val_loss'],
      ['unet-dice_ce-patch_train_loss', 'unet-dice_ce-patch_val_loss'],
      ['unet-dice_focal-full_train_loss', 'unet-dice_focal-full_val_loss'],
      ['unet-dice_focal-patch_train_loss', 'unet-dice_focal-patch_val_loss']
    ];
    
    pairs.forEach(([trainKey, valKey]) => {
      const trainLoss = lastEpoch[trainKey];
      const valLoss = lastEpoch[valKey];
      
      if (trainLoss !== null && valLoss !== null) {
        overfittingMetrics.push({
          config: trainKey.replace('_train_loss', ''),
          trainLoss,
          valLoss,
          gap: valLoss - trainLoss,
          overfittingRatio: valLoss / trainLoss
        });
      }
    });
    
    return overfittingMetrics.sort((a, b) => b.gap - a.gap);
  };

  const finalPerformance = calculateFinalPerformance();
  const overfittingAnalysis = calculateOverfitting();
  const bestPerforming = finalPerformance.length > 0 ? 
    finalPerformance.reduce((best, current) => current.finalLoss < best.finalLoss ? current : best) : null;

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h1 className="text-2xl font-bold mb-4">Deep Learning Pipeline Loss Analysis</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-semibold text-blue-800">Architectures</h3>
            <p className="text-sm text-blue-600">SegResNet vs U-Net</p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-semibold text-green-800">Loss Functions</h3>
            <p className="text-sm text-green-600">Dice+CrossEntropy vs Dice+Focal</p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="font-semibold text-purple-800">Training Strategies</h3>
            <p className="text-sm text-purple-600">Full Image vs Patch-based</p>
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">View Mode:</label>
          <select 
            value={viewMode} 
            onChange={(e) => setViewMode(e.target.value)}
            className="border rounded px-3 py-2"
          >
            <option value="comparison">Custom Comparison</option>
            <option value="architectures">Architecture Comparison</option>
            <option value="loss_functions">Loss Function Comparison</option>
            <option value="training_strategies">Training Strategy Comparison</option>
          </select>
        </div>

        {viewMode === 'comparison' && (
          <div className="mb-4">
            <label className="block text-sm font-medium mb-2">Select Metrics:</label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {Object.keys(colors).map(metric => (
                <label key={metric} className="flex items-center space-x-2 text-xs">
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(metric)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedMetrics([...selectedMetrics, metric]);
                      } else {
                        setSelectedMetrics(selectedMetrics.filter(m => m !== metric));
                      }
                    }}
                  />
                  <span>{metric.replace(/_/g, ' ')}</span>
                </label>
              ))}
            </div>
          </div>
        )}

        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              {getMetricsByView().map(metric => (
                <Line 
                  key={metric}
                  type="monotone" 
                  dataKey={metric} 
                  stroke={colors[metric]}
                  strokeWidth={2}
                  dot={false}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4">Final Performance Ranking</h2>
          {bestPerforming && (
            <div className="mb-4 p-4 bg-green-50 rounded-lg">
              <h3 className="font-semibold text-green-800">Best Performer</h3>
              <p className="text-sm text-green-600">
                {bestPerforming.architecture} with {bestPerforming.lossFunction} ({bestPerforming.strategy})
              </p>
              <p className="text-lg font-bold text-green-800">
                Final Loss: {bestPerforming.finalLoss.toFixed(4)}
              </p>
            </div>
          )}
          
          <div className="space-y-2">
            {finalPerformance.sort((a, b) => a.finalLoss - b.finalLoss).map((item, index) => (
              <div key={item.metric} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                <div>
                  <span className="font-medium">{item.architecture}</span>
                  <span className="text-sm text-gray-600 ml-2">
                    {item.lossFunction} ({item.strategy})
                  </span>
                </div>
                <div className="text-right">
                  <span className="font-mono text-sm">{item.finalLoss.toFixed(4)}</span>
                  <span className="text-xs text-gray-500 ml-2">#{index + 1}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold mb-4">Overfitting Analysis</h2>
          <p className="text-sm text-gray-600 mb-4">
            Gap between validation and training loss (higher = more overfitting)
          </p>
          
          <div className="space-y-2">
            {overfittingAnalysis.map((item, index) => (
              <div key={item.config} className="p-3 bg-gray-50 rounded">
                <div className="flex justify-between items-center">
                  <span className="font-medium text-sm">
                    {item.config.replace(/-/g, ' ').replace(/_/g, ' ')}
                  </span>
                  <span className={`font-mono text-sm ${item.gap > 0.1 ? 'text-red-600' : 'text-green-600'}`}>
                    +{item.gap.toFixed(4)}
                  </span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  Train: {item.trainLoss.toFixed(4)} | Val: {item.valLoss.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-bold mb-4">Key Insights</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="p-4 bg-blue-50 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">Architecture Comparison</h3>
            <p className="text-sm text-blue-600">
              {finalPerformance.length > 0 && 
                (finalPerformance.filter(f => f.architecture === 'SegResNet').reduce((sum, f) => sum + f.finalLoss, 0) / 
                 finalPerformance.filter(f => f.architecture === 'SegResNet').length <
                 finalPerformance.filter(f => f.architecture === 'U-Net').reduce((sum, f) => sum + f.finalLoss, 0) / 
                 finalPerformance.filter(f => f.architecture === 'U-Net').length
                ? "SegResNet generally outperforms U-Net"
                : "U-Net generally outperforms SegResNet"
              )}
            </p>
          </div>
          
          <div className="p-4 bg-green-50 rounded-lg">
            <h3 className="font-semibold text-green-800 mb-2">Loss Function Impact</h3>
            <p className="text-sm text-green-600">
              {finalPerformance.length > 0 && 
                (finalPerformance.filter(f => f.lossFunction === 'Dice+Focal').reduce((sum, f) => sum + f.finalLoss, 0) / 
                 finalPerformance.filter(f => f.lossFunction === 'Dice+Focal').length <
                 finalPerformance.filter(f => f.lossFunction === 'Dice+CE').reduce((sum, f) => sum + f.finalLoss, 0) / 
                 finalPerformance.filter(f => f.lossFunction === 'Dice+CE').length
                ? "Dice+Focal loss generally performs better"
                : "Dice+CrossEntropy loss generally performs better"
              )}
            </p>
          </div>
          
          <div className="p-4 bg-purple-50 rounded-lg">
            <h3 className="font-semibold text-purple-800 mb-2">Training Strategy</h3>
            <p className="text-sm text-purple-600">
              {finalPerformance.length > 0 && 
                (finalPerformance.filter(f => f.strategy === 'Full Image').reduce((sum, f) => sum + f.finalLoss, 0) / 
                 finalPerformance.filter(f => f.strategy === 'Full Image').length <
                 finalPerformance.filter(f => f.strategy === 'Patch-based').reduce((sum, f) => sum + f.finalLoss, 0) / 
                 finalPerformance.filter(f => f.strategy === 'Patch-based').length
                ? "Full image training generally outperforms patch-based"
                : "Patch-based training generally outperforms full image"
              )}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LossAnalysis;