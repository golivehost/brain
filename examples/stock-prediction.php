<?php
/**
 * Stock price prediction example using the GoLiveHost Brain library
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\Brain;
use GoLiveHost\Brain\Utilities\Normalizer;

// This example demonstrates how to use LSTM for stock price prediction

// Function to generate synthetic stock data
function generateStockData($days = 1000, $volatility = 0.02) {
    $prices = [100]; // Start with $100
    
    for ($i = 1; $i < $days; $i++) {
        // Random walk with drift and some seasonality
        $drift = 0.0001; // Slight upward trend
        $seasonality = 0.1 * sin(2 * M_PI * $i / 252); // Yearly cycle (252 trading days)
        $random = $volatility * (mt_rand() / mt_getrandmax() * 2 - 1);
        
        $change = $drift + $seasonality + $random;
        $prices[] = $prices[$i - 1] * (1 + $change);
    }
    
    return $prices;
}

// Function to prepare training data for time series prediction
function prepareTimeSeriesData($prices, $lookback = 10, $horizon = 5) {
    $sequences = [];
    $normalizer = new Normalizer();
    
    // Prepare data for normalization
    $data = [];
    for ($i = 0; $i < count($prices); $i++) {
        $data[] = [
            'input' => [$prices[$i]],
            'output' => [$prices[$i]]
        ];
    }
    
    // Fit normalizer
    $normalizer->fit($data);
    
    // Create sequences
    for ($i = $lookback; $i < count($prices) - $horizon; $i++) {
        $inputSeq = [];
        $outputSeq = [];
        
        // Input sequence (lookback period)
        for ($j = 0; $j < $lookback; $j++) {
            $price = $prices[$i - $lookback + $j];
            $normalized = $normalizer->transformInput([$price]);
            $inputSeq[] = $normalized;
        }
        
        // Output sequence (forecast horizon)
        for ($j = 0; $j < $horizon; $j++) {
            $price = $prices[$i + $j];
            $normalized = $normalizer->transformInput([$price]);
            $outputSeq[] = $normalized;
        }
        
        $sequences[] = [
            'input' => $inputSeq,
            'output' => $outputSeq
        ];
    }
    
    return [
        'sequences' => $sequences,
        'normalizer' => $normalizer
    ];
}

// Function to evaluate predictions
function evaluatePredictions($actual, $predicted) {
    $n = count($actual);
    $mse = 0;
    $mae = 0;
    $mape = 0;
    
    for ($i = 0; $i < $n; $i++) {
        $mse += pow($actual[$i] - $predicted[$i], 2);
        $mae += abs($actual[$i] - $predicted[$i]);
        $mape += abs(($actual[$i] - $predicted[$i]) / $actual[$i]);
    }
    
    return [
        'mse' => $mse / $n,
        'rmse' => sqrt($mse / $n),
        'mae' => $mae / $n,
        'mape' => ($mape / $n) * 100
    ];
}

// Generate synthetic stock data
echo "Generating synthetic stock data...\n";
$stockPrices = generateStockData();

echo "Number of data points: " . count($stockPrices) . "\n";
echo "First price: $" . number_format($stockPrices[0], 2) . "\n";
echo "Last price: $" . number_format(end($stockPrices), 2) . "\n\n";

// Prepare training data
echo "Preparing training data...\n";
$lookback = 20; // Use 20 days of history
$horizon = 5;   // Predict 5 days ahead
$data = prepareTimeSeriesData($stockPrices, $lookback, $horizon);

// Split into training and test sets
$testSize = 50;
$testSequences = array_splice($data['sequences'], -$testSize);
$trainingSequences = $data['sequences'];

echo "Training sequences: " . count($trainingSequences) . "\n";
echo "Test sequences: " . count($testSequences) . "\n\n";

// Create LSTM
$lstm = Brain::lstm([
    'inputSize' => 1,
    'hiddenLayers' => [50, 25],
    'outputSize' => 1,
    'activation' => 'tanh',
    'learningRate' => 0.01,
    'iterations' => 100,
    'batchSize' => 32,
    'log' => true,
    'logPeriod' => 10
]);

// Train LSTM
echo "Training LSTM for stock price prediction...\n";
$result = $lstm->train($trainingSequences);
echo "Training completed with error {$result['error']}\n\n";

// Test LSTM
echo "Testing LSTM on test set...\n";
$normalizer = $data['normalizer'];
$allActual = [];
$allPredicted = [];

foreach ($testSequences as $i => $sequence) {
    $input = $sequence['input'];
    $actualOutput = $sequence['output'];
    
    $prediction = $lstm->run($input);
    
    // Denormalize for evaluation
    $actualDenormalized = [];
    $predictedDenormalized = [];
    
    for ($j = 0; $j < count($actualOutput); $j++) {
        $actualDenormalized[] = $normalizer->inverseTransformOutput($actualOutput[$j])[0];
        $predictedDenormalized[] = $normalizer->inverseTransformOutput($prediction[$j])[0];
    }
    
    $allActual = array_merge($allActual, $actualDenormalized);
    $allPredicted = array_merge($allPredicted, $predictedDenormalized);
    
    // Print some examples
    if ($i < 3) {
        echo "Example " . ($i + 1) . ":\n";
        echo "Actual: [" . implode(", ", array_map(function($p) { 
            return "$" . number_format($p, 2); 
        }, $actualDenormalized)) . "]\n";
        
        echo "Predicted: [" . implode(", ", array_map(function($p) { 
            return "$" . number_format($p, 2); 
        }, $predictedDenormalized)) . "]\n\n";
    }
}

// Evaluate overall performance
$metrics = evaluatePredictions($allActual, $allPredicted);

echo "Performance metrics:\n";
echo "RMSE: $" . number_format($metrics['rmse'], 2) . "\n";
echo "MAE: $" . number_format($metrics['mae'], 2) . "\n";
echo "MAPE: " . number_format($metrics['mape'], 2) . "%\n\n";

// Save the model
$json = $lstm->toJSON();
file_put_contents(__DIR__ . '/stock-prediction-model.json', $json);
echo "Model saved to stock-prediction-model.json\n";
