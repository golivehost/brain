<?php
/**
 * Advanced examples of using the GoLiveHost Brain library
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\Brain;

// Example 1: Advanced Neural Network with Normalization and Dropout
echo "Example 1: Advanced Neural Network with Normalization and Dropout\n";
echo "------------------------------------------------------------\n";

// Generate some non-linear data
$trainingData = [];
for ($i = 0; $i < 1000; $i++) {
    $x1 = mt_rand() / mt_getrandmax() * 10 - 5;
    $x2 = mt_rand() / mt_getrandmax() * 10 - 5;
    $y = sin($x1) * cos($x2) + 0.1 * (mt_rand() / mt_getrandmax() - 0.5); // Add some noise
    
    $trainingData[] = [
        'input' => [$x1, $x2],
        'output' => [$y]
    ];
}

// Split into training and test sets
$testData = array_splice($trainingData, 800);

// Create a neural network with advanced options
$net = Brain::neuralNetwork([
    'hiddenLayers' => [20, 10],
    'activation' => 'tanh',
    'learningRate' => 0.01,
    'momentum' => 0.1,
    'dropout' => 0.2,
    'iterations' => 1000,
    'batchSize' => 32,
    'praxis' => 'adam',
    'normalize' => true,
    'log' => true,
    'logPeriod' => 100
]);

echo "Training neural network...\n";
$result = $net->train($trainingData);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n";

// Test the network
$testResult = $net->test($testData);
echo "Test MSE: {$testResult['error']}\n\n";

// Example 2: LSTM for Time Series Prediction
echo "Example 2: LSTM for Time Series Prediction\n";
echo "-------------------------------------\n";

// Generate sine wave data
$data = [];
for ($i = 0; $i < 1000; $i++) {
    $data[] = sin($i * 0.1);
}

// Prepare sequences for LSTM (predict next 5 values based on previous 10 values)
$sequences = [];
for ($i = 10; $i < count($data) - 5; $i++) {
    $input = [];
    $output = [];
    
    for ($j = 0; $j < 10; $j++) {
        $input[] = [$data[$i - 10 + $j]];
    }
    
    for ($j = 0; $j < 5; $j++) {
        $output[] = [$data[$i + $j]];
    }
    
    $sequences[] = [
        'input' => $input,
        'output' => $output
    ];
}

// Split into training and test sets
$testSequences = array_splice($sequences, 800);

// Create and train LSTM
$lstm = Brain::lstm([
    'inputSize' => 1,
    'hiddenLayers' => [20],
    'outputSize' => 1,
    'learningRate' => 0.01,
    'iterations' => 100,
    'batchSize' => 16,
    'praxis' => 'adam',
    'log' => true,
    'logPeriod' => 10
]);

echo "Training LSTM for time series prediction...\n";
$result = $lstm->train($sequences);
echo "Training completed with error {$result['error']}\n";

// Test LSTM with a sequence
$testSequence = $testSequences[0]['input'];
$actualNext = $testSequences[0]['output'];
$prediction = $lstm->run($testSequence);

echo "Last input values: [" . implode(", ", array_map(function($val) { 
    return number_format($val[0], 4); 
}, array_slice($testSequence, -3))) . "]\n";

echo "Actual next values: [" . implode(", ", array_map(function($val) { 
    return number_format($val[0], 4); 
}, $actualNext)) . "]\n";

echo "Predicted next values: [" . implode(", ", array_map(function($val) { 
    return number_format($val[0], 4); 
}, $prediction)) . "]\n\n";

// Example 3: Liquid State Machine for Classification
echo "Example 3: Liquid State Machine for Classification\n";
echo "--------------------------------------------\n";

// Generate some classification data (3 classes in 2D space)
$classData = [];
$classes = 3;
$pointsPerClass = 100;

for ($c = 0; $c < $classes; $c++) {
    $centerX = cos(2 * M_PI * $c / $classes) * 3;
    $centerY = sin(2 * M_PI * $c / $classes) * 3;
    
    for ($i = 0; $i < $pointsPerClass; $i++) {
        $x = $centerX + (mt_rand() / mt_getrandmax() * 2 - 1);
        $y = $centerY + (mt_rand() / mt_getrandmax() * 2 - 1);
        
        $output = array_fill(0, $classes, 0);
        $output[$c] = 1;
        
        $classData[] = [
            'input' => [[$x, $y]],
            'output' => [$output]
        ];
    }
}

// Split into training and test sets
shuffle($classData);
$testClassData = array_splice($classData, 200);

// Create and train LSM
$lsm = Brain::liquidStateMachine([
    'inputSize' => 2,
    'reservoirSize' => 100,
    'outputSize' => $classes,
    'connectivity' => 0.1,
    'spectralRadius' => 0.9,
    'leakingRate' => 0.3,
    'regularization' => 0.001
]);

echo "Training Liquid State Machine for classification...\n";
$result = $lsm->train($classData);
echo "Training completed with error {$result['error']}\n";

// Test LSM
$correct = 0;
$total = count($testClassData);

foreach ($testClassData as $item) {
    $prediction = $lsm->run($item['input']);
    $predictedClass = array_search(max($prediction[0]), $prediction[0]);
    $actualClass = array_search(1, $item['output'][0]);
    
    if ($predictedClass === $actualClass) {
        $correct++;
    }
}

$accuracy = $correct / $total;
echo "Classification accuracy: " . number_format($accuracy * 100, 2) . "%\n";

// Save models
$netJson = $net->toJSON();
file_put_contents(__DIR__ . '/advanced_nn.json', $netJson);

$lstmJson = $lstm->toJSON();
file_put_contents(__DIR__ . '/advanced_lstm.json', $lstmJson);

$lsmJson = $lsm->toJSON();
file_put_contents(__DIR__ . '/advanced_lsm.json', $lsmJson);

echo "\nAll models saved to JSON files\n";
