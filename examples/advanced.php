<?php
/**
 * Advanced example of using the GoLiveHost Brain library
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetwork;
use GoLiveHost\Brain\RNN;

// Example 1: Character recognition
echo "Example 1: Character Recognition\n";
echo "--------------------------------\n";

// Define training data for character recognition (simplified)
// Each character is represented as a 5x5 grid (flattened to 25 inputs)
$trainingData = [
    // Letter 'A'
    [
        'input' => [
            0, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 1,
            1, 0, 0, 0, 1,
            1, 0, 0, 0, 1
        ],
        'output' => [1, 0, 0] // One-hot encoding: A
    ],
    // Letter 'B'
    [
        'input' => [
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 0,
            1, 0, 0, 0, 1,
            1, 1, 1, 1, 0
        ],
        'output' => [0, 1, 0] // One-hot encoding: B
    ],
    // Letter 'C'
    [
        'input' => [
            0, 1, 1, 1, 1,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            0, 1, 1, 1, 1
        ],
        'output' => [0, 0, 1] // One-hot encoding: C
    ]
];

// Create and train the neural network
$net = new NeuralNetwork([
    'inputSize' => 25,
    'hiddenLayers' => [10],
    'outputSize' => 3,
    'learningRate' => 0.1,
    'iterations' => 10000,
]);

echo "Training character recognition network...\n";
$result = $net->train($trainingData);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n\n";

// Test with a slightly modified 'A'
$testA = [
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 0  // Slight modification
];

$output = $net->run($testA);
echo "Testing with modified 'A':\n";
echo "Output: [" . implode(", ", array_map(function($val) { 
    return number_format($val, 4); 
}, $output)) . "]\n";

$prediction = array_search(max($output), $output);
$letters = ['A', 'B', 'C'];
echo "Predicted letter: " . $letters[$prediction] . "\n\n";

// Example 2: Time Series Prediction with RNN
echo "Example 2: Time Series Prediction with RNN\n";
echo "----------------------------------------\n";

// Generate sine wave data
$data = [];
for ($i = 0; $i < 100; $i++) {
    $data[] = sin($i * 0.1);
}

// Prepare sequences for RNN (predict next value based on 5 previous values)
$sequences = [];
for ($i = 5; $i < count($data) - 1; $i++) {
    $input = [];
    for ($j = 0; $j < 5; $j++) {
        $input[] = [$data[$i - 5 + $j]];
    }
    $sequences[] = [
        'input' => $input,
        'output' => [[$data[$i]]]
    ];
}

// Create and train RNN
$rnn = new RNN([
    'inputSize' => 1,
    'hiddenLayers' => [10],
    'outputSize' => 1,
    'learningRate' => 0.01,
    'iterations' => 1000,
]);

echo "Training RNN for time series prediction...\n";
$result = $rnn->train($sequences);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n\n";

// Test RNN with a sequence
$testSequence = [];
for ($j = 0; $j < 5; $j++) {
    $testSequence[] = [$data[90 + $j]];
}

$prediction = $rnn->run($testSequence);
echo "Last 5 values: [" . implode(", ", array_map(function($val) { 
    return number_format($val[0], 4); 
}, $testSequence)) . "]\n";
echo "Actual next value: " . number_format($data[95], 4) . "\n";
echo "Predicted next value: " . number_format($prediction[count($prediction)-1][0], 4) . "\n";

// Save the RNN model
$json = $rnn->toJSON();
file_put_contents(__DIR__ . '/rnn-model.json', $json);
echo "\nRNN model saved to rnn-model.json\n";
