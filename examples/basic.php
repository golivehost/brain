<?php
/**
 * Basic example of using the GoLiveHost Brain library
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetwork;

// XOR problem
$trainingData = [
    ['input' => [0, 0], 'output' => [0]],
    ['input' => [0, 1], 'output' => [1]],
    ['input' => [1, 0], 'output' => [1]],
    ['input' => [1, 1], 'output' => [0]]
];

// Create a neural network
$net = new NeuralNetwork([
    'hiddenLayers' => [3],
    'learningRate' => 0.3,
    'iterations' => 20000,
]);

// Train the network
echo "Training the network...\n";
$result = $net->train($trainingData);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n\n";

// Test the network
echo "Testing the network:\n";
foreach ($trainingData as $data) {
    $output = $net->run($data['input']);
    echo "Input: [" . implode(", ", $data['input']) . "] => Output: " . round($output[0]) . 
         " (Raw: " . number_format($output[0], 4) . ")\n";
}

// Save the trained model
$json = $net->toJSON();
file_put_contents(__DIR__ . '/xor-model.json', $json);
echo "\nModel saved to xor-model.json\n";

// Load the model
echo "\nLoading the model from JSON...\n";
$loadedNet = NeuralNetwork::fromJSON($json);

// Test the loaded model
echo "Testing the loaded model:\n";
foreach ($trainingData as $data) {
    $output = $loadedNet->run($data['input']);
    echo "Input: [" . implode(", ", $data['input']) . "] => Output: " . round($output[0]) . 
         " (Raw: " . number_format($output[0], 4) . ")\n";
}
