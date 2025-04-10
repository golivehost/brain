# GoLiveHost Brain

A PHP neural network library

Developed by: Go Live Web Solutions (golive.host)
Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)

## Installation

\`\`\`bash
composer require golivehost/brain
\`\`\`

## Features

- Multiple neural network types:
  - Feedforward Neural Network
  - Long Short-Term Memory (LSTM)
  - Liquid State Machine (LSM)
- Advanced training options:
  - Batch training
  - Learning rate decay
  - Momentum
  - Dropout regularization
  - Early stopping
- Multiple optimization algorithms:
  - Stochastic Gradient Descent (SGD)
  - Adam optimizer
- Multiple activation functions:
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
  - Softmax
  - Linear
- Data preprocessing:
  - Normalization
  - Data formatting
- Matrix and tensor operations
- Model serialization and export
- Compatible with PHP 8.0+

## Basic Usage

\`\`\`php
<?php
require_once 'vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetworks\NeuralNetwork;

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
]);

// Train the network
$result = $net->train($trainingData);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n";

// Test the network
foreach ($trainingData as $data) {
    $output = $net->run($data['input']);
    echo "Input: [" . implode(", ", $data['input']) . "] => Output: " . round($output[0]) . "\n";
}

// Save the trained model
$json = $net->toJSON();
file_put_contents('xor-model.json', $json);

// Load the model
$loadedNet = NeuralNetwork::fromJSON($json);
\`\`\`

## Advanced Usage

### LSTM for Time Series Prediction

\`\`\`php
<?php
use GoLiveHost\Brain\NeuralNetworks\LSTM;

// Prepare sequences for LSTM
$sequences = [
    [
        'input' => [[0.1], [0.2], [0.3], [0.4], [0.5]],
        'output' => [[0.6], [0.7], [0.8]]
    ],
    // More sequences...
];

// Create and train LSTM
$lstm = new LSTM([
