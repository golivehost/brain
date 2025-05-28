<?php
/**
 * Cross-validation example
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetworks\NeuralNetwork;
use GoLiveHost\Brain\Utilities\CrossValidation;

// Create a simple dataset (XOR problem)
$data = [
    ['input' => [0, 0], 'output' => [0]],
    ['input' => [0, 1], 'output' => [1]],
    ['input' => [1, 0], 'output' => [1]],
    ['input' => [1, 1], 'output' => [0]]
];

// Create a neural network
$nn = new NeuralNetwork([
    'hiddenLayers' => [3],
    'learningRate' => 0.3,
    'iterations' => 5000,
    'log' => true,
    'logPeriod' => 1000
]);

// Perform k-fold cross-validation
echo "Performing 2-fold cross-validation...\n";
$kFoldResults = CrossValidation::kFold($nn, $data, 2);

// Print results
echo "\nK-Fold Cross-Validation Results:\n";
echo "Average MSE: " . $kFoldResults['averageMetrics']['mse'] . "\n";
echo "Average Accuracy: " . $kFoldResults['averageMetrics']['accuracy'] . "\n";
echo "Average Training Time: " . $kFoldResults['averageMetrics']['trainTime'] . " seconds\n";

// Perform train-test split
echo "\nPerforming train-test split...\n";
$split = CrossValidation::trainTestSplit($data, 0.5);

// Train on training set
$nn = new NeuralNetwork([
    'hiddenLayers' => [3],
    'learningRate' => 0.3,
    'iterations' => 5000
]);

$trainStats = $nn->train($split['train']);
echo "Training completed in {$trainStats['iterations']} iterations with error {$trainStats['error']}\n";

// Evaluate on test set
$evalStats = CrossValidation::evaluateModel($nn, $split['test']);
echo "Test MSE: " . $evalStats['mse'] . "\n";
echo "Test Accuracy: " . $evalStats['accuracy'] . "\n";

// Print predictions for test set
echo "\nPredictions for test set:\n";
foreach ($split['test'] as $item) {
    $output = $nn->run($item['input']);
    echo "Input: [" . implode(", ", $item['input']) . "] => ";
    echo "Predicted: " . round($output[0]) . ", Actual: " . $item['output'][0] . "\n";
}
