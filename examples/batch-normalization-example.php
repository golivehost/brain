<?php
/**
 * Batch normalization example
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\Layers\BatchNormalization;

// Create some sample data
$batchSize = 4;
$inputSize = 3;

$input = [
    [0.5, 1.2, -0.3],
    [0.1, 0.8, -0.5],
    [0.9, 1.5, -0.1],
    [0.3, 1.0, -0.4]
];

// Create a batch normalization layer
$batchNorm = new BatchNormalization($inputSize, [
    'epsilon' => 1e-5,
    'momentum' => 0.9
]);

// Set to training mode
$batchNorm->setTraining(true);

// Forward pass
echo "Input data:\n";
foreach ($input as $i => $sample) {
    echo "Sample $i: [" . implode(", ", array_map(function($x) { return round($x, 4); }, $sample)) . "]\n";
}

$output = $batchNorm->forward($input);

echo "\nNormalized output:\n";
foreach ($output as $i => $sample) {
    echo "Sample $i: [" . implode(", ", array_map(function($x) { return round($x, 4); }, $sample)) . "]\n";
}

// Get parameters
$params = $batchNorm->getParameters();

echo "\nBatch normalization parameters:\n";
echo "Gamma (scale): [" . implode(", ", array_map(function($x) { return round($x, 4); }, $params['gamma'])) . "]\n";
echo "Beta (shift): [" . implode(", ", array_map(function($x) { return round($x, 4); }, $params['beta'])) . "]\n";
echo "Running Mean: [" . implode(", ", array_map(function($x) { return round($x, 4); }, $params['runningMean'])) . "]\n";
echo "Running Variance: [" . implode(", ", array_map(function($x) { return round($x, 4); }, $params['runningVar'])) . "]\n";

// Switch to inference mode
$batchNorm->setTraining(false);

// Test with new data
$testInput = [
    [0.4, 1.1, -0.2],
    [0.2, 0.9, -0.4]
];

$testOutput = $batchNorm->forward($testInput);

echo "\nTest input:\n";
foreach ($testInput as $i => $sample) {
    echo "Sample $i: [" . implode(", ", array_map(function($x) { return round($x, 4); }, $sample)) . "]\n";
}

echo "\nNormalized test output:\n";
foreach ($testOutput as $i => $sample) {
    echo "Sample $i: [" . implode(", ", array_map(function($x) { return round($x, 4); }, $sample)) . "]\n";
}
