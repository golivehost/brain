<?php
/**
 * Model checkpoint example
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetworks\NeuralNetwork;
use GoLiveHost\Brain\Utilities\ModelCheckpoint;

// Create a simple dataset (XOR problem)
$data = [
    ['input' => [0, 0], 'output' => [0]],
    ['input' => [0, 1], 'output' => [1]],
    ['input' => [1, 0], 'output' => [1]],
    ['input' => [1, 1], 'output' => [0]]
];

// Create a neural network with better initialization for XOR
$nn = new NeuralNetwork([
    'hiddenLayers' => [4],
    'learningRate' => 0.1,
    'activation' => 'tanh',  // tanh works better for XOR
    'iterations' => 1000,
    'log' => true,
    'logPeriod' => 100
]);

// Create a checkpoint manager
$checkpointManager = new ModelCheckpoint([
    'directory' => __DIR__ . '/checkpoints',
    'filePrefix' => 'xor_model',
    'saveFrequency' => 1000,
    'saveOnlyBest' => false,
    'maxCheckpoints' => 3
]);

// Custom training loop with checkpoints
$maxIterations = 5000;
$errorThresh = 0.001;

$error = 1;
$iteration = 0;

echo "Starting training with checkpoints...\n";

// Create checkpoints directory if it doesn't exist
if (!is_dir(__DIR__ . '/checkpoints')) {
    mkdir(__DIR__ . '/checkpoints', 0755, true);
    echo "Created checkpoints directory\n";
}

// Check if directory is writable
if (!is_writable(__DIR__ . '/checkpoints')) {
    echo "Warning: Checkpoints directory is not writable!\n";
}

// First train the network to solve XOR properly
echo "Pre-training network to solve XOR...\n";
$preTrainStats = $nn->train($data, [
    'iterations' => 2000,
    'errorThresh' => 0.01
]);

echo "Pre-training completed with error: {$preTrainStats['error']}\n";
echo "Testing pre-trained network:\n";
foreach ($data as $item) {
    $output = $nn->run($item['input']);
    echo "Input: [" . implode(", ", $item['input']) . "] => ";
    echo "Output: " . round($output[0]) . ", Expected: " . $item['output'][0] . "\n";
}

// Now continue training with checkpoints
echo "\nContinuing training with checkpoints...\n";
while ($error > $errorThresh && $iteration < $maxIterations) {
    // Train for a batch
    $trainStats = $nn->train($data, [
        'iterations' => 1000,
        'errorThresh' => 0
    ]);
    
    $error = $trainStats['error'];
    $iteration += 1000;
    
    echo "Iteration: $iteration, Error: $error\n";
    
    // Save checkpoint with error handling
    try {
        $checkpointPath = $checkpointManager->save($nn, ['error' => $error], $iteration);
        
        if ($checkpointPath) {
            echo "Saved checkpoint to: $checkpointPath (Size: " . filesize($checkpointPath) . " bytes)\n";
            
            // Verify the saved file
            $jsonContent = file_get_contents($checkpointPath);
            $jsonData = json_decode($jsonContent, true);
            if (json_last_error() !== JSON_ERROR_NONE) {
                echo "Warning: Saved checkpoint contains invalid JSON: " . json_last_error_msg() . "\n";
            } else {
                echo "Checkpoint JSON validation: OK\n";
            }
        }
    } catch (\Exception $e) {
        echo "Error saving checkpoint: " . $e->getMessage() . "\n";
    }
    
    if ($error <= $errorThresh) {
        echo "Error threshold reached. Stopping training.\n";
        break;
    }
}

// List available checkpoints
$checkpoints = $checkpointManager->getCheckpoints();
echo "\nAvailable checkpoints:\n";
foreach ($checkpoints as $checkpoint) {
    echo "- {$checkpoint['filename']} (Epoch: {$checkpoint['epoch']}, Error: {$checkpoint['value']}, Size: {$checkpoint['size']} bytes)\n";
}

// Load the best checkpoint
try {
    $bestModel = $checkpointManager->loadBest(NeuralNetwork::class);
    echo "\nLoaded best model with error: {$bestModel->getTrainStats()['error']}\n";
    
    // Test the loaded model
    echo "\nTesting loaded model:\n";
    foreach ($data as $item) {
        $output = $bestModel->run($item['input']);
        echo "Input: [" . implode(", ", $item['input']) . "] => ";
        echo "Output: " . round($output[0]) . ", Expected: " . $item['output'][0] . "\n";
    }
} catch (\Exception $e) {
    echo "\nError loading best model: " . $e->getMessage() . "\n";
    echo "This can happen if no checkpoints were saved or if there was an issue with the saved files.\n";
    
    // Continue with the original model
    echo "\nTesting original model instead:\n";
    foreach ($data as $item) {
        $output = $nn->run($item['input']);
        echo "Input: [" . implode(", ", $item['input']) . "] => ";
        echo "Output: " . round($output[0]) . ", Expected: " . $item['output'][0] . "\n";
    }
}

// Clean up checkpoints (optional)
// $checkpointManager->clearCheckpoints();
// echo "\nCheckpoints cleared.\n";
