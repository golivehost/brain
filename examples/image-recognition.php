<?php
/**
 * Image recognition example using the GoLiveHost Brain library
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\Brain;

// This example demonstrates how to use the neural network for simple image recognition
// We'll create a dataset of small 10x10 "images" representing simple shapes

// Function to create a training dataset of shapes
function createShapesDataset($numSamples = 100) {
    $dataset = [];
    $shapes = ['circle', 'square', 'triangle'];
    
    for ($i = 0; $i < $numSamples; $i++) {
        $shape = $shapes[array_rand($shapes)];
        $image = createShapeImage($shape, 10, 10);
        
        // One-hot encode the output
        $output = [0, 0, 0];
        switch ($shape) {
            case 'circle':
                $output[0] = 1;
                break;
            case 'square':
                $output[1] = 1;
                break;
            case 'triangle':
                $output[2] = 1;
                break;
        }
        
        $dataset[] = [
            'input' => $image,
            'output' => $output
        ];
    }
    
    return $dataset;
}

// Function to create a simple shape image
function createShapeImage($shape, $width, $height) {
    $image = array_fill(0, $width * $height, 0);
    
    switch ($shape) {
        case 'circle':
            $centerX = $width / 2;
            $centerY = $height / 2;
            $radius = min($width, $height) * 0.4;
            
            for ($y = 0; $y < $height; $y++) {
                for ($x = 0; $x < $width; $x++) {
                    $distance = sqrt(pow($x - $centerX, 2) + pow($y - $centerY, 2));
                    if ($distance <= $radius) {
                        $image[$y * $width + $x] = 1;
                    }
                }
            }
            break;
            
        case 'square':
            $margin = max(1, floor(min($width, $height) * 0.2));
            
            for ($y = $margin; $y < $height - $margin; $y++) {
                for ($x = $margin; $x < $width - $margin; $x++) {
                    $image[$y * $width + $x] = 1;
                }
            }
            break;
            
        case 'triangle':
            $baseY = $height - floor($height * 0.2);
            $topX = $width / 2;
            $topY = floor($height * 0.2);
            $leftX = floor($width * 0.2);
            $rightX = $width - floor($width * 0.2);
            
            for ($y = 0; $y < $height; $y++) {
                for ($x = 0; $x < $width; $x++) {
                    // Check if point is inside triangle using barycentric coordinates
                    $alpha = ((($baseY - $rightX) * ($x - $rightX)) + (($rightX - $topX) * ($y - $baseY))) /
                             ((($baseY - $rightX) * ($leftX - $rightX)) + (($rightX - $topX) * ($topY - $baseY)));
                    
                    $beta = ((($baseY - $topX) * ($x - $topX)) + (($topX - $leftX) * ($y - $baseY))) /
                            ((($baseY - $topX) * ($rightX - $topX)) + (($topX - $leftX) * ($topY - $baseY)));
                    
                    $gamma = 1.0 - $alpha - $beta;
                    
                    if ($alpha >= 0 && $beta >= 0 && $gamma >= 0) {
                        $image[$y * $width + $x] = 1;
                    }
                }
            }
            break;
    }
    
    // Add some noise
    for ($i = 0; $i < $width * $height; $i++) {
        if (mt_rand(0, 100) < 5) { // 5% chance of noise
            $image[$i] = $image[$i] ? 0 : 1;
        }
    }
    
    return $image;
}

// Function to display an image in the console
function displayImage($image, $width, $height) {
    for ($y = 0; $y < $height; $y++) {
        for ($x = 0; $x < $width; $x++) {
            echo $image[$y * $width + $x] ? '█' : ' ';
        }
        echo "\n";
    }
}

// Create dataset
echo "Creating dataset of shapes...\n";
$dataset = createShapesDataset(300);
shuffle($dataset);

// Split into training and test sets
$testSet = array_splice($dataset, 0, 50);
$trainingSet = $dataset;

echo "Training set size: " . count($trainingSet) . "\n";
echo "Test set size: " . count($testSet) . "\n\n";

// Create neural network
$nn = Brain::neuralNetwork([
    'inputSize' => 100, // 10x10 image
    'hiddenLayers' => [50, 25],
    'outputSize' => 3, // circle, square, triangle
    'activation' => 'sigmoid',
    'learningRate' => 0.1,
    'iterations' => 1000,
    'errorThresh' => 0.005,
    'log' => true,
    'logPeriod' => 100
]);

// Train the network
echo "Training neural network...\n";
$result = $nn->train($trainingSet);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n\n";

// Test the network
echo "Testing neural network...\n";
$correct = 0;
$shapeNames = ['circle', 'square', 'triangle'];

foreach ($testSet as $i => $test) {
    $output = $nn->run($test['input']);
    $predictedIndex = array_search(max($output), $output);
    $actualIndex = array_search(1, $test['output']);
    
    $isCorrect = $predictedIndex === $actualIndex;
    if ($isCorrect) {
        $correct++;
    }
    
    if ($i < 5 || !$isCorrect) { // Show first 5 and all incorrect predictions
        echo "Example " . ($i + 1) . ":\n";
        displayImage($test['input'], 10, 10);
        echo "Actual: " . $shapeNames[$actualIndex] . "\n";
        echo "Predicted: " . $shapeNames[$predictedIndex] . 
             " (" . implode(", ", array_map(function($v) { return number_format($v, 2); }, $output)) . ")\n";
        echo $isCorrect ? "CORRECT\n\n" : "INCORRECT\n\n";
    }
}

$accuracy = $correct / count($testSet) * 100;
echo "Accuracy: " . number_format($accuracy, 2) . "%\n";

// Save the model
$json = $nn->toJSON();
file_put_contents(__DIR__ . '/shape-recognition-model.json', $json);
echo "Model saved to shape-recognition-model.json\n";
