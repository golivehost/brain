<?php
/**
 * Optimizer example
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\Optimizers\Adam;
use GoLiveHost\Brain\Optimizers\RMSprop;
use GoLiveHost\Brain\Optimizers\AdaGrad;

// Define a simple function to optimize: f(x, y) = x^2 + y^2
function computeFunction($x, $y) {
    return pow($x, 2) + pow($y, 2);
}

// Compute gradients: df/dx = 2x, df/dy = 2y
function computeGradients($x, $y) {
    return [
        'x' => 2 * $x,
        'y' => 2 * $y
    ];
}

// Initialize parameters
$parameters = [
    'x' => 5.0,
    'y' => 5.0
];

// Create optimizers
$adam = new Adam([
    'learningRate' => 0.1,
    'beta1' => 0.9,
    'beta2' => 0.999,
    'epsilon' => 1e-8
]);

$rmsprop = new RMSprop([
    'learningRate' => 0.01,
    'decay' => 0.9,
    'epsilon' => 1e-8
]);

$adagrad = new AdaGrad([
    'learningRate' => 0.1,
    'epsilon' => 1e-8
]);

// Initialize optimizers
$adam->initialize($parameters);
$rmsprop->initialize($parameters);
$adagrad->initialize($parameters);

// Optimization loop
$numIterations = 100;

echo "Starting optimization...\n";
echo "Initial parameters: x = {$parameters['x']}, y = {$parameters['y']}\n";
echo "Initial function value: " . computeFunction($parameters['x'], $parameters['y']) . "\n\n";

echo "Optimizing with Adam:\n";
$adamParams = $parameters;
for ($i = 0; $i < $numIterations; $i++) {
    $gradients = computeGradients($adamParams['x'], $adamParams['y']);
    $adam->update($adamParams, ['x' => $gradients['x'], 'y' => $gradients['y']]);
    
    if ($i % 10 === 0) {
        $value = computeFunction($adamParams['x'], $adamParams['y']);
        echo "Iteration $i: x = " . round($adamParams['x'], 4) . ", y = " . round($adamParams['y'], 4) . ", f(x,y) = " . round($value, 4) . "\n";
    }
}

echo "\nOptimizing with RMSprop:\n";
$rmspropParams = $parameters;
for ($i = 0; $i < $numIterations; $i++) {
    $gradients = computeGradients($rmspropParams['x'], $rmspropParams['y']);
    $rmsprop->update($rmspropParams, ['x' => $gradients['x'], 'y' => $gradients['y']]);
    
    if ($i % 10 === 0) {
        $value = computeFunction($rmspropParams['x'], $rmspropParams['y']);
        echo "Iteration $i: x = " . round($rmspropParams['x'], 4) . ", y = " . round($rmspropParams['y'], 4) . ", f(x,y) = " . round($value, 4) . "\n";
    }
}

echo "\nOptimizing with AdaGrad:\n";
$adagradParams = $parameters;
for ($i = 0; $i < $numIterations; $i++) {
    $gradients = computeGradients($adagradParams['x'], $adagradParams['y']);
    $adagrad->update($adagradParams, ['x' => $gradients['x'], 'y' => $gradients['y']]);
    
    if ($i % 10 === 0) {
        $value = computeFunction($adagradParams['x'], $adagradParams['y']);
        echo "Iteration $i: x = " . round($adagradParams['x'], 4) . ", y = " . round($adagradParams['y'], 4) . ", f(x,y) = " . round($value, 4) . "\n";
    }
}

echo "\nFinal results:\n";
echo "Adam: x = " . round($adamParams['x'], 4) . ", y = " . round($adamParams['y'], 4) . ", f(x,y) = " . round(computeFunction($adamParams['x'], $adamParams['y']), 4) . "\n";
echo "RMSprop: x = " . round($rmspropParams['x'], 4) . ", y = " . round($rmspropParams['y'], 4) . ", f(x,y) = " . round(computeFunction($rmspropParams['x'], $rmspropParams['y']), 4) . "\n";
echo "AdaGrad: x = " . round($adagradParams['x'], 4) . ", y = " . round($adagradParams['y'], 4) . ", f(x,y) = " . round(computeFunction($adagradParams['x'], $adagradParams['y']), 4) . "\n";
