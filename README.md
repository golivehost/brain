# GoLiveHost Brain

A comprehensive PHP neural network library for machine learning and artificial intelligence applications.

**Developed by:** Go Live Web Solutions ([golive.host](https://golive.host))  
**Author:** Shubhdeep Singh ([GitHub.com/shubhdeepdev](https://github.com/shubhdeepdev))

## Table of Contents

- [Installation](#installation)
- [Features](#features)
- [Quick Start](#quick-start)
- [Neural Network Types](#neural-network-types)
- [Advanced Features](#advanced-features)
- [Configuration Options](#configuration-options)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Installation

Install via Composer:

\`\`\`bash
composer require golivehost/brain
\`\`\`

## Features

### Neural Network Architectures
- **Feedforward Neural Network** - Traditional multi-layer perceptron
- **Recurrent Neural Network (RNN)** - For sequential data processing
- **Long Short-Term Memory (LSTM)** - Advanced RNN with memory cells
- **Gated Recurrent Unit (GRU)** - Simplified LSTM variant
- **Liquid State Machine (LSM)** - Reservoir computing for complex temporal patterns

### Training Features
- **Multiple Optimization Algorithms**
  - Stochastic Gradient Descent (SGD) with momentum
  - Adam optimizer
  - RMSprop optimizer
  - AdaGrad optimizer
- **Advanced Training Options**
  - Batch training with configurable batch sizes
  - Learning rate decay
  - Dropout regularization
  - Early stopping with patience
  - Gradient clipping
  - Model checkpointing
- **Activation Functions**
  - Sigmoid
  - Tanh
  - ReLU
  - Leaky ReLU
  - Softmax
  - Linear

### Data Processing
- **Preprocessing Utilities**
  - Data normalization
  - Data formatting for sequences
  - One-hot encoding
- **Evaluation Tools**
  - Cross-validation (k-fold, stratified k-fold, leave-one-out)
  - Train-test split
  - Multiple evaluation metrics (MSE, RMSE, R², accuracy, precision, recall, F1)
- **Model Management**
  - Model serialization (JSON)
  - Model export to standalone PHP
  - Model checkpointing during training

### Additional Features
- **Batch Normalization** - For improved training stability
- **Matrix and Tensor Operations** - Comprehensive mathematical utilities
- **Model Validation** - Input validation and error handling
- **GPU Support Consideration** - Architecture designed for future GPU acceleration
- **PHP 8.0+ Compatibility** - Modern PHP features and type safety

## Quick Start

### Basic Neural Network (XOR Problem)

\`\`\`php
<?php
require_once 'vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetworks\NeuralNetwork;

// Training data for XOR problem
$trainingData = [
    ['input' => [0, 0], 'output' => [0]],
    ['input' => [0, 1], 'output' => [1]],
    ['input' => [1, 0], 'output' => [1]],
    ['input' => [1, 1], 'output' => [0]]
];

// Create and configure neural network
$net = new NeuralNetwork([
    'hiddenLayers' => [3],
    'activation' => 'sigmoid',
    'learningRate' => 0.3,
    'iterations' => 20000
]);

// Train the network
$result = $net->train($trainingData);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n";

// Test the network
foreach ($trainingData as $data) {
    $output = $net->run($data['input']);
    echo "Input: [" . implode(", ", $data['input']) . "] => Output: " . round($output[0]) . "\n";
}

// Save the model
$json = $net->toJSON();
file_put_contents('xor-model.json', $json);
\`\`\`

## Neural Network Types

### Using the Brain Factory

The Brain class provides a convenient factory for creating different types of neural networks:

\`\`\`php
use GoLiveHost\Brain\Brain;

// Create a feedforward neural network
$nn = Brain::neuralNetwork([
    'hiddenLayers' => [20, 10],
    'activation' => 'relu',
    'dropout' => 0.2
]);

// Create an LSTM network
$lstm = Brain::lstm([
    'hiddenLayers' => [50, 25],
    'learningRate' => 0.01,
    'praxis' => 'adam'
]);

// Create a Liquid State Machine
$lsm = Brain::liquidStateMachine([
    'reservoirSize' => 100,
    'connectivity' => 0.1,
    'spectralRadius' => 0.9
]);

// Load a model from JSON
$model = Brain::fromJSON($json);
\`\`\`

### LSTM for Time Series Prediction

\`\`\`php
use GoLiveHost\Brain\NeuralNetworks\LSTM;

// Prepare sequence data
$sequences = [
    [
        'input' => [[0.1], [0.2], [0.3], [0.4], [0.5]],
        'output' => [[0.6], [0.7], [0.8]]
    ],
    // ... more sequences
];

// Create and train LSTM
$lstm = new LSTM([
    'inputSize' => 1,
    'hiddenLayers' => [50, 25],
    'outputSize' => 1,
    'learningRate' => 0.01,
    'iterations' => 1000,
    'batchSize' => 32,
    'praxis' => 'adam'
]);

$result = $lstm->train($sequences);

// Generate predictions
$testSequence = [[0.5], [0.6], [0.7], [0.8], [0.9]];
$predictions = $lstm->run($testSequence);

// Generate future values
$generated = $lstm->generate($testSequence, 10);
\`\`\`

### GRU for Sequential Data

\`\`\`php
use GoLiveHost\Brain\GRU;

$gru = new GRU([
    'inputSize' => 10,
    'hiddenSize' => 20,
    'outputSize' => 5,
    'learningRate' => 0.01,
    'activation' => 'tanh'
]);

$result = $gru->train($sequences);
$output = $gru->run($inputSequence);
\`\`\`

### Liquid State Machine for Complex Patterns

\`\`\`php
use GoLiveHost\Brain\NeuralNetworks\LiquidStateMachine;

$lsm = new LiquidStateMachine([
    'inputSize' => 3,
    'reservoirSize' => 100,
    'outputSize' => 2,
    'connectivity' => 0.1,
    'spectralRadius' => 0.9,
    'leakingRate' => 0.3,
    'regularization' => 0.001
]);

$result = $lsm->train($sequences);
$outputs = $lsm->run($inputSequence);
\`\`\`

## Advanced Features

### Cross-Validation

\`\`\`php
use GoLiveHost\Brain\Utilities\CrossValidation;

// K-fold cross-validation
$results = CrossValidation::kFold($model, $data, 5);
echo "Average accuracy: " . $results['averageMetrics']['accuracy'] . "\n";

// Stratified k-fold for classification
$results = CrossValidation::stratifiedKFold(
    $model, 
    $data, 
    5, 
    function($item) { return $item['output'][0]; }
);

// Train-test split
$split = CrossValidation::trainTestSplit($data, 0.2);
$model->train($split['train']);
$testResults = CrossValidation::evaluateModel($model, $split['test']);
\`\`\`

### Model Checkpointing

\`\`\`php
use GoLiveHost\Brain\Utilities\ModelCheckpoint;

$checkpoint = new ModelCheckpoint([
    'directory' => './checkpoints',
    'filePrefix' => 'my_model',
    'saveFrequency' => 100,
    'saveOnlyBest' => true,
    'monitorMetric' => 'error',
    'maxCheckpoints' => 5
]);

// During training loop
for ($epoch = 0; $epoch < 1000; $epoch++) {
    // ... training code ...
    
    $metrics = ['error' => $error, 'accuracy' => $accuracy];
    $checkpoint->save($model, $metrics, $epoch);
}

// Load the best model
$bestModel = $checkpoint->loadBest(NeuralNetwork::class);
\`\`\`

### Batch Normalization

\`\`\`php
use GoLiveHost\Brain\Layers\BatchNormalization;

$batchNorm = new BatchNormalization(100, [
    'epsilon' => 1e-5,
    'momentum' => 0.9
]);

// During training
$batchNorm->setTraining(true);
$normalized = $batchNorm->forward($batchData);

// During inference
$batchNorm->setTraining(false);
$output = $batchNorm->forward($input);
\`\`\`

### Custom Optimizers

\`\`\`php
use GoLiveHost\Brain\Optimizers\Adam;
use GoLiveHost\Brain\Optimizers\RMSprop;
use GoLiveHost\Brain\Optimizers\AdaGrad;

// Adam optimizer
$adam = new Adam([
    'learningRate' => 0.001,
    'beta1' => 0.9,
    'beta2' => 0.999,
    'epsilon' => 1e-8
]);

// RMSprop optimizer
$rmsprop = new RMSprop([
    'learningRate' => 0.01,
    'decay' => 0.9,
    'epsilon' => 1e-8
]);

// AdaGrad optimizer
$adagrad = new AdaGrad([
    'learningRate' => 0.01,
    'epsilon' => 1e-8
]);
\`\`\`

### Data Preprocessing

\`\`\`php
use GoLiveHost\Brain\Utilities\Normalizer;
use GoLiveHost\Brain\Utilities\DataFormatter;

// Normalization
$normalizer = new Normalizer();
$normalizer->fit($trainingData);
$normalizedData = $normalizer->transform($trainingData);

// Format data for sequences
$formatter = new DataFormatter();
$formattedSequences = $formatter->formatSequences($sequences);
\`\`\`

### Model Validation

\`\`\`php
use GoLiveHost\Brain\Utilities\ModelValidator;

// Validate neural network options
$validatedOptions = ModelValidator::validateNeuralNetworkOptions($options);

// Validate training data
ModelValidator::validateTrainingData($data);

// Validate sequence data
ModelValidator::validateTrainingData($sequences, true);
\`\`\`

## Configuration Options

### Neural Network Options

| Option | Description | Default |
|--------|-------------|---------|
| `inputSize` | Number of input neurons | 0 (auto-detect) |
| `hiddenLayers` | Array of hidden layer sizes | [10] |
| `outputSize` | Number of output neurons | 0 (auto-detect) |
| `activation` | Activation function | 'sigmoid' |
| `learningRate` | Initial learning rate | 0.3 |
| `momentum` | Momentum for SGD | 0.1 |
| `iterations` | Maximum training iterations | 20000 |
| `errorThresh` | Error threshold to stop training | 0.005 |
| `log` | Enable training progress logging | false |
| `logPeriod` | Iterations between log outputs | 10 |
| `dropout` | Dropout rate for regularization | 0 |
| `decayRate` | Learning rate decay factor | 0.999 |
| `batchSize` | Batch size for training | 10 |
| `praxis` | Optimization algorithm ('sgd', 'adam') | 'adam' |
| `beta1` | Adam optimizer beta1 | 0.9 |
| `beta2` | Adam optimizer beta2 | 0.999 |
| `epsilon` | Adam optimizer epsilon | 1e-8 |
| `normalize` | Auto-normalize data | true |

### LSTM Options

| Option | Description | Default |
|--------|-------------|---------|
| `inputSize` | Size of input vectors | 0 (auto-detect) |
| `hiddenLayers` | Array of LSTM layer sizes | [20] |
| `outputSize` | Size of output vectors | 0 (auto-detect) |
| `activation` | Activation function | 'tanh' |
| `learningRate` | Initial learning rate | 0.01 |
| `iterations` | Maximum training iterations | 20000 |
| `clipGradient` | Gradient clipping threshold | 5 |
| `batchSize` | Batch size for training | 10 |

### Liquid State Machine Options

| Option | Description | Default |
|--------|-------------|---------|
| `inputSize` | Number of input neurons | 0 (auto-detect) |
| `reservoirSize` | Size of the reservoir | 100 |
| `outputSize` | Number of output neurons | 0 (auto-detect) |
| `connectivity` | Reservoir connectivity ratio | 0.1 |
| `spectralRadius` | Spectral radius of reservoir | 0.9 |
| `leakingRate` | Leaking rate for neurons | 0.3 |
| `regularization` | L2 regularization parameter | 0.0001 |
| `washoutPeriod` | Initial timesteps to discard | 10 |

## Examples

<div style="color: red; font-weight: bold; border: 2px solid red; padding: 10px; margin: 15px 0; background-color: #ffeeee;">
⚠️ IMPORTANT NOTE: The examples provided are for demonstration purposes only and are trained on very limited datasets. The outputs may not be accurate or reliable for real-world applications. For production use, you must train models on larger, representative datasets and thoroughly validate their performance before deployment.
</div>

The library includes comprehensive examples in the `examples` directory:

- **`basic.php`** - Simple XOR problem demonstration
- **`advanced.php`** - Character recognition and time series prediction
- **`advanced-examples.php`** - Non-linear regression, LSTM time series, and LSM classification
- **`image-recognition.php`** - Simple shape recognition with neural networks
- **`text-generation.php`** - Character-level text generation using LSTM
- **`stock-prediction.php`** - Time series prediction for financial data
- **`cross-validation-example.php`** - Demonstration of cross-validation techniques
- **`model-checkpoint-example.php`** - Model checkpointing during training
- **`batch-normalization-example.php`** - Using batch normalization layers
- **`optimizer-example.php`** - Comparing different optimization algorithms

## API Reference

### Matrix Operations

\`\`\`php
use GoLiveHost\Brain\Utilities\Matrix;

// Matrix multiplication
$result = Matrix::multiply($matrixA, $matrixB);

// Matrix addition
$sum = Matrix::add($matrixA, $matrixB);

// Matrix transpose
$transposed = Matrix::transpose($matrix);

// Element-wise multiplication
$hadamard = Matrix::elementMultiply($matrixA, $matrixB);

// Matrix inverse
$inverse = Matrix::inverse($matrix);

// Determinant
$det = Matrix::determinant($matrix);
\`\`\`

### Tensor Operations

\`\`\`php
use GoLiveHost\Brain\Utilities\Tensor;

// Apply function to each element
$doubled = Tensor::map($tensor, fn($x) => $x * 2);

// Tensor operations
$sum = Tensor::sum($tensor);
$mean = Tensor::mean($tensor);
$max = Tensor::max($tensor);
$min = Tensor::min($tensor);

// Reshape tensor
$reshaped = Tensor::reshape($tensor, [10, 10]);
\`\`\`

### Model Export

\`\`\`php
// Export to standalone PHP class
$phpCode = $neuralNetwork->exportToPhp('MyNeuralNetwork');
file_put_contents('MyNeuralNetwork.php', $phpCode);

// Use the exported model
require_once 'MyNeuralNetwork.php';
$model = new MyNeuralNetwork();
$output = $model->run($input);
\`\`\`

## Error Handling

The library uses custom exceptions for better error handling:

\`\`\`php
use GoLiveHost\Brain\Exceptions\BrainException;

try {
    $model = new NeuralNetwork($options);
    $result = $model->train($data);
} catch (BrainException $e) {
    echo "Brain Error: " . $e->getMessage() . "\n";
}
\`\`\`

## Performance Considerations

1. **Memory Usage**: The library is optimized for memory efficiency, especially in LSTM implementations
2. **Batch Processing**: Use batch training for better performance with large datasets
3. **Learning Rate Decay**: Helps achieve better convergence
4. **Early Stopping**: Prevents overfitting and reduces training time
5. **Model Checkpointing**: Save progress during long training sessions

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

\`\`\`bash
# Clone the repository
git clone https://github.com/golivehost/brain.git
cd brain

# Install dependencies
composer install

# Run tests
composer test

# Check code style
composer cs-check

# Fix code style
composer cs-fix
\`\`\`

## License

This library is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/golivehost/brain/issues)
- **Documentation**: [Wiki](https://github.com/golivehost/brain/wiki)
- **Examples**: See the `examples` directory

## Credits

Developed by **Go Live Web Solutions** ([golive.host](https://golive.host))  
Author: **Shubhdeep Singh** ([GitHub.com/shubhdeepdev](https://github.com/shubhdeepdev))

---

*Building intelligent PHP applications with neural networks made simple.*
