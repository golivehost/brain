# GoLiveHost Brain

A PHP neural network library

**Developed by:** Go Live Web Solutions ([golive.host](https://golive.host))  
**Author:** Shubhdeep Singh ([GitHub.com/shubhdeepdev](https://github.com/shubhdeepdev))

## Installation

```bash
composer require golivehost/brain
```

## Features

- Multiple neural network types:
  - Feedforward Neural Network
  - Recurrent Neural Network (RNN)
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
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

```php
<?php
require_once 'vendor/autoload.php';

use GoLiveHost\Brain\NeuralNetworks\NeuralNetwork;

$trainingData = [
    ['input' => [0, 0], 'output' => [0]],
    ['input' => [0, 1], 'output' => [1]],
    ['input' => [1, 0], 'output' => [1]],
    ['input' => [1, 1], 'output' => [0]]
];

$net = new NeuralNetwork([
    'hiddenLayers' => [3],
    'learningRate' => 0.3,
]);

$result = $net->train($trainingData);
echo "Training completed in {$result['iterations']} iterations with error {$result['error']}\n";

foreach ($trainingData as $data) {
    $output = $net->run($data['input']);
    echo "Input: [" . implode(", ", $data['input']) . "] => Output: " . round($output[0]) . "\n";
}

$json = $net->toJSON();
file_put_contents('xor-model.json', $json);

$loadedNet = NeuralNetwork::fromJSON($json);
```

## Advanced Usage

### LSTM for Time Series Prediction

```php
<?php
use GoLiveHost\Brain\NeuralNetworks\LSTM;

$sequences = [
    [
        'input' => [[0.1], [0.2], [0.3], [0.4], [0.5]],
        'output' => [[0.6], [0.7], [0.8]]
    ],
];

$lstm = new LSTM([
    'hiddenLayers' => [20],
    'learningRate' => 0.01,
    'iterations' => 1000,
    'praxis' => 'adam'
]);

$result = $lstm->train($sequences);
echo "Training completed with error {$result['error']}\n";

$testSequence = [[0.5], [0.6], [0.7], [0.8], [0.9]];
$predictions = $lstm->run($testSequence);
$generatedSequence = $lstm->generate($testSequence, 10);
```

### RNN for Text Generation

```php
<?php
use GoLiveHost\Brain\RNN;

$sequences = [
    [
        'input' => [['h'], ['e'], ['l'], ['l']],
        'output' => [['o']]
    ],
];

$rnn = new RNN([
    'hiddenLayers' => [20],
    'learningRate' => 0.01,
    'iterations' => 1000
]);

$result = $rnn->train($sequences);

$seed = [['s'], ['t'], ['a'], ['r'], ['t']];
$output = $rnn->run($seed);
```

### GRU for Sequential Data

```php
<?php
use GoLiveHost\Brain\GRU;

$gru = new GRU([
    'inputSize' => 5,
    'hiddenSize' => 10,
    'outputSize' => 2,
    'learningRate' => 0.01
]);

$result = $gru->train($sequences);
$output = $gru->run($inputSequence);
```

### Liquid State Machine for Complex Temporal Patterns

```php
<?php
use GoLiveHost\Brain\NeuralNetworks\LiquidStateMachine;

$lsm = new LiquidStateMachine([
    'inputSize' => 3,
    'reservoirSize' => 100,
    'outputSize' => 2,
    'connectivity' => 0.1,
    'spectralRadius' => 0.9,
    'leakingRate' => 0.3
]);

$result = $lsm->train($sequences);
$outputs = $lsm->run($inputSequence);
```

## Using the Brain Factory

```php
<?php
use GoLiveHost\Brain\Brain;

$nn = Brain::neuralNetwork([
    'hiddenLayers' => [10, 5],
    'activation' => 'relu'
]);

$lstm = Brain::lstm([
    'hiddenLayers' => [20],
    'learningRate' => 0.01
]);

$lsm = Brain::liquidStateMachine([
    'reservoirSize' => 100,
    'connectivity' => 0.1
]);

$model = Brain::fromJSON($json);
```

## Data Preprocessing

### Normalization

```php
<?php
use GoLiveHost\Brain\Utilities\Normalizer;

$normalizer = new Normalizer();
$normalizer->fit($trainingData);
$normalizedData = $normalizer->transform($trainingData);

$normalizedInput = $normalizer->transformInput($input);
$denormalizedOutput = $normalizer->inverseTransformOutput($output);
```

### Data Formatting

```php
<?php
use GoLiveHost\Brain\Utilities\DataFormatter;

$formatter = new DataFormatter();
$formattedData = $formatter->format($data);
$formattedSequences = $formatter->formatSequences($sequences);
```

## Matrix and Tensor Operations

```php
<?php
use GoLiveHost\Brain\Utilities\Matrix;
use GoLiveHost\Brain\Utilities\Tensor;

$result = Matrix::multiply($matrixA, $matrixB);
$sum = Matrix::add($matrixA, $matrixB);
$transposed = Matrix::transpose($matrix);

$mapped = Tensor::map($tensor, function($x) { return $x * 2; });
$sum = Tensor::sum($tensor);
$mean = Tensor::mean($tensor);
```

## Configuration Options

### Neural Network Options

| Option        | Description                          | Default   |
|---------------|--------------------------------------|-----------|
| inputSize     | Size of input layer                  | 0         |
| hiddenLayers  | Array of hidden layer sizes          | [10]      |
| outputSize    | Size of output layer                 | 0         |
| activation    | Activation function                  | sigmoid   |
| learningRate  | Learning rate                        | 0.3       |
| momentum      | Momentum for SGD                     | 0.1       |
| iterations    | Maximum training iterations          | 20000     |
| errorThresh   | Error threshold to stop training     | 0.005     |
| log           | Whether to log training progress     | false     |
| logPeriod     | How often to log (iterations)        | 10        |
| dropout       | Dropout rate for regularization      | 0         |
| decayRate     | Learning rate decay                  | 0.999     |
| batchSize     | Batch size for training              | 10        |
| praxis        | Optimization algorithm               | adam      |

### LSTM Options

| Option        | Description                          | Default   |
|---------------|--------------------------------------|-----------|
| inputSize     | Size of input layer                  | 0         |
| hiddenLayers  | Array of hidden layer sizes          | [20]      |
| outputSize    | Size of output layer                 | 0         |
| activation    | Activation function                  | tanh      |
| learningRate  | Learning rate                        | 0.01      |
| iterations    | Maximum training iterations          | 20000     |
| clipGradient  | Maximum gradient value               | 5         |

### Liquid State Machine Options

| Option         | Description                          | Default   |
|----------------|--------------------------------------|-----------|
| inputSize      | Size of input layer                  | 0         |
| reservoirSize  | Size of the reservoir                | 100       |
| outputSize     | Size of output layer                 | 0         |
| connectivity   | Connectivity of the reservoir        | 0.1       |
| spectralRadius | Spectral radius of reservoir weights | 0.9       |
| leakingRate    | Leaking rate for reservoir neurons   | 0.3       |
| washoutPeriod  | Initial timesteps to discard         | 10        |

## Exporting Models

```php
<?php
$code = $neuralNetwork->exportToPhp('MyNeuralNetwork');
file_put_contents('MyNeuralNetwork.php', $code);
```

## Examples

The library includes several examples in the `examples` directory:

- `basic.php`: Basic XOR example
- `advanced.php`: Advanced examples with RNN
- `image-recognition.php`: Simple image recognition
- `text-generation.php`: Character-level text generation with LSTM
- `stock-prediction.php`: Time series prediction for stock prices
- `advanced-examples.php`: Various advanced examples

## Error Handling

```php
<?php
use GoLiveHost\Brain\Exceptions\BrainException;

try {
    // Your code
} catch (BrainException $e) {
    echo "Error: " . $e->getMessage();
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This library is licensed under the MIT License.

## Credits

Developed by Go Live Web Solutions ([golive.host](https://golive.host))  
Author: Shubhdeep Singh ([GitHub.com/shubhdeepdev](https://github.com/shubhdeepdev))

