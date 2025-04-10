<?php
/**
 * Neural Network implementation for PHP
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain;

class NeuralNetwork
{
    private array $options;
    private array $layers = [];
    private array $weights = [];
    private array $biases = [];
    private $activationFunction;
    private $activationDerivative;

    public function __construct(array $options = [])
    {
        // Default options
        $this->options = array_merge([
            'inputSize' => 0,
            'hiddenLayers' => [10],
            'outputSize' => 0,
            'learningRate' => 0.3,
            'activation' => 'sigmoid',
            'iterations' => 20000,
            'errorThresh' => 0.005,
        ], $options);

        $this->setActivationFunction($this->options['activation']);
    }

    private function setActivationFunction(string $name): void
    {
        switch ($name) {
            case 'sigmoid':
                $this->activationFunction = function ($x) {
                    return 1 / (1 + exp(-$x));
                };
                $this->activationDerivative = function ($x) {
                    $sigmoid = 1 / (1 + exp(-$x));
                    return $sigmoid * (1 - $sigmoid);
                };
                break;
            case 'relu':
                $this->activationFunction = function ($x) {
                    return max(0, $x);
                };
                $this->activationDerivative = function ($x) {
                    return $x > 0 ? 1 : 0;
                };
                break;
            case 'tanh':
                $this->activationFunction = function ($x) {
                    return tanh($x);
                };
                $this->activationDerivative = function ($x) {
                    $tanhX = tanh($x);
                    return 1 - ($tanhX * $tanhX);
                };
                break;
            default:
                throw new \InvalidArgumentException("Unknown activation function: {$name}");
        }
    }

    public function initialize(): void
    {
        $sizes = array_merge(
            [$this->options['inputSize']],
            $this->options['hiddenLayers'],
            [$this->options['outputSize']]
        );

        // Initialize weights and biases
        for ($i = 0; $i < count($sizes) - 1; $i++) {
            $this->weights[$i] = [];
            $this->biases[$i] = [];

            for ($j = 0; $j < $sizes[$i + 1]; $j++) {
                $this->biases[$i][$j] = $this->random() * 2 - 1; // Random between -1 and 1

                $this->weights[$i][$j] = [];
                for ($k = 0; $k < $sizes[$i]; $k++) {
                    $this->weights[$i][$j][$k] = $this->random() * 2 - 1; // Random between -1 and 1
                }
            }
        }
    }

    private function random(): float
    {
        return mt_rand() / mt_getrandmax();
    }

    public function train(array $data, array $trainingOptions = []): array
    {
        $options = array_merge($this->options, $trainingOptions);
        
        if (empty($this->weights)) {
            // Auto-detect input and output sizes if not provided
            if ($this->options['inputSize'] === 0) {
                $this->options['inputSize'] = count($data[0]['input']);
            }
            if ($this->options['outputSize'] === 0) {
                $this->options['outputSize'] = count($data[0]['output']);
            }
            $this->initialize();
        }

        $error = 1;
        $iterations = 0;
        $errorLog = [];

        while ($error > $options['errorThresh'] && $iterations < $options['iterations']) {
            $sum = 0;

            foreach ($data as $item) {
                $output = $this->forward($item['input']);
                $this->backward($item['output']);
                
                // Calculate error
                $itemError = 0;
                for ($i = 0; $i < count($output); $i++) {
                    $itemError += pow($output[$i] - $item['output'][$i], 2);
                }
                $sum += $itemError / count($output);
            }

            $error = $sum / count($data);
            $errorLog[] = $error;
            $iterations++;
        }

        return [
            'error' => $error,
            'iterations' => $iterations,
            'errorLog' => $errorLog
        ];
    }

    public function forward(array $input): array
    {
        $this->layers = [];
        $this->layers[0] = $input;

        // Forward propagation
        for ($i = 0; $i < count($this->weights); $i++) {
            $this->layers[$i + 1] = [];
            
            for ($j = 0; $j < count($this->weights[$i]); $j++) {
                $sum = $this->biases[$i][$j];
                
                for ($k = 0; $k < count($this->layers[$i]); $k++) {
                    $sum += $this->layers[$i][$k] * $this->weights[$i][$j][$k];
                }
                
                $this->layers[$i + 1][$j] = ($this->activationFunction)($sum);
            }
        }

        return end($this->layers);
    }

    private function backward(array $target): void
    {
        $outputLayer = count($this->layers) - 1;
        $deltas = [];
        
        // Calculate deltas for output layer
        $deltas[$outputLayer] = [];
        for ($i = 0; $i < count($this->layers[$outputLayer]); $i++) {
            $output = $this->layers[$outputLayer][$i];
            $error = $target[$i] - $output;
            $delta = $error * ($this->activationDerivative)($output);
            $deltas[$outputLayer][$i] = $delta;
        }
        
        // Calculate deltas for hidden layers
        for ($l = $outputLayer - 1; $l > 0; $l--) {
            $deltas[$l] = [];
            for ($i = 0; $i < count($this->layers[$l]); $i++) {
                $error = 0;
                for ($j = 0; $j < count($deltas[$l + 1]); $j++) {
                    $error += $deltas[$l + 1][$j] * $this->weights[$l][$j][$i];
                }
                $delta = $error * ($this->activationDerivative)($this->layers[$l][$i]);
                $deltas[$l][$i] = $delta;
            }
        }
        
        // Update weights and biases
        for ($l = 0; $l < $outputLayer; $l++) {
            for ($i = 0; $i < count($this->weights[$l]); $i++) {
                for ($j = 0; $j < count($this->weights[$l][$i]); $j++) {
                    $delta = $deltas[$l + 1][$i];
                    $input = $this->layers[$l][$j];
                    $this->weights[$l][$i][$j] += $this->options['learningRate'] * $delta * $input;
                }
                $this->biases[$l][$i] += $this->options['learningRate'] * $deltas[$l + 1][$i];
            }
        }
    }

    public function run(array $input): array
    {
        return $this->forward($input);
    }

    public function toJSON(): string
    {
        return json_encode([
            'options' => $this->options,
            'weights' => $this->weights,
            'biases' => $this->biases
        ]);
    }

    public static function fromJSON(string $json): self
    {
        $data = json_decode($json, true);
        $network = new self($data['options']);
        $network->weights = $data['weights'];
        $network->biases = $data['biases'];
        return $network;
    }
}
