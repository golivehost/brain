<?php
/**
 * Recurrent Neural Network implementation for PHP
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain;

class RNN
{
    private array $options;
    private array $model = [];
    private $activationFunction;
    private $activationDerivative;

    public function __construct(array $options = [])
    {
        // Default options
        $this->options = array_merge([
            'inputSize' => 0,
            'hiddenLayers' => [10],
            'outputSize' => 0,
            'learningRate' => 0.01,
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
        $inputSize = $this->options['inputSize'];
        $hiddenSizes = $this->options['hiddenLayers'];
        $outputSize = $this->options['outputSize'];
        
        // Initialize model parameters
        $this->model = [
            'Wxh' => $this->randomMatrix($hiddenSizes[0], $inputSize),
            'Whh' => $this->randomMatrix($hiddenSizes[0], $hiddenSizes[0]),
            'Why' => $this->randomMatrix($outputSize, $hiddenSizes[0]),
            'bh' => array_fill(0, $hiddenSizes[0], 0),
            'by' => array_fill(0, $outputSize, 0)
        ];
    }

    private function randomMatrix(int $rows, int $cols): array
    {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            $matrix[$i] = [];
            for ($j = 0; $j < $cols; $j++) {
                $matrix[$i][$j] = $this->random() * 0.2 - 0.1; // Random between -0.1 and 0.1
            }
        }
        return $matrix;
    }

    private function random(): float
    {
        return mt_rand() / mt_getrandmax();
    }

    public function train(array $sequences, array $trainingOptions = []): array
    {
        $options = array_merge($this->options, $trainingOptions);
        
        if (empty($this->model)) {
            // Auto-detect input and output sizes if not provided
            if ($this->options['inputSize'] === 0 && !empty($sequences) && isset($sequences[0]['input']) && is_array($sequences[0]['input']) && !empty($sequences[0]['input'])) {
                $this->options['inputSize'] = count($sequences[0]['input'][0]);
            }
            if ($this->options['outputSize'] === 0 && !empty($sequences) && isset($sequences[0]['output']) && is_array($sequences[0]['output']) && !empty($sequences[0]['output'])) {
                $this->options['outputSize'] = count($sequences[0]['output'][0]);
            }
            $this->initialize();
        }

        $error = 1;
        $iterations = 0;
        $errorLog = [];

        while ($error > $options['errorThresh'] && $iterations < $options['iterations']) {
            $totalError = 0;
            
            foreach ($sequences as $sequence) {
                // Ensure sequence has required structure
                if (!isset($sequence['input']) || !isset($sequence['output']) || 
                    !is_array($sequence['input']) || !is_array($sequence['output'])) {
                    continue;
                }
                
                $inputs = $sequence['input'];
                $targets = $sequence['output'];
                
                // Forward pass
                $hPrev = array_fill(0, $this->options['hiddenLayers'][0], 0);
                $outputs = [];
                $hiddenStates = [];
                
                for ($t = 0; $t < count($inputs); $t++) {
                    // Ensure input exists and is valid
                    if (!isset($inputs[$t]) || !is_array($inputs[$t])) {
                        continue;
                    }
                    
                    $hiddenStates[$t] = $this->forward($inputs[$t], $hPrev);
                    $hPrev = $hiddenStates[$t]['h'];
                    $outputs[$t] = $hiddenStates[$t]['y'];
                }
                
                // Backward pass
                $dWxh = $this->zeroMatrix(count($this->model['Wxh']), count($this->model['Wxh'][0]));
                $dWhh = $this->zeroMatrix(count($this->model['Whh']), count($this->model['Whh'][0]));
                $dWhy = $this->zeroMatrix(count($this->model['Why']), count($this->model['Why'][0]));
                $dbh = array_fill(0, count($this->model['bh']), 0);
                $dby = array_fill(0, count($this->model['by']), 0);
                
                $dhnext = array_fill(0, $this->options['hiddenLayers'][0], 0);
                $loss = 0;
                
                for ($t = count($inputs) - 1; $t >= 0; $t--) {
                    // Ensure output and hidden state exist for this timestep
                    if (!isset($outputs[$t]) || !isset($targets[$t]) || !isset($hiddenStates[$t])) {
                        continue;
                    }
                    
                    // Compute error
                    $dy = $outputs[$t];
                    for ($i = 0; $i < count($dy); $i++) {
                        // Ensure target value exists
                        if (isset($targets[$t][$i])) {
                            $dy[$i] -= $targets[$t][$i];
                            $loss += pow($dy[$i], 2);
                        }
                    }
                    
                    // Backprop into Why and by
                    for ($i = 0; $i < count($this->model['Why']); $i++) {
                        for ($j = 0; $j < count($this->model['Why'][$i]); $j++) {
                            if (isset($hiddenStates[$t]['h'][$j])) {
                                $dWhy[$i][$j] += $dy[$i] * $hiddenStates[$t]['h'][$j];
                            }
                        }
                        $dby[$i] += $dy[$i];
                    }
                    
                    // Backprop into hidden layer
                    $dh = [];
                    for ($i = 0; $i < count($this->model['Why'][0]); $i++) {
                        $dh[$i] = 0;
                        for ($j = 0; $j < count($this->model['Why']); $j++) {
                            $dh[$i] += $this->model['Why'][$j][$i] * $dy[$j];
                        }
                        $dh[$i] += $dhnext[$i];
                        
                        // Apply activation derivative
                        if (isset($hiddenStates[$t]['h'][$i])) {
                            $dh[$i] *= ($this->activationDerivative)($hiddenStates[$t]['h'][$i]);
                        }
                    }
                    
                    // Backprop into Wxh, Whh, and bh
                    for ($i = 0; $i < count($dh); $i++) {
                        $dbh[$i] += $dh[$i];
                        
                        for ($j = 0; $j < count($inputs[$t]); $j++) {
                            $dWxh[$i][$j] += $dh[$i] * $inputs[$t][$j];
                        }
                        
                        if ($t > 0 && isset($hiddenStates[$t-1]['h'])) {
                            for ($j = 0; $j < count($hiddenStates[$t-1]['h']); $j++) {
                                $dWhh[$i][$j] += $dh[$i] * $hiddenStates[$t-1]['h'][$j];
                            }
                        }
                    }
                    
                    $dhnext = $dh;
                }
                
                // Clip gradients to prevent exploding gradients
                $this->clipGradients($dWxh, 5);
                $this->clipGradients($dWhh, 5);
                $this->clipGradients($dWhy, 5);
                
                // Update model parameters
                $this->updateMatrix($this->model['Wxh'], $dWxh, $this->options['learningRate']);
                $this->updateMatrix($this->model['Whh'], $dWhh, $this->options['learningRate']);
                $this->updateMatrix($this->model['Why'], $dWhy, $this->options['learningRate']);
                $this->updateVector($this->model['bh'], $dbh, $this->options['learningRate']);
                $this->updateVector($this->model['by'], $dby, $this->options['learningRate']);
                
                $totalError += $loss / count($inputs);
            }
            
            $error = $totalError / count($sequences);
            $errorLog[] = $error;
            $iterations++;
        }

        return [
            'error' => $error,
            'iterations' => $iterations,
            'errorLog' => $errorLog
        ];
    }

    private function forward(array $x, array $hPrev): array
    {
        $h = array_fill(0, count($this->model['bh']), 0);
        
        // Calculate hidden state
        for ($i = 0; $i < count($h); $i++) {
            $sum = $this->model['bh'][$i];
            
            // Input to hidden
            for ($j = 0; $j < count($x); $j++) {
                if (isset($this->model['Wxh'][$i][$j])) {
                    $sum += $this->model['Wxh'][$i][$j] * $x[$j];
                }
            }
            
            // Hidden to hidden
            for ($j = 0; $j < count($hPrev); $j++) {
                if (isset($this->model['Whh'][$i][$j])) {
                    $sum += $this->model['Whh'][$i][$j] * $hPrev[$j];
                }
            }
            
            $h[$i] = ($this->activationFunction)($sum);
        }
        
        // Calculate output
        $y = array_fill(0, count($this->model['by']), 0);
        for ($i = 0; $i < count($y); $i++) {
            $sum = $this->model['by'][$i];
            
            for ($j = 0; $j < count($h); $j++) {
                if (isset($this->model['Why'][$i][$j])) {
                    $sum += $this->model['Why'][$i][$j] * $h[$j];
                }
            }
            
            $y[$i] = ($this->activationFunction)($sum);
        }
        
        return ['h' => $h, 'y' => $y];
    }

    public function run(array $inputs): array
    {
        $hPrev = array_fill(0, $this->options['hiddenLayers'][0], 0);
        $outputs = [];
        
        foreach ($inputs as $input) {
            $result = $this->forward($input, $hPrev);
            $hPrev = $result['h'];
            $outputs[] = $result['y'];
        }
        
        return $outputs;
    }

    private function zeroMatrix(int $rows, int $cols): array
    {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            $matrix[$i] = array_fill(0, $cols, 0);
        }
        return $matrix;
    }

    private function clipGradients(array &$gradients, float $maxValue): void
    {
        foreach ($gradients as &$row) {
            foreach ($row as &$value) {
                if ($value > $maxValue) {
                    $value = $maxValue;
                } elseif ($value < -$maxValue) {
                    $value = -$maxValue;
                }
            }
        }
    }

    private function updateMatrix(array &$matrix, array $gradients, float $learningRate): void
    {
        for ($i = 0; $i < count($matrix); $i++) {
            for ($j = 0; $j < count($matrix[$i]); $j++) {
                if (isset($gradients[$i][$j])) {
                    $matrix[$i][$j] -= $learningRate * $gradients[$i][$j];
                }
            }
        }
    }

    private function updateVector(array &$vector, array $gradients, float $learningRate): void
    {
        for ($i = 0; $i < count($vector); $i++) {
            if (isset($gradients[$i])) {
                $vector[$i] -= $learningRate * $gradients[$i];
            }
        }
    }

    public function toJSON(): string
    {
        return json_encode([
            'options' => $this->options,
            'model' => $this->model
        ]);
    }

    public static function fromJSON(string $json): self
    {
        $data = json_decode($json, true);
        $network = new self($data['options']);
        $network->model = $data['model'];
        return $network;
    }
}
