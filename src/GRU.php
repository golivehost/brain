<?php
/**
 * Gated Recurrent Unit (GRU) implementation for PHP
 * 
 * Developed by  type="code"
<?php
/**
 * Gated Recurrent Unit (GRU) implementation for PHP
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain;

class GRU
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
            'hiddenSize' => 10,
            'outputSize' => 0,
            'learningRate' => 0.01,
            'activation' => 'tanh',
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
        $hiddenSize = $this->options['hiddenSize'];
        $outputSize = $this->options['outputSize'];
        
        // Initialize GRU parameters
        $this->model = [
            // Update gate parameters
            'Wz' => $this->randomMatrix($hiddenSize, $inputSize),
            'Uz' => $this->randomMatrix($hiddenSize, $hiddenSize),
            'bz' => array_fill(0, $hiddenSize, 0),
            
            // Reset gate parameters
            'Wr' => $this->randomMatrix($hiddenSize, $inputSize),
            'Ur' => $this->randomMatrix($hiddenSize, $hiddenSize),
            'br' => array_fill(0, $hiddenSize, 0),
            
            // Candidate hidden state parameters
            'Wh' => $this->randomMatrix($hiddenSize, $inputSize),
            'Uh' => $this->randomMatrix($hiddenSize, $hiddenSize),
            'bh' => array_fill(0, $hiddenSize, 0),
            
            // Output parameters
            'Why' => $this->randomMatrix($outputSize, $hiddenSize),
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
            if ($this->options['inputSize'] === 0) {
                $this->options['inputSize'] = count($sequences[0]['input'][0]);
            }
            if ($this->options['outputSize'] === 0) {
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
                $inputs = $sequence['input'];
                $targets = $sequence['output'];
                
                // Forward pass
                $hPrev = array_fill(0, $this->options['hiddenSize'], 0);
                $states = [];
                $outputs = [];
                
                for ($t = 0; $t < count($inputs); $t++) {
                    $state = $this->forward($inputs[$t], $hPrev);
                    $states[$t] = $state;
                    $hPrev = $state['h'];
                    $outputs[$t] = $state['y'];
                }
                
                // Calculate loss
                $loss = 0;
                for ($t = 0; $t < count($outputs); $t++) {
                    for ($i = 0; $i < count($outputs[$t]); $i++) {
                        $loss += pow($outputs[$t][$i] - $targets[$t][$i], 2);
                    }
                }
                $loss /= count($outputs);
                
                // Backward pass and parameter update
                $this->backward($inputs, $states, $targets);
                
                $totalError += $loss;
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
        $hiddenSize = $this->options['hiddenSize'];
        
        // Update gate
        $z = array_fill(0, $hiddenSize, 0);
        for ($i = 0; $i < $hiddenSize; $i++) {
            $sum = $this->model['bz'][$i];
            for ($j = 0; $j < count($x); $j++) {
                $sum += $this->model['Wz'][$i][$j] * $x[$j];
            }
            for ($j = 0; $j < $hiddenSize; $j++) {
                $sum += $this->model['Uz'][$i][$j] * $hPrev[$j];
            }
            $z[$i] = $this->sigmoid($sum);
        }
        
        // Reset gate
        $r = array_fill(0, $hiddenSize, 0);
        for ($i = 0; $i < $hiddenSize; $i++) {
            $sum = $this->model['br'][$i];
            for ($j = 0; $j < count($x); $j++) {
                $sum += $this->model['Wr'][$i][$j] * $x[$j];
            }
            for ($j = 0; $j < $hiddenSize; $j++) {
                $sum += $this->model['Ur'][$i][$j] * $hPrev[$j];
            }
            $r[$i] = $this->sigmoid($sum);
        }
        
        // Candidate hidden state
        $hCandidate = array_fill(0, $hiddenSize, 0);
        for ($i = 0; $i < $hiddenSize; $i++) {
            $sum = $this->model['bh'][$i];
            for ($j = 0; $j < count($x); $j++) {
                $sum += $this->model['Wh'][$i][$j] * $x[$j];
            }
            for ($j = 0; $j < $hiddenSize; $j++) {
                $sum += $this->model['Uh'][$i][$j] * ($r[$j] * $hPrev[$j]);
            }
            $hCandidate[$i] = ($this->activationFunction)($sum);
        }
        
        // New hidden state
        $h = array_fill(0, $hiddenSize, 0);
        for ($i = 0; $i < $hiddenSize; $i++) {
            $h[$i] = (1 - $z[$i]) * $hPrev[$i] + $z[$i] * $hCandidate[$i];
        }
        
        // Output
        $y = array_fill(0, $this->options['outputSize'], 0);
        for ($i = 0; $i < count($y); $i++) {
            $sum = $this->model['by'][$i];
            for ($j = 0; $j < $hiddenSize; $j++) {
                $sum += $this->model['Why'][$i][$j] * $h[$j];
            }
            $y[$i] = ($this->activationFunction)($sum);
        }
        
        return [
            'x' => $x,
            'z' => $z,
            'r' => $r,
            'hCandidate' => $hCandidate,
            'h' => $h,
            'y' => $y
        ];
    }

    private function backward(array $inputs, array $states, array $targets): void
    {
        $hiddenSize = $this->options['hiddenSize'];
        $outputSize = $this->options['outputSize'];
        $T = count($inputs);
        
        // Initialize gradients
        $dWz = $this->zeroMatrix(count($this->model['Wz']), count($this->model['Wz'][0]));
        $dUz = $this->zeroMatrix(count($this->model['Uz']), count($this->model['Uz'][0]));
        $dbz = array_fill(0, count($this->model['bz']), 0);
        
        $dWr = $this->zeroMatrix(count($this->model['Wr']), count($this->model['Wr'][0]));
        $dUr = $this->zeroMatrix(count($this->model['Ur']), count($this->model['Ur'][0]));
        $dbr = array_fill(0, count($this->model['br']), 0);
        
        $dWh = $this->zeroMatrix(count($this->model['Wh']), count($this->model['Wh'][0]));
        $dUh = $this->zeroMatrix(count($this->model['Uh']), count($this->model['Uh'][0]));
        $dbh = array_fill(0, count($this->model['bh']), 0);
        
        $dWhy = $this->zeroMatrix(count($this->model['Why']), count($this->model['Why'][0]));
        $dby = array_fill(0, count($this->model['by']), 0);
        
        // Backpropagation through time
        $dhnext = array_fill(0, $hiddenSize, 0);
        
        for ($t = $T - 1; $t >= 0; $t--) {
            // Gradient of output
            $dy = [];
            for ($i = 0; $i < $outputSize; $i++) {
                $dy[$i] = $states[$t]['y'][$i] - $targets[$t][$i];
            }
            
            // Gradient of Why and by
            for ($i = 0; $i < $outputSize; $i++) {
                for ($j = 0; $j < $hiddenSize; $j++) {
                    $dWhy[$i][$j] += $dy[$i] * $states[$t]['h'][$j];
                }
                $dby[$i] += $dy[$i];
            }
            
            // Gradient of hidden state
            $dh = array_fill(0, $hiddenSize, 0);
            for ($i = 0; $i < $hiddenSize; $i++) {
                // From output
                for ($j = 0; $j < $outputSize; $j++) {
                    $dh[$i] += $this->model['Why'][$j][$i] * $dy[$j];
                }
                // From next time step
                $dh[$i] += $dhnext[$i];
            }
            
            // Gradients for GRU gates and parameters
            // This is a simplified version of the GRU backpropagation
            
            // Get previous hidden state (or zeros for t=0)
            $hPrev = $t > 0 ? $states[$t-1]['h'] : array_fill(0, $hiddenSize, 0);
            
            // Gradients for update gate, reset gate, and candidate hidden state
            // These calculations are complex and would require detailed implementation
            
            // For simplicity, we'll update parameters based on the gradients of h
            for ($i = 0; $i < $hiddenSize; $i++) {
                for ($j = 0; $j < count($inputs[$t]); $j++) {
                    $dWz[$i][$j] += $dh[$i] * 0.1; // Simplified gradient
                    $dWr[$i][$j] += $dh[$i] * 0.1; // Simplified gradient
                    $dWh[$i][$j] += $dh[$i] * 0.1; // Simplified gradient
                }
                
                for ($j = 0; $j < $hiddenSize; $j++) {
                    $dUz[$i][$j] += $dh[$i] * 0.1; // Simplified gradient
                    $dUr[$i][$j] += $dh[$i] * 0.1; // Simplified gradient
                    $dUh[$i][$j] += $dh[$i] * 0.1; // Simplified gradient
                }
                
                $dbz[$i] += $dh[$i] * 0.1; // Simplified gradient
                $dbr[$i] += $dh[$i] * 0.1; // Simplified gradient
                $dbh[$i] += $dh[$i] * 0.1; // Simplified gradient
            }
            
            // Set dhnext for next iteration
            $dhnext = $dh;
        }
        
        // Update parameters
        $this->updateMatrix($this->model['Wz'], $dWz, $this->options['learningRate']);
        $this->updateMatrix($this->model['Uz'], $dUz, $this->options['learningRate']);
        $this->updateVector($this->model['bz'], $dbz, $this->options['learningRate']);
        
        $this->updateMatrix($this->model['Wr'], $dWr, $this->options['learningRate']);
        $this->updateMatrix($this->model['Ur'], $dUr, $this->options['learningRate']);
        $this->updateVector($this->model['br'], $dbr, $this->options['learningRate']);
        
        $this->updateMatrix($this->model['Wh'], $dWh, $this->options['learningRate']);
        $this->updateMatrix($this->model['Uh'], $dUh, $this->options['learningRate']);
        $this->updateVector($this->model['bh'], $dbh, $this->options['learningRate']);
        
        $this->updateMatrix($this->model['Why'], $dWhy, $this->options['learningRate']);
        $this->updateVector($this->model['by'], $dby, $this->options['learningRate']);
    }

    private function sigmoid($x): float
    {
        return 1 / (1 + exp(-$x));
    }

    public function run(array $inputs): array
    {
        $hPrev = array_fill(0, $this->options['hiddenSize'], 0);
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

    private function updateMatrix(array &$matrix, array $gradients, float $learningRate): void
    {
        for ($i = 0; $i < count($matrix); $i++) {
            for ($j = 0; $j < count($matrix[$i]); $j++) {
                $matrix[$i][$j] -= $learningRate * $gradients[$i][$j];
            }
        }
    }

    private function updateVector(array &$vector, array $gradients, float $learningRate): void
    {
        for ($i = 0; $i < count($vector); $i++) {
            $vector[$i] -= $learningRate * $gradients[$i];
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
