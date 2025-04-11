<?php
/**
 * Long Short-Term Memory (LSTM) implementation for PHP
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\NeuralNetworks;

use GoLiveHost\Brain\Utilities\Matrix;
use GoLiveHost\Brain\Activation\ActivationFactory;
use GoLiveHost\Brain\Utilities\DataFormatter;
use GoLiveHost\Brain\Exceptions\BrainException;

class LSTM
{
    protected array $options;
    protected array $model = [];
    protected $activation;
    protected $activationDerivative;
    protected ?DataFormatter $dataFormatter = null;
    protected array $errorLog = [];
    protected array $trainStats = [];
    protected bool $isInitialized = false;

    /**
     * @param array $options Configuration options for the LSTM
     */
    public function __construct(array $options = [])
    {
        // Default options
        $this->options = array_merge([
            'inputSize' => 0,
            'hiddenLayers' => [20],
            'outputSize' => 0,
            'activation' => 'tanh',
            'learningRate' => 0.01,
            'iterations' => 20000,
            'errorThresh' => 0.005,
            'log' => false,
            'logPeriod' => 10,
            'dropout' => 0,
            'decayRate' => 0.999,
            'batchSize' => 10,
            'callbackPeriod' => 10,
            'timeout' => INF,
            'praxis' => 'adam',
            'beta1' => 0.9,
            'beta2' => 0.999,
            'epsilon' => 1e-8,
            'clipGradient' => 5
        ], $options);

        $this->setActivation($this->options['activation']);
        
        $this->dataFormatter = new DataFormatter();
    }

    /**
     * Set the activation function and its derivative
     */
    protected function setActivation(string $name): void
    {
        $activation = ActivationFactory::create($name);
        
        $this->activation = $activation->getFunction();
        $this->activationDerivative = $activation->getDerivative();
    }

    /**
     * Initialize the LSTM with random weights
     */
    public function initialize(): void
    {
        $inputSize = $this->options['inputSize'];
        $hiddenSizes = $this->options['hiddenLayers'];
        $outputSize = $this->options['outputSize'];
        
        // Initialize LSTM parameters for each layer
        $this->model = [];
        
        // Input gate parameters
        $this->model['inputGate'] = [];
        
        // Forget gate parameters
        $this->model['forgetGate'] = [];
        
        // Output gate parameters
        $this->model['outputGate'] = [];
        
        // Cell write parameters
        $this->model['cellWrite'] = [];
        
        // Output parameters
        $this->model['output'] = [];
        
        // Initialize parameters for each layer
        for ($l = 0; $l < count($hiddenSizes); $l++) {
            $prevSize = $l === 0 ? $inputSize : $hiddenSizes[$l - 1];
            $hiddenSize = $hiddenSizes[$l];
            
            // Input gate
            $this->model['inputGate'][$l] = [
                'Wxi' => $this->randomMatrix($hiddenSize, $prevSize),
                'Whi' => $this->randomMatrix($hiddenSize, $hiddenSize),
                'bi' => array_fill(0, $hiddenSize, 0)
            ];
            
            // Forget gate
            $this->model['forgetGate'][$l] = [
                'Wxf' => $this->randomMatrix($hiddenSize, $prevSize),
                'Whf' => $this->randomMatrix($hiddenSize, $hiddenSize),
                'bf' => array_fill(0, $hiddenSize, 1) // Initialize with 1 to avoid vanishing gradients
            ];
            
            // Output gate
            $this->model['outputGate'][$l] = [
                'Wxo' => $this->randomMatrix($hiddenSize, $prevSize),
                'Who' => $this->randomMatrix($hiddenSize, $hiddenSize),
                'bo' => array_fill(0, $hiddenSize, 0)
            ];
            
            // Cell write
            $this->model['cellWrite'][$l] = [
                'Wxc' => $this->randomMatrix($hiddenSize, $prevSize),
                'Whc' => $this->randomMatrix($hiddenSize, $hiddenSize),
                'bc' => array_fill(0, $hiddenSize, 0)
            ];
        }
        
        // Output layer
        $lastHiddenSize = end($hiddenSizes);
        $this->model['output'] = [
            'Why' => $this->randomMatrix($outputSize, $lastHiddenSize),
            'by' => array_fill(0, $outputSize, 0)
        ];
        
        // Initialize optimizer-specific parameters
        if ($this->options['praxis'] === 'adam') {
            $this->initializeAdam();
        }
        
        $this->isInitialized = true;
    }
    
    /**
     * Initialize Adam optimizer parameters
     */
    protected function initializeAdam(): void
    {
        $this->m = []; // First moment vector
        $this->v = []; // Second moment vector
        
        // Use a more memory-efficient approach for initializing Adam parameters
        foreach ($this->model as $gateType => $layers) {
            $this->m[$gateType] = [];
            $this->v[$gateType] = [];
            
            if ($gateType === 'output') {
                // Initialize output layer parameters
                $outputSize = count($this->model['output']['by']);
                $hiddenSize = count($this->model['output']['Why'][0]);
                
                // Use array_fill for more efficient memory allocation
                $this->m[$gateType]['Why'] = [];
                $this->v[$gateType]['Why'] = [];
                
                for ($i = 0; $i < $outputSize; $i++) {
                    $this->m[$gateType]['Why'][$i] = array_fill(0, $hiddenSize, 0);
                    $this->v[$gateType]['Why'][$i] = array_fill(0, $hiddenSize, 0);
                }
                
                $this->m[$gateType]['by'] = array_fill(0, $outputSize, 0);
                $this->v[$gateType]['by'] = array_fill(0, $outputSize, 0);
            } else {
                // Initialize gate parameters for each layer
                foreach ($layers as $l => $params) {
                    $this->m[$gateType][$l] = [];
                    $this->v[$gateType][$l] = [];
                    
                    foreach ($params as $paramName => $paramValue) {
                        // Check if it's a matrix or vector
                        if (is_array($paramValue) && isset($paramValue[0]) && is_array($paramValue[0])) {
                            $rows = count($paramValue);
                            $cols = count($paramValue[0]);
                            
                            $this->m[$gateType][$l][$paramName] = [];
                            $this->v[$gateType][$l][$paramName] = [];
                            
                            for ($i = 0; $i < $rows; $i++) {
                                $this->m[$gateType][$l][$paramName][$i] = array_fill(0, $cols, 0);
                                $this->v[$gateType][$l][$paramName][$i] = array_fill(0, $cols, 0);
                            }
                        } else {
                            // It's a vector
                            $size = count($paramValue);
                            $this->m[$gateType][$l][$paramName] = array_fill(0, $size, 0);
                            $this->v[$gateType][$l][$paramName] = array_fill(0, $size, 0);
                        }
                    }
                }
            }
        }
    }

    /**
     * Generate a random matrix with values between -0.1 and 0.1
     */
    protected function randomMatrix(int $rows, int $cols): array
    {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            $matrix[$i] = [];
            for ($j = 0; $j < $cols; $j++) {
                $matrix[$i][$j] = (mt_rand() / mt_getrandmax() * 0.2) - 0.1;
            }
        }
        return $matrix;
    }
    
    /**
     * Generate a zero matrix
     */
    protected function zeroMatrix(int $rows, int $cols): array
    {
        $matrix = [];
        for ($i = 0; $i < $rows; $i++) {
            $matrix[$i] = array_fill(0, $cols, 0);
        }
        return $matrix;
    }

    /**
     * Train the LSTM with the provided sequences
     */
    public function train(array $sequences, array $trainingOptions = []): array
    {
        
        $options = array_merge($this->options, $trainingOptions);
        $startTime = microtime(true);
        
        // Format data if needed
        if ($this->dataFormatter !== null && !empty($sequences)) {
            $sequences = $this->dataFormatter->formatSequences($sequences);
        }
        
        // Auto-detect input and output sizes if not provided
        if ($this->options['inputSize'] === 0 && !empty($sequences) && isset($sequences[0]['input']) && !empty($sequences[0]['input']) && isset($sequences[0]['input'][0])) {
            $this->options['inputSize'] = count($sequences[0]['input'][0]);
        }
        if ($this->options['outputSize'] === 0 && !empty($sequences) && isset($sequences[0]['output']) && !empty($sequences[0]['output']) && isset($sequences[0]['output'][0])) {
            $this->options['outputSize'] = count($sequences[0]['output'][0]);
        }
        
        if (!$this->isInitialized) {
            $this->initialize();
        }
        
        $error = 1;
        $iterations = 0;
        $this->errorLog = [];
        $t = 1; // Time step for Adam optimizer
        
        // For early stopping
        $bestError = INF;
        $bestModel = null;
        $patience = 10;
        $patienceCounter = 0;
        
        // For batch training
        $batchSize = min($options['batchSize'], count($sequences));
        
        while ($error > $options['errorThresh'] && $iterations < $options['iterations']) {
            // Check timeout
            if (microtime(true) - $startTime > $options['timeout']) {
                break;
            }
            
            // Shuffle sequences for each epoch
            shuffle($sequences);
            
            $totalError = 0;
            $batchCount = 0;
            
            // Process in batches
            for ($i = 0; $i < count($sequences); $i += $batchSize) {
                $batchSequences = array_slice($sequences, $i, $batchSize);
                $batchError = $this->trainBatch($batchSequences, $t);
                $totalError += $batchError;
                $batchCount++;
                $t++;
            }
            
            $error = $totalError / max(1, $batchCount);
            $this->errorLog[] = $error;
            $iterations++;
            
            // Learning rate decay
            $this->options['learningRate'] *= $options['decayRate'];
            
            // Logging
            if ($options['log'] && $iterations % $options['logPeriod'] === 0) {
                echo "Iteration: $iterations, Error: $error\n";
            }
            
            // Early stopping
            if ($error < $bestError) {
                $bestError = $error;
                $bestModel = $this->deepCopy($this->model);
                $patienceCounter = 0;
            } else {
                $patienceCounter++;
                if ($patienceCounter >= $patience) {
                    // Restore best model
                    $this->model = $bestModel;
                    break;
                }
            }
            
            // Callback
            if (isset($options['callback']) && $iterations % $options['callbackPeriod'] === 0) {
                $options['callback']([
                    'iterations' => $iterations,
                    'error' => $error
                ]);
            }
        }
        
        $trainingTime = microtime(true) - $startTime;
        
        $this->trainStats = [
            'error' => $error,
            'iterations' => $iterations,
            'time' => $trainingTime,
            'errorLog' => $this->errorLog
        ];
        
        return $this->trainStats;
    }
    
    /**
     * Train the LSTM with a batch of sequences
     */
    protected function trainBatch(array $sequences, int $t): float
    {
        $totalError = 0;
        $gradients = $this->initializeGradients();
        $sequenceCount = 0;
        
        foreach ($sequences as $sequence) {
            // Skip invalid sequences
            if (!isset($sequence['input']) || !isset($sequence['output']) || 
                empty($sequence['input']) || empty($sequence['output'])) {
                continue;
            }
            
            $inputs = $sequence['input'];
            $targets = $sequence['output'];
            
            // Forward pass
            $states = $this->forwardSequence($inputs);
            
            // Calculate error
            $error = 0;
            $outputCount = 0;
            
            for ($i = 0; $i < count($states['outputs']); $i++) {
                if (isset($targets[$i])) {
                    for ($j = 0; $j < count($states['outputs'][$i]); $j++) {
                        if (isset($targets[$i][$j])) {
                            $error += pow($states['outputs'][$i][$j] - $targets[$i][$j], 2);
                            $outputCount++;
                        }
                    }
                }
            }
            
            $error = $outputCount > 0 ? $error / $outputCount : 0;
            
            // Backward pass
            $sequenceGradients = $this->backwardSequence($inputs, $states, $targets);
            
            // Accumulate gradients
            $this->accumulateGradients($gradients, $sequenceGradients);
            
            $totalError += $error;
            $sequenceCount++;
        }
        
        // Apply gradients using the selected optimizer
        if ($this->options['praxis'] === 'adam') {
            $this->updateWithAdam($gradients, $t, max(1, $sequenceCount));
        } else {
            $this->updateWithSGD($gradients, max(1, $sequenceCount));
        }
        
        return $sequenceCount > 0 ? $totalError / $sequenceCount : 0;
    }
    
    /**
     * Initialize gradient structures
     */
    protected function initializeGradients(): array
    {
        $gradients = [];
        
        // Deep copy of model structure with zeros
        foreach ($this->model as $gateType => $layers) {
            $gradients[$gateType] = [];
            
            if ($gateType === 'output') {
                $gradients[$gateType] = [
                    'Why' => $this->zeroMatrix(count($layers['Why']), count($layers['Why'][0])),
                    'by' => array_fill(0, count($layers['by']), 0)
                ];
            } else {
                foreach ($layers as $l => $params) {
                    $gradients[$gateType][$l] = [];
                    
                    foreach ($params as $paramName => $paramValue) {
                        if (is_array($paramValue) && isset($paramValue[0]) && is_array($paramValue[0])) {
                            $gradients[$gateType][$l][$paramName] = $this->zeroMatrix(count($paramValue), count($paramValue[0]));
                        } else {
                            $gradients[$gateType][$l][$paramName] = array_fill(0, count($paramValue), 0);
                        }
                    }
                }
            }
        }
        
        return $gradients;
    }
    
    /**
     * Accumulate gradients from multiple sequences
     */
    protected function accumulateGradients(array &$accumGradients, array $gradients): void
    {
        foreach ($gradients as $gateType => $layers) {
            if ($gateType === 'output') {
                for ($i = 0; $i < count($layers['Why']); $i++) {
                    for ($j = 0; $j < count($layers['Why'][$i]); $j++) {
                        $accumGradients[$gateType]['Why'][$i][$j] += $layers['Why'][$i][$j];
                    }
                }
                
                for ($i = 0; $i < count($layers['by']); $i++) {
                    $accumGradients[$gateType]['by'][$i] += $layers['by'][$i];
                }
            } else {
                foreach ($layers as $l => $params) {
                    foreach ($params as $paramName => $paramValue) {
                        if (is_array($paramValue) && isset($paramValue[0]) && is_array($paramValue[0])) {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                for ($j = 0; $j < count($paramValue[$i]); $j++) {
                                    $accumGradients[$gateType][$l][$paramName][$i][$j] += $paramValue[$i][$j];
                                }
                            }
                        } else {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                $accumGradients[$gateType][$l][$paramName][$i] += $paramValue[$i];
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Forward pass through the LSTM for a sequence
     */
    protected function forwardSequence(array $inputs): array
    {
        $T = count($inputs);
        $numLayers = count($this->options['hiddenLayers']);
        
        // Initialize states
        $states = [
            'hiddenStates' => [],
            'cellStates' => [],
            'inputGates' => [],
            'forgetGates' => [],
            'outputGates' => [],
            'cellWrites' => [],
            'outputs' => []
        ];
        
        for ($l = 0; $l < $numLayers; $l++) {
            $hiddenSize = $this->options['hiddenLayers'][$l];
            
            $states['hiddenStates'][$l] = [];
            $states['cellStates'][$l] = [];
            $states['inputGates'][$l] = [];
            $states['forgetGates'][$l] = [];
            $states['outputGates'][$l] = [];
            $states['cellWrites'][$l] = [];
            
            // Initialize with zeros
            $states['hiddenStates'][$l][0] = array_fill(0, $hiddenSize, 0);
            $states['cellStates'][$l][0] = array_fill(0, $hiddenSize, 0);
        }
        
        // Forward pass through time
        for ($t = 0; $t < $T; $t++) {
            // Skip if input doesn't exist
            if (!isset($inputs[$t])) {
                continue;
            }
            
            $x = $inputs[$t];
            
            // Process each layer
            for ($l = 0; $l < $numLayers; $l++) {
                // Make sure we have valid previous states
                $prevH = $t > 0 && isset($states['hiddenStates'][$l][$t - 1]) 
                    ? $states['hiddenStates'][$l][$t - 1] 
                    : $states['hiddenStates'][$l][0];
                $prevC = $t > 0 && isset($states['cellStates'][$l][$t - 1]) 
                    ? $states['cellStates'][$l][$t - 1] 
                    : $states['cellStates'][$l][0];
                
                // Input for this layer is either the original input or the output from the previous layer
                $layerInput = $l === 0 ? $x : (isset($states['hiddenStates'][$l - 1][$t]) ? $states['hiddenStates'][$l - 1][$t] : array_fill(0, $this->options['hiddenLayers'][$l-1], 0));
                
                // Calculate gates
                $inputGate = $this->calculateGate('inputGate', $l, $layerInput, $prevH);
                $forgetGate = $this->calculateGate('forgetGate', $l, $layerInput, $prevH);
                $outputGate = $this->calculateGate('outputGate', $l, $layerInput, $prevH);
                $cellWrite = $this->calculateGate('cellWrite', $l, $layerInput, $prevH, 'tanh');
                
                // Update cell state
                $cellState = [];
                for ($i = 0; $i < count($prevC); $i++) {
                    $cellState[$i] = $forgetGate[$i] * $prevC[$i] + $inputGate[$i] * $cellWrite[$i];
                }
                
                // Calculate hidden state
                $hiddenState = [];
                for ($i = 0; $i < count($cellState); $i++) {
                    $hiddenState[$i] = $outputGate[$i] * tanh($cellState[$i]);
                }
                
                // Store states
                $states['inputGates'][$l][$t] = $inputGate;
                $states['forgetGates'][$l][$t] = $forgetGate;
                $states['outputGates'][$l][$t] = $outputGate;
                $states['cellWrites'][$l][$t] = $cellWrite;
                $states['cellStates'][$l][$t] = $cellState;
                $states['hiddenStates'][$l][$t] = $hiddenState;
            }
            
            // Calculate output
            $lastLayerH = $states['hiddenStates'][$numLayers - 1][$t];
            $output = [];
            
            for ($i = 0; $i < count($this->model['output']['by']); $i++) {
                $sum = $this->model['output']['by'][$i];
                
                for ($j = 0; $j < count($lastLayerH); $j++) {
                    $sum += $this->model['output']['Why'][$i][$j] * $lastLayerH[$j];
                }
                
                $output[$i] = ($this->activation)($sum);
            }
            
            $states['outputs'][$t] = $output;
        }
        
        return $states;
    }
    
    /**
     * Calculate a gate value
     */
    protected function calculateGate(string $gateType, int $layer, array $x, array $h, string $activation = 'sigmoid'): array
    {
        $gate = [];
        
        // Check if the gate parameters exist
        if (!isset($this->model[$gateType]) || !isset($this->model[$gateType][$layer])) {
            // Return a zero-filled array of the same size as h
            return array_fill(0, count($h), 0);
        }
        
        $params = $this->model[$gateType][$layer];
        
        // Get the correct weight matrices based on gate type
        if ($gateType === 'inputGate') {
            $wxName = 'Wxi';
            $whName = 'Whi';
            $bName = 'bi';
        } elseif ($gateType === 'forgetGate') {
            $wxName = 'Wxf';
            $whName = 'Whf';
            $bName = 'bf';
        } elseif ($gateType === 'outputGate') {
            $wxName = 'Wxo';
            $whName = 'Who';
            $bName = 'bo';
        } elseif ($gateType === 'cellWrite') {
            $wxName = 'Wxc';
            $whName = 'Whc';
            $bName = 'bc';
        } else {
            throw new BrainException("Unknown gate type: $gateType");
        }
        
        // Check if all required parameters exist
        if (!isset($params[$wxName]) || !isset($params[$whName]) || !isset($params[$bName])) {
            return array_fill(0, count($h), 0);
        }
        
        // Weight matrices
        $Wx = $params[$wxName]; // e.g., Wxi, Wxf, Wxo, Wxc
        $Wh = $params[$whName]; // e.g., Whi, Whf, Who, Whc
        $b = $params[$bName];   // e.g., bi, bf, bo, bc
        
        for ($i = 0; $i < count($b); $i++) {
            $sum = $b[$i];
            
            // Input to gate
            for ($j = 0; $j < count($x); $j++) {
                if (isset($Wx[$i][$j])) {
                    $sum += $Wx[$i][$j] * $x[$j];
                }
            }
            
            // Hidden to gate
            for ($j = 0; $j < count($h); $j++) {
                if (isset($Wh[$i][$j])) {
                    $sum += $Wh[$i][$j] * $h[$j];
                }
            }
            
            // Apply activation
            if ($activation === 'sigmoid') {
                $gate[$i] = 1 / (1 + exp(-$sum));
            } else if ($activation === 'tanh') {
                $gate[$i] = tanh($sum);
            }
        }
        
        return $gate;
    }
    
    /**
     * Backward pass through the LSTM for a sequence
     */
    protected function backwardSequence(array $inputs, array $states, array $targets): array
    {
        $T = count($inputs);
        $numLayers = count($this->options['hiddenLayers']);
        $gradients = $this->initializeGradients();
        
        // Initialize deltas
        $dOutput = [];
        $dHiddenStates = [];
        $dCellStates = [];
        
        for ($l = 0; $l < $numLayers; $l++) {
            $hiddenSize = $this->options['hiddenLayers'][$l];
            $dHiddenStates[$l] = [];
            $dCellStates[$l] = [];
            
            for ($t = 0; $t < $T; $t++) {
                $dHiddenStates[$l][$t] = array_fill(0, $hiddenSize, 0);
                $dCellStates[$l][$t] = array_fill(0, $hiddenSize, 0);
            }
        }
        
        // Backward pass through time
        for ($t = $T - 1; $t >= 0; $t--) {
            // Skip if output or target doesn't exist
            if (!isset($states['outputs'][$t]) || !isset($targets[$t])) {
                continue;
            }
            
            // Output layer gradients
            $dOutput = [];
            for ($i = 0; $i < count($states['outputs'][$t]); $i++) {
                if (isset($targets[$t][$i])) {
                    $dOutput[$i] = $states['outputs'][$t][$i] - $targets[$t][$i];
                    
                    // Gradient for output weights
                    for ($j = 0; $j < count($states['hiddenStates'][$numLayers - 1][$t]); $j++) {
                        $gradients['output']['Why'][$i][$j] += $dOutput[$i] * $states['hiddenStates'][$numLayers - 1][$t][$j];
                    }
                    
                    // Gradient for output bias
                    $gradients['output']['by'][$i] += $dOutput[$i];
                }
            }
            
            // Propagate error to last hidden layer
            for ($i = 0; $i < count($states['hiddenStates'][$numLayers - 1][$t]); $i++) {
                for ($j = 0; $j < count($dOutput); $j++) {
                    if (isset($this->model['output']['Why'][$j][$i])) {
                        $dHiddenStates[$numLayers - 1][$t][$i] += $this->model['output']['Why'][$j][$i] * $dOutput[$j];
                    }
                }
            }
            
            // Backward through layers
            for ($l = $numLayers - 1; $l >= 0; $l--) {
                // Add gradient from the next timestep if not the last timestep
                if ($t < $T - 1) {
                    for ($i = 0; $i < count($dHiddenStates[$l][$t]); $i++) {
                        if (isset($dHiddenStates[$l][$t + 1][$i])) {
                            $dHiddenStates[$l][$t][$i] += $dHiddenStates[$l][$t + 1][$i];
                        }
                        if (isset($dCellStates[$l][$t + 1][$i])) {
                            $dCellStates[$l][$t][$i] += $dCellStates[$l][$t + 1][$i];
                        }
                    }
                }
                
                // Add gradient from the next layer if not the last layer
                if ($l < $numLayers - 1) {
                    for ($i = 0; $i < count($dHiddenStates[$l][$t]); $i++) {
                        if (isset($dHiddenStates[$l + 1][$t][$i])) {
                            for ($j = 0; $j < count($this->model['inputGate'][$l + 1]['Wxi']); $j++) {
                                if (isset($this->model['inputGate'][$l + 1]['Wxi'][$j][$i])) {
                                    $dHiddenStates[$l][$t][$i] += $this->model['inputGate'][$l + 1]['Wxi'][$j][$i] * $dHiddenStates[$l + 1][$t][$i];
                                }
                                if (isset($this->model['forgetGate'][$l + 1]['Wxf'][$j][$i])) {
                                    $dHiddenStates[$l][$t][$i] += $this->model['forgetGate'][$l + 1]['Wxf'][$j][$i] * $dHiddenStates[$l + 1][$t][$i];
                                }
                                if (isset($this->model['outputGate'][$l + 1]['Wxo'][$j][$i])) {
                                    $dHiddenStates[$l][$t][$i] += $this->model['outputGate'][$l + 1]['Wxo'][$j][$i] * $dHiddenStates[$l + 1][$t][$i];
                                }
                                if (isset($this->model['cellWrite'][$l + 1]['Wxc'][$j][$i])) {
                                    $dHiddenStates[$l][$t][$i] += $this->model['cellWrite'][$l + 1]['Wxc'][$j][$i] * $dHiddenStates[$l + 1][$t][$i];
                                }
                            }
                        }
                    }
                }
                
                // Backprop through LSTM gates
                if (isset($states['hiddenStates'][$l][$t])) {
                    $h = $states['hiddenStates'][$l][$t];
                    $c = isset($states['cellStates'][$l][$t]) ? $states['cellStates'][$l][$t] : array_fill(0, count($h), 0);
                    $prevC = $t > 0 && isset($states['cellStates'][$l][$t - 1]) 
                        ? $states['cellStates'][$l][$t - 1] 
                        : array_fill(0, count($c), 0);
                    $i = isset($states['inputGates'][$l][$t]) ? $states['inputGates'][$l][$t] : array_fill(0, count($h), 0);
                    $f = isset($states['forgetGates'][$l][$t]) ? $states['forgetGates'][$l][$t] : array_fill(0, count($h), 0);
                    $o = isset($states['outputGates'][$l][$t]) ? $states['outputGates'][$l][$t] : array_fill(0, count($h), 0);
                    $g = isset($states['cellWrites'][$l][$t]) ? $states['cellWrites'][$l][$t] : array_fill(0, count($h), 0);
                
                    // Gradient for output gate
                    $dOutputGate = [];
                    for ($j = 0; $j < count($o); $j++) {
                        $dOutputGate[$j] = $dHiddenStates[$l][$t][$j] * tanh($c[$j]) * $o[$j] * (1 - $o[$j]);
                    }
                    
                    // Gradient for cell state
                    $dCellState = [];
                    for ($j = 0; $j < count($c); $j++) {
                        $dCellState[$j] = $dHiddenStates[$l][$t][$j] * $o[$j] * (1 - pow(tanh($c[$j]), 2));
                        $dCellState[$j] += $dCellStates[$l][$t][$j];
                    }
                    
                    // Gradient for input gate
                    $dInputGate = [];
                    for ($j = 0; $j < count($i); $j++) {
                        $dInputGate[$j] = $dCellState[$j] * $g[$j] * $i[$j] * (1 - $i[$j]);
                    }
                    
                    // Gradient for forget gate
                    $dForgetGate = [];
                    for ($j = 0; $j < count($f); $j++) {
                        $dForgetGate[$j] = $dCellState[$j] * $prevC[$j] * $f[$j] * (1 - $f[$j]);
                    }
                    
                    // Gradient for cell write
                    $dCellWrite = [];
                    for ($j = 0; $j < count($g); $j++) {
                        $dCellWrite[$j] = $dCellState[$j] * $i[$j] * (1 - pow($g[$j], 2));
                    }
                    
                    // Gradient for previous cell state
                    if ($t > 0) {
                        for ($j = 0; $j < count($prevC); $j++) {
                            if (isset($dCellStates[$l][$t - 1][$j])) {
                                $dCellStates[$l][$t - 1][$j] += $dCellState[$j] * $f[$j];
                            }
                        }
                    }
                    
                    // Get input for this layer
                    $x = $l === 0 ? $inputs[$t] : (isset($states['hiddenStates'][$l - 1][$t]) ? $states['hiddenStates'][$l - 1][$t] : array_fill(0, $this->options['hiddenLayers'][$l-1], 0));
                    $prevH = $t > 0 && isset($states['hiddenStates'][$l][$t - 1]) 
                        ? $states['hiddenStates'][$l][$t - 1] 
                        : array_fill(0, count($h), 0);
                    
                    // Update gradients for this layer
                    $this->updateGateGradients('inputGate', $l, $x, $prevH, $dInputGate, $gradients);
                    $this->updateGateGradients('forgetGate', $l, $x, $prevH, $dForgetGate, $gradients);
                    $this->updateGateGradients('outputGate', $l, $x, $prevH, $dOutputGate, $gradients);
                    $this->updateGateGradients('cellWrite', $l, $x, $prevH, $dCellWrite, $gradients);
                    
                    // Propagate gradients to previous hidden state
                    if ($t > 0) {
                        for ($j = 0; $j < count($prevH); $j++) {
                            if (isset($dHiddenStates[$l][$t - 1]) && isset($dHiddenStates[$l][$t - 1][$j])) {
                                $dHiddenStates[$l][$t - 1][$j] = 0;
                                
                                // From input gate
                                for ($k = 0; $k < count($dInputGate); $k++) {
                                    if (isset($this->model['inputGate'][$l]['Whi']) && 
                                        isset($this->model['inputGate'][$l]['Whi'][$k]) && 
                                        isset($this->model['inputGate'][$l]['Whi'][$k][$j])) {
                                        $dHiddenStates[$l][$t - 1][$j] += $this->model['inputGate'][$l]['Whi'][$k][$j] * $dInputGate[$k];
                                    }
                                }
                                
                                // From forget gate
                                for ($k = 0; $k < count($dForgetGate); $k++) {
                                    if (isset($this->model['forgetGate'][$l]['Whf']) && 
                                        isset($this->model['forgetGate'][$l]['Whf'][$k]) && 
                                        isset($this->model['forgetGate'][$l]['Whf'][$k][$j])) {
                                        $dHiddenStates[$l][$t - 1][$j] += $this->model['forgetGate'][$l]['Whf'][$k][$j] * $dForgetGate[$k];
                                    }
                                }
                                
                                // From output gate
                                for ($k = 0; $k < count($dOutputGate); $k++) {
                                    if (isset($this->model['outputGate'][$l]['Who']) && 
                                        isset($this->model['outputGate'][$l]['Who'][$k]) && 
                                        isset($this->model['outputGate'][$l]['Who'][$k][$j])) {
                                        $dHiddenStates[$l][$t - 1][$j] += $this->model['outputGate'][$l]['Who'][$k][$j] * $dOutputGate[$k];
                                    }
                                }
                                
                                // From cell write
                                for ($k = 0; $k < count($dCellWrite); $k++) {
                                    if (isset($this->model['cellWrite'][$l]['Whc']) && 
                                        isset($this->model['cellWrite'][$l]['Whc'][$k]) && 
                                        isset($this->model['cellWrite'][$l]['Whc'][$k][$j])) {
                                        $dHiddenStates[$l][$t - 1][$j] += $this->model['cellWrite'][$l]['Whc'][$k][$j] * $dCellWrite[$k];
                                    }
                                }
                            }
                        }
                    }
                    
                    // Propagate gradients to lower layer if not the first layer
                    if ($l > 0) {
                        for ($j = 0; $j < count($x); $j++) {
                            if (isset($dHiddenStates[$l - 1][$t]) && isset($dHiddenStates[$l - 1][$t][$j])) {
                                $dHiddenStates[$l - 1][$t][$j] = 0;
                                
                                // From input gate
                                for ($k = 0; $k < count($dInputGate); $k++) {
                                    if (isset($this->model['inputGate'][$l]['Wxi']) && 
                                        isset($this->model['inputGate'][$l]['Wxi'][$k]) && 
                                        isset($this->model['inputGate'][$l]['Wxi'][$k][$j])) {
                                        $dHiddenStates[$l - 1][$t][$j] += $this->model['inputGate'][$l]['Wxi'][$k][$j] * $dInputGate[$k];
                                    }
                                }
                                
                                // From forget gate
                                for ($k = 0; $k < count($dForgetGate); $k++) {
                                    if (isset($this->model['forgetGate'][$l]['Wxf']) && 
                                        isset($this->model['forgetGate'][$l]['Wxf'][$k]) && 
                                        isset($this->model['forgetGate'][$l]['Wxf'][$k][$j])) {
                                        $dHiddenStates[$l - 1][$t][$j] += $this->model['forgetGate'][$l]['Wxf'][$k][$j] * $dForgetGate[$k];
                                    }
                                }
                                
                                // From output gate
                                for ($k = 0; $k < count($dOutputGate); $k++) {
                                    if (isset($this->model['outputGate'][$l]['Wxo']) && 
                                        isset($this->model['outputGate'][$l]['Wxo'][$k]) && 
                                        isset($this->model['outputGate'][$l]['Wxo'][$k][$j])) {
                                        $dHiddenStates[$l - 1][$t][$j] += $this->model['outputGate'][$l]['Wxo'][$k][$j] * $dOutputGate[$k];
                                    }
                                }
                                
                                // From cell write
                                for ($k = 0; $k < count($dCellWrite); $k++) {
                                    if (isset($this->model['cellWrite'][$l]['Wxc']) && 
                                        isset($this->model['cellWrite'][$l]['Wxc'][$k]) && 
                                        isset($this->model['cellWrite'][$l]['Wxc'][$k][$j])) {
                                        $dHiddenStates[$l - 1][$t][$j] += $this->model['cellWrite'][$l]['Wxc'][$k][$j] * $dCellWrite[$k];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Clip gradients to prevent exploding gradients
        $this->clipGradients($gradients);
        
        return $gradients;
    }
    
    /**
     * Update gradients for a specific gate
     */
    protected function updateGateGradients(string $gateType, int $layer, array $x, array $h, array $dGate, array &$gradients): void
    {
        // Get the correct parameter names based on gate type
        if ($gateType === 'inputGate') {
            $wxName = 'Wxi';
            $whName = 'Whi';
            $bName = 'bi';
        } elseif ($gateType === 'forgetGate') {
            $wxName = 'Wxf';
            $whName = 'Whf';
            $bName = 'bf';
        } elseif ($gateType === 'outputGate') {
            $wxName = 'Wxo';
            $whName = 'Who';
            $bName = 'bo';
        } elseif ($gateType === 'cellWrite') {
            $wxName = 'Wxc';
            $whName = 'Whc';
            $bName = 'bc';
        } else {
            throw new BrainException("Unknown gate type: $gateType");
        }
        
        // Check if the gate parameters exist
        if (!isset($gradients[$gateType][$layer]) || 
            !isset($gradients[$gateType][$layer][$wxName]) || 
            !isset($gradients[$gateType][$layer][$whName]) || 
            !isset($gradients[$gateType][$layer][$bName])) {
            return;
        }
        
        // Update bias gradients
        for ($i = 0; $i < count($dGate); $i++) {
            if (isset($gradients[$gateType][$layer][$bName][$i])) {
                $gradients[$gateType][$layer][$bName][$i] += $dGate[$i];
            }
        }
        
        // Update weight gradients for input
        for ($i = 0; $i < count($dGate); $i++) {
            for ($j = 0; $j < count($x); $j++) {
                if (isset($gradients[$gateType][$layer][$wxName][$i][$j])) {
                    $gradients[$gateType][$layer][$wxName][$i][$j] += $dGate[$i] * $x[$j];
                }
            }
        }
        
        // Update weight gradients for hidden state
        for ($i = 0; $i < count($dGate); $i++) {
            for ($j = 0; $j < count($h); $j++) {
                if (isset($gradients[$gateType][$layer][$whName][$i][$j])) {
                    $gradients[$gateType][$layer][$whName][$i][$j] += $dGate[$i] * $h[$j];
                }
            }
        }
    }
    
    /**
     * Clip gradients to prevent exploding gradients
     */
    protected function clipGradients(array &$gradients): void
    {
        $clipValue = $this->options['clipGradient'];
        
        foreach ($gradients as $gateType => $layers) {
            if ($gateType === 'output') {
                for ($i = 0; $i < count($layers['Why']); $i++) {
                    for ($j = 0; $j < count($layers['Why'][$i]); $j++) {
                        if ($layers['Why'][$i][$j] > $clipValue) {
                            $gradients[$gateType]['Why'][$i][$j] = $clipValue;
                        } elseif ($layers['Why'][$i][$j] < -$clipValue) {
                            $gradients[$gateType]['Why'][$i][$j] = -$clipValue;
                        }
                    }
                }
                
                for ($i = 0; $i < count($layers['by']); $i++) {
                    if ($layers['by'][$i] > $clipValue) {
                        $gradients[$gateType]['by'][$i] = $clipValue;
                    } elseif ($layers['by'][$i] < -$clipValue) {
                        $gradients[$gateType]['by'][$i] = -$clipValue;
                    }
                }
            } else {
                foreach ($layers as $l => $params) {
                    foreach ($params as $paramName => $paramValue) {
                        if (is_array($paramValue) && isset($paramValue[0]) && is_array($paramValue[0])) {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                for ($j = 0; $j < count($paramValue[$i]); $j++) {
                                    if ($paramValue[$i][$j] > $clipValue) {
                                        $gradients[$gateType][$l][$paramName][$i][$j] = $clipValue;
                                    } elseif ($paramValue[$i][$j] < -$clipValue) {
                                        $gradients[$gateType][$l][$paramName][$i][$j] = -$clipValue;
                                    }
                                }
                            }
                        } else {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                if ($paramValue[$i] > $clipValue) {
                                    $gradients[$gateType][$l][$paramName][$i] = $clipValue;
                                } elseif ($paramValue[$i] < -$clipValue) {
                                    $gradients[$gateType][$l][$paramName][$i] = -$clipValue;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Update weights and biases using Stochastic Gradient Descent
     */
    protected function updateWithSGD(array $gradients, int $batchSize): void
    {
        $learningRate = $this->options['learningRate'] / $batchSize;
        
        foreach ($gradients as $gateType => $layers) {
            if ($gateType === 'output') {
                for ($i = 0; $i < count($layers['Why']); $i++) {
                    for ($j = 0; $j < count($layers['Why'][$i]); $j++) {
                        $this->model[$gateType]['Why'][$i][$j] -= $learningRate * $layers['Why'][$i][$j];
                    }
                }
                
                for ($i = 0; $i < count($layers['by']); $i++) {
                    $this->model[$gateType]['by'][$i] -= $learningRate * $layers['by'][$i];
                }
            } else {
                foreach ($layers as $l => $params) {
                    foreach ($params as $paramName => $paramValue) {
                        if (is_array($paramValue) && isset($paramValue[0]) && is_array($paramValue[0])) {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                for ($j = 0; $j < count($paramValue[$i]); $j++) {
                                    $this->model[$gateType][$l][$paramName][$i][$j] -= $learningRate * $paramValue[$i][$j];
                                }
                            }
                        } else {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                $this->model[$gateType][$l][$paramName][$i] -= $learningRate * $paramValue[$i];
                            }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Update weights and biases using Adam optimizer
     */
    protected function updateWithAdam(array $gradients, int $t, int $batchSize): void
    {
        $learningRate = $this->options['learningRate'];
        $beta1 = $this->options['beta1'];
        $beta2 = $this->options['beta2'];
        $epsilon = $this->options['epsilon'];
        
        // Bias correction
        $biasCorrection1 = 1 - pow($beta1, $t);
        $biasCorrection2 = 1 - pow($beta2, $t);
        $alphat = $learningRate * sqrt($biasCorrection2) / $biasCorrection1;
        
        foreach ($gradients as $gateType => $layers) {
            if ($gateType === 'output') {
                for ($i = 0; $i < count($layers['Why']); $i++) {
                    for ($j = 0; $j < count($layers['Why'][$i]); $j++) {
                        $gradient = $layers['Why'][$i][$j] / $batchSize;
                        $this->m[$gateType]['Why'][$i][$j] = $beta1 * $this->m[$gateType]['Why'][$i][$j] + (1 - $beta1) * $gradient;
                        $this->v[$gateType]['Why'][$i][$j] = $beta2 * $this->v[$gateType]['Why'][$i][$j] + (1 - $beta2) * pow($gradient, 2);
                        
                        $this->model[$gateType]['Why'][$i][$j] -= $alphat * $this->m[$gateType]['Why'][$i][$j] / (sqrt($this->v[$gateType]['Why'][$i][$j]) + $epsilon);
                    }
                }
                
                for ($i = 0; $i < count($layers['by']); $i++) {
                    $gradient = $layers['by'][$i] / $batchSize;
                    $this->m[$gateType]['by'][$i] = $beta1 * $this->m[$gateType]['by'][$i] + (1 - $beta1) * $gradient;
                    $this->v[$gateType]['by'][$i] = $beta2 * $this->v[$gateType]['by'][$i] + (1 - $beta2) * pow($gradient, 2);
                    
                    $this->model[$gateType]['by'][$i] -= $alphat * $this->m[$gateType]['by'][$i] / (sqrt($this->v[$gateType]['by'][$i]) + $epsilon);
                }
            } else {
                foreach ($layers as $l => $params) {
                    foreach ($params as $paramName => $paramValue) {
                        if (is_array($paramValue) && isset($paramValue[0]) && is_array($paramValue[0])) {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                for ($j = 0; $j < count($paramValue[$i]); $j++) {
                                    $gradient = $paramValue[$i][$j] / $batchSize;
                                    $this->m[$gateType][$l][$paramName][$i][$j] = $beta1 * $this->m[$gateType][$l][$paramName][$i][$j] + (1 - $beta1) * $gradient;
                                    $this->v[$gateType][$l][$paramName][$i][$j] = $beta2 * $this->v[$gateType][$l][$paramName][$i][$j] + (1 - $beta2) * pow($gradient, 2);
                                    
                                    $this->model[$gateType][$l][$paramName][$i][$j] -= $alphat * $this->m[$gateType][$l][$paramName][$i][$j] / (sqrt($this->v[$gateType][$l][$paramName][$i][$j]) + $epsilon);
                                }
                            }
                        } else {
                            for ($i = 0; $i < count($paramValue); $i++) {
                                $gradient = $paramValue[$i] / $batchSize;
                                $this->m[$gateType][$l][$paramName][$i] = $beta1 * $this->m[$gateType][$l][$paramName][$i] + (1 - $beta1) * $gradient;
                                $this->v[$gateType][$l][$paramName][$i] = $beta2 * $this->v[$gateType][$l][$paramName][$i] + (1 - $beta2) * pow($gradient, 2);
                                
                                $this->model[$gateType][$l][$paramName][$i] -= $alphat * $this->m[$gateType][$l][$paramName][$i] / (sqrt($this->v[$gateType][$l][$paramName][$i]) + $epsilon);
                            }
                        }
                    }
                }
            }
        }
    }

    /**
     * Run the LSTM with the provided input sequence
     */
    public function run(array $inputSequence): array
    {
        // Format input if needed
        if ($this->dataFormatter !== null) {
            $inputSequence = $this->dataFormatter->formatInputSequence($inputSequence);
        }
        
        $states = $this->forwardSequence($inputSequence);
        
        // Format output if needed
        if ($this->dataFormatter !== null) {
            $states['outputs'] = $this->dataFormatter->formatOutputSequence($states['outputs']);
        }
        
        return $states['outputs'];
    }
    
    /**
     * Generate a sequence of specified length
     */
    public function generate(array $seed, int $length): array
    {
        if (!$this->isInitialized) {
            throw new BrainException('LSTM must be trained before generating sequences');
        }
        
        $sequence = $seed;
        $generated = [];
        
        for ($i = 0; $i < $length; $i++) {
            $output = $this->run($sequence);
            $lastOutput = end($output);
            $generated[] = $lastOutput;
            
            // Add the generated output to the sequence for the next iteration
            $sequence[] = $lastOutput;
            
            // Optionally, remove the first element to keep sequence length constant
            if (count($sequence) > count($seed)) {
                array_shift($sequence);
            }
        }
        
        return $generated;
    }

    /**
     * Convert the LSTM to JSON
     */
    public function toJSON(): string
    {
        $data = [
            'type' => 'LSTM',
            'options' => $this->options,
            'model' => $this->model,
            'trainStats' => $this->trainStats
        ];
        
        if ($this->dataFormatter !== null) {
            $data['dataFormatter'] = $this->dataFormatter->toArray();
        }
        
        return json_encode($data);
    }

    /**
     * Create an LSTM from JSON
     */
    public static function fromJSON(string $json): self
    {
        $data = json_decode($json, true);
        
        if (!isset($data['type']) || $data['type'] !== 'LSTM') {
            throw new BrainException('Invalid JSON format for LSTM');
        }
        
        $lstm = new self($data['options']);
        $lstm->model = $data['model'];
        $lstm->trainStats = $data['trainStats'] ?? [];
        $lstm->isInitialized = true;
        
        if (isset($data['dataFormatter'])) {
            $lstm->dataFormatter = DataFormatter::fromArray($data['dataFormatter']);
        }
        
        return $lstm;
    }
    
    /**
     * Create a deep copy of an array
     */
    protected function deepCopy(array $array): array
    {
        $copy = [];
        
        foreach ($array as $key => $value) {
            if (is_array($value)) {
                $copy[$key] = $this->deepCopy($value);
            } else {
                $copy[$key] = $value;
            }
        }
        
        return $copy;
    }
    
    /**
     * Get training statistics
     */
    public function getTrainStats(): array
    {
        return $this->trainStats;
    }
    
    /**
     * Get error log
     */
    public function getErrorLog(): array
    {
        return $this->errorLog;
    }
}
