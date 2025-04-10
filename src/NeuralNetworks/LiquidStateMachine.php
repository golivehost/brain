<?php
/**
 * Liquid State Machine (LSM) implementation for PHP
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\NeuralNetworks;

use GoLiveHost\Brain\Utilities\Matrix;
use GoLiveHost\Brain\Activation\ActivationFactory;
use GoLiveHost\Brain\Exceptions\BrainException;

class LiquidStateMachine
{
    protected array $options;
    protected array $reservoir = [];
    protected array $readoutWeights = [];
    protected array $readoutBias = [];
    protected $activation;
    protected $activationDerivative;
    protected array $errorLog = [];
    protected array $trainStats = [];
    protected bool $isInitialized = false;

    /**
     * @param array $options Configuration options for the LSM
     */
    public function __construct(array $options = [])
    {
        // Default options
        $this->options = array_merge([
            'inputSize' => 0,
            'reservoirSize' => 100,
            'outputSize' => 0,
            'connectivity' => 0.1,
            'spectralRadius' => 0.9,
            'leakingRate' => 0.3,
            'activation' => 'tanh',
            'readoutActivation' => 'sigmoid',
            'learningRate' => 0.01,
            'iterations' => 1000,
            'errorThresh' => 0.005,
            'regularization' => 0.0001,
            'washoutPeriod' => 10
        ], $options);

        $this->setActivation($this->options['activation']);
    }

    /**
     * Set the activation function and its derivative
     */
    protected function setActivation(string $name): void
    {
        $activation = ActivationFactory::create($name);
        
        $this->activation = $activation->getFunction();
        $this->activationDerivative = $activation->getDerivative();
        
        $readoutActivation = ActivationFactory::create($this->options['readoutActivation']);
        $this->readoutActivation = $readoutActivation->getFunction();
        $this->readoutActivationDerivative = $readoutActivation->getDerivative();
    }

    /**
     * Initialize the LSM with random weights
     */
    public function initialize(): void
    {
        $inputSize = $this->options['inputSize'];
        $reservoirSize = $this->options['reservoirSize'];
        $outputSize = $this->options['outputSize'];
        $connectivity = $this->options['connectivity'];
        $spectralRadius = $this->options['spectralRadius'];
        
        // Initialize input weights (fully connected)
        $this->inputWeights = [];
        for ($i = 0; $i < $reservoirSize; $i++) {
            $this->inputWeights[$i] = [];
            for ($j = 0; $j < $inputSize; $j++) {
                $this->inputWeights[$i][$j] = $this->gaussianRandom(0, 1 / sqrt($inputSize));
            }
        }
        
        // Initialize reservoir weights (sparse connectivity)
        $this->reservoirWeights = [];
        for ($i = 0; $i < $reservoirSize; $i++) {
            $this->reservoirWeights[$i] = [];
            for ($j = 0; $j < $reservoirSize; $j++) {
                // Sparse connectivity
                if (mt_rand() / mt_getrandmax() < $connectivity) {
                    $this->reservoirWeights[$i][$j] = $this->gaussianRandom(0, 1);
                } else {
                    $this->reservoirWeights[$i][$j] = 0;
                }
            }
        }
        
        // Scale reservoir weights to desired spectral radius
        $this->scaleReservoirWeights($spectralRadius);
        
        // Initialize readout weights
        $this->readoutWeights = [];
        for ($i = 0; $i < $outputSize; $i++) {
            $this->readoutWeights[$i] = [];
            for ($j = 0; $j < $reservoirSize; $j++) {
                $this->readoutWeights[$i][$j] = $this->gaussianRandom(0, 1 / sqrt($reservoirSize));
            }
        }
        
        // Initialize readout bias
        $this->readoutBias = array_fill(0, $outputSize, 0);
        
        // Initialize reservoir state
        $this->reservoirState = array_fill(0, $reservoirSize, 0);
        
        $this->isInitialized = true;
    }
    
    /**
     * Scale reservoir weights to achieve desired spectral radius
     */
    protected function scaleReservoirWeights(float $targetRadius): void
    {
        // Calculate the current spectral radius (largest eigenvalue)
        $currentRadius = $this->calculateSpectralRadius();
        
        // Scale the weights
        if ($currentRadius > 0) {
            $scaleFactor = $targetRadius / $currentRadius;
            
            for ($i = 0; $i < count($this->reservoirWeights); $i++) {
                for ($j = 0; $j < count($this->reservoirWeights[$i]); $j++) {
                    $this->reservoirWeights[$i][$j] *= $scaleFactor;
                }
            }
        }
    }
    
    /**
     * Calculate the spectral radius (largest eigenvalue) of the reservoir weights
     */
    protected function calculateSpectralRadius(): float
    {
        // This is a simplified approximation using the power method
        $size = count($this->reservoirWeights);
        $v = array_fill(0, $size, 1 / sqrt($size)); // Initial normalized vector
        
        for ($iter = 0; $iter < 100; $iter++) {
            // Matrix-vector multiplication
            $Av = array_fill(0, $size, 0);
            for ($i = 0; $i < $size; $i++) {
                for ($j = 0; $j < $size; $j++) {
                    $Av[$i] += $this->reservoirWeights[$i][$j] * $v[$j];
                }
            }
            
            // Calculate the norm
            $norm = 0;
            foreach ($Av as $val) {
                $norm += $val * $val;
            }
            $norm = sqrt($norm);
            
            // Normalize
            if ($norm > 0) {
                for ($i = 0; $i < $size; $i++) {
                    $v[$i] = $Av[$i] / $norm;
                }
            }
        }
        
        // Calculate Rayleigh quotient
        $numerator = 0;
        for ($i = 0; $i < $size; $i++) {
            $sum = 0;
            for ($j = 0; $j < $size; $j++) {
                $sum += $this->reservoirWeights[$i][$j] * $v[$j];
            }
            $numerator += $v[$i] * $sum;
        }
        
        $denominator = 0;
        foreach ($v as $val) {
            $denominator += $val * $val;
        }
        
        return $denominator > 0 ? abs($numerator / $denominator) : 0;
    }
    
    /**
     * Generate a random number from a Gaussian distribution
     */
    protected function gaussianRandom(float $mean = 0, float $stdDev = 1): float
    {
        // Box-Muller transform
        $u1 = 1 - mt_rand() / mt_getrandmax();
        $u2 = 1 - mt_rand() / mt_getrandmax();
        $randStdNormal = sqrt(-2 * log($u1)) * sin(2 * M_PI * $u2);
        return $mean + $stdDev * $randStdNormal;
    }

    /**
     * Train the LSM with the provided sequences
     */
    public function train(array $sequences, array $trainingOptions = []): array
    {
        $options = array_merge($this->options, $trainingOptions);
        $startTime = microtime(true);
        
        // Check if sequences is empty
        if (empty($sequences)) {
            return [
                'error' => 0,
                'time' => 0
            ];
        }
        
        // Auto-detect input and output sizes if not provided
        if ($this->options['inputSize'] === 0) {
            $this->options['inputSize'] = count($sequences[0]['input'][0]);
        }
        if ($this->options['outputSize'] === 0) {
            $this->options['outputSize'] = count($sequences[0]['output'][0]);
        }
        
        if (!$this->isInitialized) {
            $this->initialize();
        }
        
        // Collect reservoir states for all sequences
        $allStates = [];
        $allTargets = [];
        
        foreach ($sequences as $sequence) {
            // Skip invalid sequences
            if (!isset($sequence['input']) || !isset($sequence['output']) || 
                empty($sequence['input']) || empty($sequence['output'])) {
                continue;
            }
            
            $inputs = $sequence['input'];
            $targets = $sequence['output'];
            
            // Reset reservoir state
            $this->resetState();
            
            // Washout period (discard initial transient)
            for ($i = 0; $i < $options['washoutPeriod'] && $i < count($inputs); $i++) {
                $this->update($inputs[$i]);
            }
            
            // Collect states and targets after washout
            for ($i = $options['washoutPeriod']; $i < count($inputs); $i++) {
                if (isset($inputs[$i]) && isset($targets[$i])) {
                    $state = $this->update($inputs[$i]);
                    $allStates[] = $state;
                    $allTargets[] = $targets[$i];
                }
            }
        }
        
        // Train readout weights using ridge regression or gradient descent
        if (count($allStates) > 0) {
            if ($options['regularization'] > 0) {
                $this->trainReadoutRidge($allStates, $allTargets, $options['regularization']);
            } else {
                $this->trainReadoutGradient($allStates, $allTargets, $options);
            }
        }
        
        // Calculate final error
        $error = $this->calculateError($allStates, $allTargets);
        
        $trainingTime = microtime(true) - $startTime;
        
        $this->trainStats = [
            'error' => $error,
            'time' => $trainingTime
        ];
        
        return $this->trainStats;
    }
    
    /**
     * Reset the reservoir state
     */
    public function resetState(): void
    {
        $this->reservoirState = array_fill(0, $this->options['reservoirSize'], 0);
    }
    
    /**
     * Update the reservoir state with an input
     */
    public function update(array $input): array
    {
        if (!$this->isInitialized) {
            throw new BrainException('LSM must be initialized before updating');
        }
        
        $reservoirSize = $this->options['reservoirSize'];
        $leakingRate = $this->options['leakingRate'];
        
        // Calculate input contribution
        $inputContribution = array_fill(0, $reservoirSize, 0);
        for ($i = 0; $i < $reservoirSize; $i++) {
            for ($j = 0; $j < count($input); $j++) {
                $inputContribution[$i] += $this->inputWeights[$i][$j] * $input[$j];
            }
        }
        
        // Calculate reservoir contribution
        $reservoirContribution = array_fill(0, $reservoirSize, 0);
        for ($i = 0; $i < $reservoirSize; $i++) {
            for ($j = 0; $j < $reservoirSize; $j++) {
                $reservoirContribution[$i] += $this->reservoirWeights[$i][$j] * $this->reservoirState[$j];
            }
        }
        
        // Update reservoir state with leaking rate
        $newState = [];
        for ($i = 0; $i < $reservoirSize; $i++) {
            $sum = $inputContribution[$i] + $reservoirContribution[$i];
            $activation = ($this->activation)($sum);
            $newState[$i] = (1 - $leakingRate) * $this->reservoirState[$i] + $leakingRate * $activation;
        }
        
        $this->reservoirState = $newState;
        
        return $this->reservoirState;
    }
    
    /**
     * Train readout weights using ridge regression
     */
    protected function trainReadoutRidge(array $states, array $targets, float $regularization): void
    {
        $outputSize = $this->options['outputSize'];
        $reservoirSize = $this->options['reservoirSize'];
        
        // Prepare matrices for ridge regression
        $X = $states; // States matrix
        $Y = $targets; // Targets matrix
        
        // Calculate X^T * X + lambda * I
        $XTX = array_fill(0, $reservoirSize, array_fill(0, $reservoirSize, 0));
        for ($i = 0; $i < $reservoirSize; $i++) {
            for ($j = 0; $j < $reservoirSize; $j++) {
                $sum = 0;
                for ($k = 0; $k < count($X); $k++) {
                    $sum += $X[$k][$i] * $X[$k][$j];
                }
                $XTX[$i][$j] = $sum;
                
                // Add regularization to diagonal
                if ($i === $j) {
                    $XTX[$i][$j] += $regularization;
                }
            }
        }
        
        // Calculate X^T * Y
        $XTY = array_fill(0, $reservoirSize, array_fill(0, $outputSize, 0));
        for ($i = 0; $i < $reservoirSize; $i++) {
            for ($j = 0; $j < $outputSize; $j++) {
                $sum = 0;
                for ($k = 0; $k < count($X); $k++) {
                    $sum += $X[$k][$i] * $Y[$k][$j];
                }
                $XTY[$i][$j] = $sum;
            }
        }
        
        // Solve the system (XTX)^-1 * XTY
        // This is a simplified approach; in practice, use a library for matrix inversion
        $inverse = $this->pseudoInverse($XTX);
        
        // Calculate W = (XTX)^-1 * XTY
        $W = array_fill(0, $outputSize, array_fill(0, $reservoirSize, 0));
        for ($i = 0; $i < $outputSize; $i++) {
            for ($j = 0; $j < $reservoirSize; $j++) {
                $sum = 0;
                for ($k = 0; $k < $reservoirSize; $k++) {
                    $sum += $inverse[$j][$k] * $XTY[$k][$i];
                }
                $W[$i][$j] = $sum;
            }
        }
        
        // Update readout weights
        $this->readoutWeights = $W;
        
        // Calculate bias
        $this->readoutBias = array_fill(0, $outputSize, 0);
        $meanTarget = array_fill(0, $outputSize, 0);
        $meanPrediction = array_fill(0, $outputSize, 0);
        
        for ($i = 0; $i < count($Y); $i++) {
            for ($j = 0; $j < $outputSize; $j++) {
                $meanTarget[$j] += $Y[$i][$j];
                
                $sum = 0;
                for ($k = 0; $k < $reservoirSize; $k++) {
                    $sum += $W[$j][$k] * $X[$i][$k];
                }
                $meanPrediction[$j] += $sum;
            }
        }
        
        for ($j = 0; $j < $outputSize; $j++) {
            $meanTarget[$j] /= count($Y);
            $meanPrediction[$j] /= count($Y);
            $this->readoutBias[$j] = $meanTarget[$j] - $meanPrediction[$j];
        }
    }
    
    /**
     * Calculate the pseudoinverse of a matrix
     */
    protected function pseudoInverse(array $matrix): array
    {
        $size = count($matrix);
        
        // This is a simplified pseudoinverse calculation
        // In practice, use a library for SVD or other methods
        
        // Identity matrix
        $I = [];
        for ($i = 0; $i < $size; $i++) {
            $I[$i] = [];
            for ($j = 0; $j < $size; $j++) {
                $I[$i][$j] = ($i === $j) ? 1 : 0;
            }
        }
        
        // Gauss-Jordan elimination
        $augmented = [];
        for ($i = 0; $i < $size; $i++) {
            $augmented[$i] = array_merge($matrix[$i], $I[$i]);
        }
        
        // Forward elimination
        for ($i = 0; $i < $size; $i++) {
            // Find pivot
            $maxRow = $i;
            $maxVal = abs($augmented[$i][$i]);
            
            for ($j = $i + 1; $j < $size; $j++) {
                if (abs($augmented[$j][$i]) > $maxVal) {
                    $maxRow = $j;
                    $maxVal = abs($augmented[$j][$i]);
                }
            }
            
            // Swap rows if needed
            if ($maxRow !== $i) {
                $temp = $augmented[$i];
                $augmented[$i] = $augmented[$maxRow];
                $augmented[$maxRow] = $temp;
            }
            
            // Scale pivot row
            $pivot = $augmented[$i][$i];
            if (abs($pivot) < 1e-10) {
                // Matrix is singular or nearly singular
                $pivot = 1e-10;
            }
            
            for ($j = 0; $j < 2 * $size; $j++) {
                $augmented[$i][$j] /= $pivot;
            }
            
            // Eliminate other rows
            for ($j = 0; $j < $size; $j++) {
                if ($j !== $i) {
                    $factor = $augmented[$j][$i];
                    
                    for ($k = 0; $k < 2 * $size; $k++) {
                        $augmented[$j][$k] -= $factor * $augmented[$i][$k];
                    }
                }
            }
        }
        
        // Extract inverse
        $inverse = [];
        for ($i = 0; $i < $size; $i++) {
            $inverse[$i] = array_slice($augmented[$i], $size, $size);
        }
        
        return $inverse;
    }
    
    /**
     * Train readout weights using gradient descent
     */
    protected function trainReadoutGradient(array $states, array $targets, array $options): void
    {
        $outputSize = $this->options['outputSize'];
        $reservoirSize = $this->options['reservoirSize'];
        $learningRate = $options['learningRate'];
        $iterations = $options['iterations'];
        $errorThresh = $options['errorThresh'];
        
        $error = 1;
        $iter = 0;
        
        while ($error > $errorThresh && $iter < $iterations) {
            $totalError = 0;
            
            // Shuffle data
            $indices = range(0, count($states) - 1);
            shuffle($indices);
            
            foreach ($indices as $idx) {
                $state = $states[$idx];
                $target = $targets[$idx];
                
                // Forward pass
                $output = $this->readout($state);
                
                // Calculate error
                $itemError = 0;
                $deltas = [];
                
                for ($i = 0; $i < $outputSize; $i++) {
                    $delta = $target[$i] - $output[$i];
                    $deltas[$i] = $delta * ($this->readoutActivationDerivative)($output[$i]);
                    $itemError += $delta * $delta;
                }
                $itemError /= $outputSize;
                $totalError += $itemError;
                
                // Update weights
                for ($i = 0; $i < $outputSize; $i++) {
                    for ($j = 0; $j < $reservoirSize; $j++) {
                        $this->readoutWeights[$i][$j] += $learningRate * $deltas[$i] * $state[$j];
                    }
                    $this->readoutBias[$i] += $learningRate * $deltas[$i];
                }
            }
            
            $error = $totalError / count($states);
            $iter++;
        }
    }
    
    /**
     * Calculate the output from the readout layer
     */
    protected function readout(array $state): array
    {
        $outputSize = $this->options['outputSize'];
        $output = [];
        
        for ($i = 0; $i < $outputSize; $i++) {
            $sum = $this->readoutBias[$i];
            
            for ($j = 0; $j < count($state); $j++) {
                $sum += $this->readoutWeights[$i][$j] * $state[$j];
            }
            
            $output[$i] = ($this->readoutActivation)($sum);
        }
        
        return $output;
    }
    
    /**
     * Calculate error on a dataset
     */
    protected function calculateError(array $states, array $targets): float
    {
        $totalError = 0;
        $count = 0;
        
        for ($i = 0; $i < count($states); $i++) {
            if (isset($targets[$i])) {
                $output = $this->readout($states[$i]);
                $target = $targets[$i];
                
                $itemError = 0;
                $itemCount = 0;
                
                for ($j = 0; $j < count($output); $j++) {
                    if (isset($target[$j])) {
                        $itemError += pow($output[$j] - $target[$j], 2);
                        $itemCount++;
                    }
                }
                
                if ($itemCount > 0) {
                    $itemError /= $itemCount;
                    $totalError += $itemError;
                    $count++;
                }
            }
        }
        
        // Prevent division by zero
        return $count > 0 ? $totalError / $count : 0;
    }

    /**
     * Run the LSM with the provided input sequence
     */
    public function run(array $inputSequence): array
    {
        if (!$this->isInitialized) {
            throw new BrainException('LSM must be initialized before running');
        }
        
        $this->resetState();
        $outputs = [];
        
        foreach ($inputSequence as $input) {
            $state = $this->update($input);
            $outputs[] = $this->readout($state);
        }
        
        return $outputs;
    }

    /**
     * Convert the LSM to JSON
     */
    public function toJSON(): string
    {
        $data = [
            'type' => 'LiquidStateMachine',
            'options' => $this->options,
            'inputWeights' => $this->inputWeights,
            'reservoirWeights' => $this->reservoirWeights,
            'readoutWeights' => $this->readoutWeights,
            'readoutBias' => $this->readoutBias,
            'trainStats' => $this->trainStats
        ];
        
        return json_encode($data);
    }

    /**
     * Create an LSM from JSON
     */
    public static function fromJSON(string $json): self
    {
        $data = json_decode($json, true);
        
        if (!isset($data['type']) || $data['type'] !== 'LiquidStateMachine') {
            throw new BrainException('Invalid JSON format for LiquidStateMachine');
        }
        
        $lsm = new self($data['options']);
        $lsm->inputWeights = $data['inputWeights'];
        $lsm->reservoirWeights = $data['reservoirWeights'];
        $lsm->readoutWeights = $data['readoutWeights'];
        $lsm->readoutBias = $data['readoutBias'];
        $lsm->trainStats = $data['trainStats'] ?? [];
        $lsm->isInitialized = true;
        
        return $lsm;
    }
    
    /**
     * Get training statistics
     */
    public function getTrainStats(): array
    {
        return $this->trainStats;
    }
}
