<?php
/**
 * Batch Normalization Layer
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Layers;

class BatchNormalization
{
    protected float $epsilon;
    protected float $momentum;
    protected array $gamma; // Scale parameter
    protected array $beta; // Shift parameter
    protected array $runningMean;
    protected array $runningVar;
    protected array $cache; // Cache for backward pass
    protected bool $isTraining;

    /**
     * @param int $size Size of the input
     * @param array $options Configuration options
     */
    public function __construct(int $size, array $options = [])
    {
        $this->epsilon = $options['epsilon'] ?? 1e-5;
        $this->momentum = $options['momentum'] ?? 0.9;
        $this->gamma = array_fill(0, $size, 1.0); // Initialize to ones
        $this->beta = array_fill(0, $size, 0.0); // Initialize to zeros
        $this->runningMean = array_fill(0, $size, 0.0);
        $this->runningVar = array_fill(0, $size, 1.0);
        $this->isTraining = false;
    }

    /**
     * Set training mode
     */
    public function setTraining(bool $isTraining): void
    {
        $this->isTraining = $isTraining;
    }

    /**
     * Forward pass
     */
    public function forward(array $input): array
    {
        if ($this->isTraining) {
            return $this->trainForward($input);
        } else {
            return $this->testForward($input);
        }
    }

    /**
     * Forward pass during training
     */
    protected function trainForward(array $input): array
    {
        $batchSize = count($input);
        $inputSize = count($input[0]);
        
        // Calculate batch mean
        $batchMean = array_fill(0, $inputSize, 0.0);
        foreach ($input as $sample) {
            for ($i = 0; $i < $inputSize; $i++) {
                $batchMean[$i] += $sample[$i];
            }
        }
        for ($i = 0; $i < $inputSize; $i++) {
            $batchMean[$i] /= $batchSize;
        }
        
        // Calculate batch variance
        $batchVar = array_fill(0, $inputSize, 0.0);
        foreach ($input as $sample) {
            for ($i = 0; $i < $inputSize; $i++) {
                $batchVar[$i] += pow($sample[$i] - $batchMean[$i], 2);
            }
        }
        for ($i = 0; $i < $inputSize; $i++) {
            $batchVar[$i] /= $batchSize;
        }
        
        // Update running statistics
        for ($i = 0; $i < $inputSize; $i++) {
            $this->runningMean[$i] = $this->momentum * $this->runningMean[$i] + (1 - $this->momentum) * $batchMean[$i];
            $this->runningVar[$i] = $this->momentum * $this->runningVar[$i] + (1 - $this->momentum) * $batchVar[$i];
        }
        
        // Normalize
        $xNorm = [];
        foreach ($input as $sample) {
            $normalizedSample = [];
            for ($i = 0; $i < $inputSize; $i++) {
                $normalizedSample[$i] = ($sample[$i] - $batchMean[$i]) / sqrt($batchVar[$i] + $this->epsilon);
            }
            $xNorm[] = $normalizedSample;
        }
        
        // Scale and shift
        $output = [];
        foreach ($xNorm as $normalizedSample) {
            $scaledSample = [];
            for ($i = 0; $i < $inputSize; $i++) {
                $scaledSample[$i] = $this->gamma[$i] * $normalizedSample[$i] + $this->beta[$i];
            }
            $output[] = $scaledSample;
        }
        
        // Cache values for backward pass
        $this->cache = [
            'input' => $input,
            'batchMean' => $batchMean,
            'batchVar' => $batchVar,
            'xNorm' => $xNorm,
            'gamma' => $this->gamma,
            'beta' => $this->beta,
            'epsilon' => $this->epsilon
        ];
        
        return $output;
    }

    /**
     * Forward pass during testing/inference
     */
    protected function testForward(array $input): array
    {
        $output = [];
        $inputSize = count($input[0]);
        
        foreach ($input as $sample) {
            $normalizedSample = [];
            for ($i = 0; $i < $inputSize; $i++) {
                $normalizedSample[$i] = ($sample[$i] - $this->runningMean[$i]) / sqrt($this->runningVar[$i] + $this->epsilon);
                $normalizedSample[$i] = $this->gamma[$i] * $normalizedSample[$i] + $this->beta[$i];
            }
            $output[] = $normalizedSample;
        }
        
        return $output;
    }

    /**
     * Backward pass
     */
    public function backward(array $dout): array
    {
        $cache = $this->cache;
        $input = $cache['input'];
        $batchMean = $cache['batchMean'];
        $batchVar = $cache['batchVar'];
        $xNorm = $cache['xNorm'];
        $gamma = $cache['gamma'];
        $epsilon = $cache['epsilon'];
        
        $batchSize = count($dout);
        $inputSize = count($dout[0]);
        
        // Initialize gradients
        $dgamma = array_fill(0, $inputSize, 0.0);
        $dbeta = array_fill(0, $inputSize, 0.0);
        
        // Compute gradients for gamma and beta
        foreach ($dout as $i => $doutSample) {
            for ($j = 0; $j < $inputSize; $j++) {
                $dgamma[$j] += $doutSample[$j] * $xNorm[$i][$j];
                $dbeta[$j] += $doutSample[$j];
            }
        }
        
        // Compute gradient with respect to input
        $dx = [];
        for ($i = 0; $i < $batchSize; $i++) {
            $dx[$i] = array_fill(0, $inputSize, 0.0);
        }
        
        for ($i = 0; $i < $batchSize; $i++) {
            for ($j = 0; $j < $inputSize; $j++) {
                $dx[$i][$j] = $dout[$i][$j] * $gamma[$j];
            }
        }
        
        // Backprop through the normalization
        $dxNorm = [];
        for ($i = 0; $i < $batchSize; $i++) {
            $dxNorm[$i] = array_fill(0, $inputSize, 0.0);
            for ($j = 0; $j < $inputSize; $j++) {
                $dxNorm[$i][$j] = $dx[$i][$j] / sqrt($batchVar[$j] + $epsilon);
            }
        }
        
        // Update parameters
        $this->updateParameters($dgamma, $dbeta);
        
        return $dxNorm;
    }

    /**
     * Update parameters with gradients
     */
    protected function updateParameters(array $dgamma, array $dbeta): void
    {
        $learningRate = 0.01; // Could be passed as an option
        
        for ($i = 0; $i < count($this->gamma); $i++) {
            $this->gamma[$i] -= $learningRate * $dgamma[$i];
            $this->beta[$i] -= $learningRate * $dbeta[$i];
        }
    }

    /**
     * Get parameters
     */
    public function getParameters(): array
    {
        return [
            'gamma' => $this->gamma,
            'beta' => $this->beta,
            'runningMean' => $this->runningMean,
            'runningVar' => $this->runningVar
        ];
    }

    /**
     * Set parameters
     */
    public function setParameters(array $parameters): void
    {
        if (isset($parameters['gamma'])) {
            $this->gamma = $parameters['gamma'];
        }
        if (isset($parameters['beta'])) {
            $this->beta = $parameters['beta'];
        }
        if (isset($parameters['runningMean'])) {
            $this->runningMean = $parameters['runningMean'];
        }
        if (isset($parameters['runningVar'])) {
            $this->runningVar = $parameters['runningVar'];
        }
    }
}
