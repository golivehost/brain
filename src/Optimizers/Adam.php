<?php
/**
 * Adam optimizer implementation
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Optimizers;

class Adam implements Optimizer
{
    protected float $learningRate;
    protected float $beta1;
    protected float $beta2;
    protected float $epsilon;
    protected array $m = []; // First moment vector
    protected array $v = []; // Second moment vector
    protected int $t = 0; // Timestep

    /**
     * @param array $options Configuration options for Adam optimizer
     */
    public function __construct(array $options = [])
    {
        $this->learningRate = $options['learningRate'] ?? 0.001;
        $this->beta1 = $options['beta1'] ?? 0.9;
        $this->beta2 = $options['beta2'] ?? 0.999;
        $this->epsilon = $options['epsilon'] ?? 1e-8;
    }

    /**
     * Initialize optimizer state for given parameters
     */
    public function initialize(array $parameters): void
    {
        $this->m = [];
        $this->v = [];
        $this->t = 0;

        foreach ($parameters as $key => $param) {
            if (is_array($param)) {
                $this->m[$key] = $this->initializeZeros($param);
                $this->v[$key] = $this->initializeZeros($param);
            }
        }
    }

    /**
     * Initialize a zero array with the same shape as the input
     */
    protected function initializeZeros(array $array): array
    {
        $result = [];
        
        foreach ($array as $key => $value) {
            if (is_array($value)) {
                $result[$key] = $this->initializeZeros($value);
            } else {
                $result[$key] = 0;
            }
        }
        
        return $result;
    }

    /**
     * Update parameters using Adam optimization
     */
    public function update(array &$parameters, array $gradients): void
    {
        $this->t++;
        
        // Bias correction
        $biasCorrection1 = 1 - pow($this->beta1, $this->t);
        $biasCorrection2 = 1 - pow($this->beta2, $this->t);
        $alphat = $this->learningRate * sqrt($biasCorrection2) / $biasCorrection1;
        
        foreach ($parameters as $key => &$param) {
            if (is_array($param)) {
                $this->updateRecursive($param, $gradients[$key], $this->m[$key], $this->v[$key], $alphat);
            }
        }
    }

    /**
     * Recursively update nested parameters
     */
    protected function updateRecursive(array &$param, array $gradient, array &$m, array &$v, float $alphat): void
    {
        foreach ($param as $key => &$value) {
            if (is_array($value)) {
                $this->updateRecursive($value, $gradient[$key], $m[$key], $v[$key], $alphat);
            } else {
                $grad = $gradient[$key];
                $m[$key] = $this->beta1 * $m[$key] + (1 - $this->beta1) * $grad;
                $v[$key] = $this->beta2 * $v[$key] + (1 - $this->beta2) * pow($grad, 2);
                
                $value -= $alphat * $m[$key] / (sqrt($v[$key]) + $this->epsilon);
            }
        }
    }
}
