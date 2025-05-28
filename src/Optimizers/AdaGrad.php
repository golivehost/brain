<?php
/**
 * AdaGrad optimizer implementation
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Optimizers;

class AdaGrad implements Optimizer
{
    protected float $learningRate;
    protected float $epsilon;
    protected array $cache = []; // Sum of squared gradients

    /**
     * @param array $options Configuration options for AdaGrad optimizer
     */
    public function __construct(array $options = [])
    {
        $this->learningRate = $options['learningRate'] ?? 0.01;
        $this->epsilon = $options['epsilon'] ?? 1e-8;
    }

    /**
     * Initialize optimizer state for given parameters
     */
    public function initialize(array $parameters): void
    {
        $this->cache = [];

        foreach ($parameters as $key => $param) {
            if (is_array($param)) {
                $this->cache[$key] = $this->initializeZeros($param);
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
     * Update parameters using AdaGrad optimization
     */
    public function update(array &$parameters, array $gradients): void
    {
        foreach ($parameters as $key => &$param) {
            if (is_array($param)) {
                $this->updateRecursive($param, $gradients[$key], $this->cache[$key]);
            }
        }
    }

    /**
     * Recursively update nested parameters
     */
    protected function updateRecursive(array &$param, array $gradient, array &$cache): void
    {
        foreach ($param as $key => &$value) {
            if (is_array($value)) {
                $this->updateRecursive($value, $gradient[$key], $cache[$key]);
            } else {
                $grad = $gradient[$key];
                $cache[$key] += pow($grad, 2);
                
                $value -= $this->learningRate * $grad / (sqrt($cache[$key]) + $this->epsilon);
            }
        }
    }
}
