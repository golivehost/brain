<?php
/**
 * RMSprop optimizer implementation
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Optimizers;

class RMSprop implements Optimizer
{
    protected float $learningRate;
    protected float $decay;
    protected float $epsilon;
    protected array $cache = []; // Running average of squared gradients

    /**
     * @param array $options Configuration options for RMSprop optimizer
     */
    public function __construct(array $options = [])
    {
        $this->learningRate = $options['learningRate'] ?? 0.01;
        $this->decay = $options['decay'] ?? 0.9;
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
     * Update parameters using RMSprop optimization
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
                $cache[$key] = $this->decay * $cache[$key] + (1 - $this->decay) * pow($grad, 2);
                
                $value -= $this->learningRate * $grad / (sqrt($cache[$key]) + $this->epsilon);
            }
        }
    }
}
