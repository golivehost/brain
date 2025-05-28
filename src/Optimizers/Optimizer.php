<?php
/**
 * Optimizer interface
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Optimizers;

interface Optimizer
{
    /**
     * Initialize optimizer state for given parameters
     */
    public function initialize(array $parameters): void;
    
    /**
     * Update parameters using gradients
     */
    public function update(array &$parameters, array $gradients): void;
}
