<?php
/**
 * Leaky ReLU Activation Function
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

class LeakyReLUActivation implements ActivationFunction
{
    private float $alpha;
    
    /**
     * @param float $alpha Slope for negative values
     */
    public function __construct(float $alpha = 0.01)
    {
        $this->alpha = $alpha;
    }
    
    /**
     * Get the Leaky ReLU function
     */
    public function getFunction(): callable
    {
        return function ($x) {
            return $x > 0 ? $x : $this->alpha * $x;
        };
    }
    
    /**
     * Get the derivative of the Leaky ReLU function
     */
    public function getDerivative(): callable
    {
        return function ($x) {
            return $x > 0 ? 1 : $this->alpha;
        };
    }
}
