<?php
/**
 * ReLU Activation Function
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

class ReLUActivation implements ActivationFunction
{
    /**
     * Get the ReLU function
     */
    public function getFunction(): callable
    {
        return function ($x) {
            return max(0, $x);
        };
    }
    
    /**
     * Get the derivative of the ReLU function
     */
    public function getDerivative(): callable
    {
        return function ($x) {
            return $x > 0 ? 1 : 0;
        };
    }
}
