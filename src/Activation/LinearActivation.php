<?php
/**
 * Linear Activation Function
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

class LinearActivation implements ActivationFunction
{
    /**
     * Get the linear function
     */
    public function getFunction(): callable
    {
        return function ($x) {
            return $x;
        };
    }
    
    /**
     * Get the derivative of the linear function
     */
    public function getDerivative(): callable
    {
        return function ($x) {
            return 1;
        };
    }
}
