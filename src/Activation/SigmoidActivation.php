<?php
/**
 * Sigmoid Activation Function
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

class SigmoidActivation implements ActivationFunction
{
    /**
     * Get the sigmoid function
     */
    public function getFunction(): callable
    {
        return function ($x) {
            return 1 / (1 + exp(-$x));
        };
    }
    
    /**
     * Get the derivative of the sigmoid function
     */
    public function getDerivative(): callable
    {
        return function ($x) {
            $sigmoid = 1 / (1 + exp(-$x));
            return $sigmoid * (1 - $sigmoid);
        };
    }
}
