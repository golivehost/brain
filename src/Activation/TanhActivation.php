<?php
/**
 * Tanh Activation Function
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

class TanhActivation implements ActivationFunction
{
    /**
     * Get the tanh function
     */
    public function getFunction(): callable
    {
        return function ($x) {
            return tanh($x);
        };
    }
    
    /**
     * Get the derivative of the tanh function
     */
    public function getDerivative(): callable
    {
        return function ($x) {
            $tanhX = tanh($x);
            return 1 - ($tanhX * $tanhX);
        };
    }
}
