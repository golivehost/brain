<?php
/**
 * Activation Function interface
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

interface ActivationFunction
{
    /**
     * Get the activation function
     */
    public function getFunction(): callable;
    
    /**
     * Get the derivative of the activation function
     */
    public function getDerivative(): callable;
}
