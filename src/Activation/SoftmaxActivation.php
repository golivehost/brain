<?php
/**
 * Softmax Activation Function
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

class SoftmaxActivation implements ActivationFunction
{
    /**
     * Get the softmax function
     */
    public function getFunction(): callable
    {
        return function (array $x) {
            $max = max($x);
            $exp = array_map(function ($val) use ($max) {
                return exp($val - $max);
            }, $x);
            $sum = array_sum($exp);
            
            return array_map(function ($val) use ($sum) {
                return $val / $sum;
            }, $exp);
        };
    }
    
    /**
     * Get the derivative of the softmax function
     * Note: This is a simplified version that assumes cross-entropy loss
     */
    public function getDerivative(): callable
    {
        return function (array $x, array $y) {
            $softmax = $this->getFunction()($x);
            $result = [];
            
            for ($i = 0; $i < count($softmax); $i++) {
                $result[$i] = $softmax[$i] - $y[$i];
            }
            
            return $result;
        };
    }
}
