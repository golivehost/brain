<?php
/**
 * Activation Function Factory
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Activation;

use GoLiveHost\Brain\Exceptions\BrainException;

class ActivationFactory
{
    /**
     * Create an activation function
     */
    public static function create(string $name, array $options = []): ActivationFunction
    {
        switch (strtolower($name)) {
            case 'sigmoid':
                return new SigmoidActivation();
            case 'tanh':
                return new TanhActivation();
            case 'relu':
                return new ReLUActivation();
            case 'leaky-relu':
                $alpha = $options['alpha'] ?? 0.01;
                return new LeakyReLUActivation($alpha);
            case 'softmax':
                return new SoftmaxActivation();
            case 'linear':
                return new LinearActivation();
            default:
                throw new BrainException("Unknown activation function: $name");
        }
    }
}
