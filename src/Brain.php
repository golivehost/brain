<?php
/**
 * Main Brain class - Factory for creating neural networks
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain;

use GoLiveHost\Brain\NeuralNetworks\NeuralNetwork;
use GoLiveHost\Brain\NeuralNetworks\LSTM;
use GoLiveHost\Brain\NeuralNetworks\LiquidStateMachine;
use GoLiveHost\Brain\Exceptions\BrainException;

class Brain
{
    /**
     * Create a neural network
     */
    public static function neuralNetwork(array $options = []): NeuralNetwork
    {
        return new NeuralNetwork($options);
    }
    
    /**
     * Create an LSTM network
     */
    public static function lstm(array $options = []): LSTM
    {
        return new LSTM($options);
    }
    
    /**
     * Create a Liquid State Machine
     */
    public static function liquidStateMachine(array $options = []): LiquidStateMachine
    {
        return new LiquidStateMachine($options);
    }
    
    /**
     * Load a model from JSON
     */
    public static function fromJSON(string $json)
    {
        $data = json_decode($json, true);
        
        if (!isset($data['type'])) {
            throw new BrainException('Invalid JSON format: missing type');
        }
        
        switch ($data['type']) {
            case 'NeuralNetwork':
                return NeuralNetwork::fromJSON($json);
            case 'LSTM':
                return LSTM::fromJSON($json);
            case 'LiquidStateMachine':
                return LiquidStateMachine::fromJSON($json);
            default:
                throw new BrainException("Unknown model type: {$data['type']}");
        }
    }
}
