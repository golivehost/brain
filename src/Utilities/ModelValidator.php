<?php
/**
 * Model validation utility
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

use GoLiveHost\Brain\Exceptions\BrainException;

class ModelValidator
{
    /**
     * Validate neural network options
     * 
     * @param array $options Options to validate
     * @return array Validated options
     * @throws BrainException If options are invalid
     */
    public static function validateNeuralNetworkOptions(array $options): array
    {
        // Required options
        if (isset($options['inputSize']) && (!is_int($options['inputSize']) || $options['inputSize'] < 0)) {
            throw new BrainException("Input size must be a non-negative integer");
        }
        
        if (isset($options['outputSize']) && (!is_int($options['outputSize']) || $options['outputSize'] < 0)) {
            throw new BrainException("Output size must be a non-negative integer");
        }
        
        // Hidden layers
        if (isset($options['hiddenLayers'])) {
            if (!is_array($options['hiddenLayers'])) {
                throw new BrainException("Hidden layers must be an array");
            }
            
            foreach ($options['hiddenLayers'] as $size) {
                if (!is_int($size) || $size <= 0) {
                    throw new BrainException("Hidden layer size must be a positive integer");
                }
            }
        }
        
        // Activation function
        if (isset($options['activation'])) {
            $validActivations = ['sigmoid', 'tanh', 'relu', 'leaky-relu', 'linear', 'softmax'];
            if (!in_array($options['activation'], $validActivations)) {
                throw new BrainException("Invalid activation function: {$options['activation']}");
            }
        }
        
        // Learning parameters
        if (isset($options['learningRate']) && (!is_numeric($options['learningRate']) || $options['learningRate'] <= 0)) {
            throw new BrainException("Learning rate must be a positive number");
        }
        
        if (isset($options['momentum']) && (!is_numeric($options['momentum']) || $options['momentum'] < 0 || $options['momentum'] > 1)) {
            throw new BrainException("Momentum must be between 0 and 1");
        }
        
        if (isset($options['iterations']) && (!is_int($options['iterations']) || $options['iterations'] <= 0)) {
            throw new BrainException("Iterations must be a positive integer");
        }
        
        if (isset($options['errorThresh']) && (!is_numeric($options['errorThresh']) || $options['errorThresh'] <= 0)) {
            throw new BrainException("Error threshold must be a positive number");
        }
        
        // Regularization
        if (isset($options['dropout']) && (!is_numeric($options['dropout']) || $options['dropout'] < 0 || $options['dropout'] >= 1)) {
            throw new BrainException("Dropout must be between 0 and 1");
        }
        
        if (isset($options['decayRate']) && (!is_numeric($options['decayRate']) || $options['decayRate'] <= 0 || $options['decayRate'] > 1)) {
            throw new BrainException("Decay rate must be between 0 and 1");
        }
        
        // Batch size
        if (isset($options['batchSize']) && (!is_int($options['batchSize']) || $options['batchSize'] <= 0)) {
            throw new BrainException("Batch size must be a positive integer");
        }
        
        // Optimizer
        if (isset($options['praxis'])) {
            $validOptimizers = ['sgd', 'adam', 'rmsprop', 'adagrad'];
            if (!in_array($options['praxis'], $validOptimizers)) {
                throw new BrainException("Invalid optimizer: {$options['praxis']}");
            }
        }
        
        // Adam optimizer parameters
        if (isset($options['beta1']) && (!is_numeric($options['beta1']) || $options['beta1'] < 0 || $options['beta1'] >= 1)) {
            throw new BrainException("Beta1 must be between 0 and 1");
        }
        
        if (isset($options['beta2']) && (!is_numeric($options['beta2']) || $options['beta2'] < 0 || $options['beta2'] >= 1)) {
            throw new BrainException("Beta2 must be between 0 and 1");
        }
        
        if (isset($options['epsilon']) && (!is_numeric($options['epsilon']) || $options['epsilon'] <= 0)) {
            throw new BrainException("Epsilon must be a positive number");
        }
        
        return $options;
    }

    /**
     * Validate LSTM options
     * 
     * @param array $options Options to validate
     * @return array Validated options
     * @throws BrainException If options are invalid
     */
    public static function validateLSTMOptions(array $options): array
    {
        // First validate common neural network options
        $options = self::validateNeuralNetworkOptions($options);
        
        // LSTM-specific options
        if (isset($options['clipGradient']) && (!is_numeric($options['clipGradient']) || $options['clipGradient'] <= 0)) {
            throw new BrainException("Gradient clipping value must be a positive number");
        }
        
        return $options;
    }

    /**
     * Validate training data format
     * 
     * @param array $data Data to validate
     * @param bool $isSequence Whether the data is sequential
     * @return bool True if data is valid
     * @throws BrainException If data is invalid
     */
    public static function validateTrainingData(array $data, bool $isSequence = false): bool
    {
        if (empty($data)) {
            throw new BrainException("Training data cannot be empty");
        }
        
        foreach ($data as $i => $item) {
            if (!isset($item['input']) || !isset($item['output'])) {
                throw new BrainException("Training data item at index {$i} must have 'input' and 'output' keys");
            }
            
            if (!is_array($item['input'])) {
                throw new BrainException("Input at index {$i} must be an array");
            }
            
            if (!is_array($item['output'])) {
                throw new BrainException("Output at index {$i} must be an array");
            }
            
            if ($isSequence) {
                if (empty($item['input'])) {
                    throw new BrainException("Input sequence at index {$i} cannot be empty");
                }
                
                if (empty($item['output'])) {
                    throw new BrainException("Output sequence at index {$i} cannot be empty");
                }
                
                // Validate sequence items
                foreach ($item['input'] as $j => $inputItem) {
                    if (!is_array($inputItem)) {
                        throw new BrainException("Input sequence item at index {$i},{$j} must be an array");
                    }
                }
                
                foreach ($item['output'] as $j => $outputItem) {
                    if (!is_array($outputItem)) {
                        throw new BrainException("Output sequence item at index {$i},{$j} must be an array");
                    }
                }
            }
        }
        
        return true;
    }

    /**
     * Validate model JSON
     * 
     * @param string $json JSON string to validate
     * @param string $expectedType Expected model type
     * @return array Decoded JSON data
     * @throws BrainException If JSON is invalid
     */
    public static function validateModelJSON(string $json, string $expectedType): array
    {
        $data = json_decode($json, true);
        
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new BrainException("Invalid JSON: " . json_last_error_msg());
        }
        
        if (!isset($data['type'])) {
            throw new BrainException("Missing 'type' field in model JSON");
        }
        
        if ($data['type'] !== $expectedType) {
            throw new BrainException("Invalid model type: expected '{$expectedType}', got '{$data['type']}'");
        }
        
        return $data;
    }
}
