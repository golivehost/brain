<?php
/**
 * Unit tests for ModelValidator
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Tests;

use PHPUnit\Framework\TestCase;
use GoLiveHost\Brain\Utilities\ModelValidator;
use GoLiveHost\Brain\Exceptions\BrainException;

class ModelValidatorTest extends TestCase
{
    public function testValidateNeuralNetworkOptions()
    {
        // Valid options
        $options = [
            'inputSize' => 10,
            'hiddenLayers' => [20, 15],
            'outputSize' => 5,
            'activation' => 'sigmoid',
            'learningRate' => 0.01,
            'momentum' => 0.9,
            'iterations' => 1000,
            'errorThresh' => 0.001,
            'dropout' => 0.2,
            'decayRate' => 0.999,
            'batchSize' => 32,
            'praxis' => 'adam',
            'beta1' => 0.9,
            'beta2' => 0.999,
            'epsilon' => 1e-8
        ];
        
        $validatedOptions = ModelValidator::validateNeuralNetworkOptions($options);
        $this->assertEquals($options, $validatedOptions);
        
        // Test invalid input size
        $invalidOptions = $options;
        $invalidOptions['inputSize'] = -1;
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($invalidOptions);
    }
    
    public function testValidateHiddenLayers()
    {
        // Invalid hidden layers (not an array)
        $options = [
            'hiddenLayers' => 10
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($options);
        
        // Invalid hidden layer size (negative)
        $options = [
            'hiddenLayers' => [10, -5]
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($options);
    }
    
    public function testValidateActivation()
    {
        // Invalid activation function
        $options = [
            'activation' => 'invalid_activation'
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($options);
    }
    
    public function testValidateLearningParameters()
    {
        // Invalid learning rate
        $options = [
            'learningRate' => -0.1
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($options);
        
        // Invalid momentum
        $options = [
            'momentum' => 1.5
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($options);
        
        // Invalid iterations
        $options = [
            'iterations' => 0
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateNeuralNetworkOptions($options);
    }
    
    public function testValidateTrainingData()
    {
        // Valid data
        $data = [
            ['input' => [1, 2, 3], 'output' => [4, 5]],
            ['input' => [6, 7, 8], 'output' => [9, 10]]
        ];
        
        $this->assertTrue(ModelValidator::validateTrainingData($data));
        
        // Invalid data (missing output)
        $invalidData = [
            ['input' => [1, 2, 3]],
            ['input' => [6, 7, 8], 'output' => [9, 10]]
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateTrainingData($invalidData);
        
        // Invalid data (input not an array)
        $invalidData = [
            ['input' => 123, 'output' => [4, 5]],
            ['input' => [6, 7, 8], 'output' => [9, 10]]
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateTrainingData($invalidData);
    }
    
    public function testValidateSequenceData()
    {
        // Valid sequence data
        $data = [
            [
                'input' => [[1, 2], [3, 4], [5, 6]],
                'output' => [[7, 8], [9, 10]]
            ],
            [
                'input' => [[11, 12], [13, 14]],
                'output' => [[15, 16]]
            ]
        ];
        
        $this->assertTrue(ModelValidator::validateTrainingData($data, true));
        
        // Invalid sequence data (empty input sequence)
        $invalidData = [
            [
                'input' => [],
                'output' => [[7, 8], [9, 10]]
            ]
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateTrainingData($invalidData, true);
        
        // Invalid sequence data (input item not an array)
        $invalidData = [
            [
                'input' => [1, [3, 4], [5, 6]],
                'output' => [[7, 8], [9, 10]]
            ]
        ];
        
        $this->expectException(BrainException::class);
        ModelValidator::validateTrainingData($invalidData, true);
    }
    
    public function testValidateModelJSON()
    {
        // Valid JSON
        $json = json_encode([
            'type' => 'NeuralNetwork',
            'options' => ['inputSize' => 10, 'outputSize' => 5],
            'weights' => [],
            'biases' => []
        ]);
        
        $data = ModelValidator::validateModelJSON($json, 'NeuralNetwork');
        $this->assertEquals('NeuralNetwork', $data['type']);
        
        // Invalid JSON
        $invalidJson = '{invalid: json}';
        
        $this->expectException(BrainException::class);
        ModelValidator::validateModelJSON($invalidJson, 'NeuralNetwork');
        
        // Missing type field
        $invalidJson = json_encode([
            'options' => ['inputSize' => 10, 'outputSize' => 5],
            'weights' => [],
            'biases' => []
        ]);
        
        $this->expectException(BrainException::class);
        ModelValidator::validateModelJSON($invalidJson, 'NeuralNetwork');
        
        // Wrong type
        $invalidJson = json_encode([
            'type' => 'LSTM',
            'options' => ['inputSize' => 10, 'outputSize' => 5],
            'weights' => [],
            'biases' => []
        ]);
        
        $this->expectException(BrainException::class);
        ModelValidator::validateModelJSON($invalidJson, 'NeuralNetwork');
    }
}
