<?php
/**
 * Neural Network implementation for PHP
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\NeuralNetworks;

use GoLiveHost\Brain\Utilities\Matrix;
use GoLiveHost\Brain\Utilities\Tensor;
use GoLiveHost\Brain\Activation\ActivationFactory;
use GoLiveHost\Brain\Utilities\DataFormatter;
use GoLiveHost\Brain\Utilities\Normalizer;
use GoLiveHost\Brain\Exceptions\BrainException;

class NeuralNetwork
{
    protected array $options;
    protected array $sizes = [];
    protected array $layers = [];
    protected array $biases = [];
    protected array $weights = [];
    protected $activation;
    protected $activationDerivative;
    protected ?DataFormatter $dataFormatter = null;
    protected ?Normalizer $normalizer = null;
    protected array $errorLog = [];
    protected array $trainStats = [];
    protected bool $isInitialized = false;
    protected bool $isTraining = false; // Added this property initialization

    /**
     * @param array $options Configuration options for the neural network
     */
    public function __construct(array $options = [])
    {
        // Default options
        $this->options = array_merge([
            'inputSize' => 0,
            'hiddenLayers' => [10],
            'outputSize' => 0,
            'binaryThresh' => 0.5,
            'activation' => 'sigmoid',
            'leakyReluAlpha' => 0.01,
            'learningRate' => 0.3,
            'momentum' => 0.1,
            'iterations' => 20000,
            'errorThresh' => 0.005,
            'log' => false,
            'logPeriod' => 10,
            'dropout' => 0,
            'decayRate' => 0.999,
            'batchSize' => 10,
            'callbackPeriod' => 10,
            'timeout' => INF,
            'praxis' => 'adam',
            'beta1' => 0.9,
            'beta2' => 0.999,
            'epsilon' => 1e-8,
            'normalize' => true,
            'formatData' => true
        ], $options);

        $this->setActivation($this->options['activation']);
        
        if ($this->options['normalize']) {
            $this->normalizer = new Normalizer();
        }
        
        if ($this->options['formatData']) {
            $this->dataFormatter = new DataFormatter();
        }
    }

    /**
     * Set the activation function and its derivative
     */
    protected function setActivation(string $name): void
    {
        $activation = ActivationFactory::create($name, [
            'alpha' => $this->options['leakyReluAlpha']
        ]);
        
        $this->activation = $activation->getFunction();
        $this->activationDerivative = $activation->getDerivative();
    }

    /**
     * Initialize the neural network with random weights and biases
     */
    public function initialize(): void
    {
        $this->sizes = array_merge(
            [$this->options['inputSize']],
            $this->options['hiddenLayers'],
            [$this->options['outputSize']]
        );

        $this->layers = array_fill(0, count($this->sizes), []);
        $this->biases = [];
        $this->weights = [];

        // Initialize weights and biases using Xavier/Glorot initialization
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            $this->biases[$i] = [];
            $this->weights[$i] = [];
            
            $fanIn = $this->sizes[$i];
            $fanOut = $this->sizes[$i + 1];
            $stdDev = sqrt(2 / ($fanIn + $fanOut));
            
            for ($j = 0; $j < $this->sizes[$i + 1]; $j++) {
                $this->biases[$i][$j] = $this->gaussianRandom(0, $stdDev);
                
                $this->weights[$i][$j] = [];
                for ($k = 0; $k < $this->sizes[$i]; $k++) {
                    $this->weights[$i][$j][$k] = $this->gaussianRandom(0, $stdDev);
                }
            }
        }
        
        // Initialize optimizer-specific parameters
        if ($this->options['praxis'] === 'adam') {
            $this->initializeAdam();
        }
        
        $this->isInitialized = true;
    }
    
    /**
     * Initialize Adam optimizer parameters
     */
    protected function initializeAdam(): void
    {
        $this->m = []; // First moment vector
        $this->v = []; // Second moment vector
        
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            $this->m[$i] = [];
            $this->v[$i] = [];
            
            for ($j = 0; $j < $this->sizes[$i + 1]; $j++) {
                $this->m[$i][$j] = array_fill(0, $this->sizes[$i], 0);
                $this->v[$i][$j] = array_fill(0, $this->sizes[$i], 0);
            }
        }
        
        $this->mBias = [];
        $this->vBias = [];
        
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            $this->mBias[$i] = array_fill(0, $this->sizes[$i + 1], 0);
            $this->vBias[$i] = array_fill(0, $this->sizes[$i + 1], 0);
        }
    }

    /**
     * Generate a random number from a Gaussian distribution
     */
    protected function gaussianRandom(float $mean = 0, float $stdDev = 1): float
    {
        // Box-Muller transform
        $u1 = 1 - mt_rand() / mt_getrandmax();
        $u2 = 1 - mt_rand() / mt_getrandmax();
        $randStdNormal = sqrt(-2 * log($u1)) * sin(2 * M_PI * $u2);
        return $mean + $stdDev * $randStdNormal;
    }

    /**
     * Train the neural network with the provided data
     */
    public function train(array $data, array $trainingOptions = []): array
    {
        $options = array_merge($this->options, $trainingOptions);
        $startTime = microtime(true);
        
        // Set training flag
        $this->isTraining = true;
        
        // Format and normalize data if needed
        if ($this->dataFormatter !== null) {
            $data = $this->dataFormatter->format($data);
        }
        
        if ($this->normalizer !== null) {
            $this->normalizer->fit($data);
            $data = $this->normalizer->transform($data);
        }
        
        // Auto-detect input and output sizes if not provided
        if ($this->options['inputSize'] === 0) {
            $this->options['inputSize'] = count($data[0]['input']);
        }
        if ($this->options['outputSize'] === 0) {
            $this->options['outputSize'] = count($data[0]['output']);
        }
        
        if (!$this->isInitialized) {
            $this->initialize();
        }
        
        $error = 1;
        $iterations = 0;
        $this->errorLog = [];
        $t = 1; // Time step for Adam optimizer
        
        // For early stopping
        $bestError = INF;
        $bestWeights = null;
        $bestBiases = null;
        $patience = 10;
        $patienceCounter = 0;
        
        // For batch training
        $batchSize = min($options['batchSize'], count($data));
        
        while ($error > $options['errorThresh'] && $iterations < $options['iterations']) {
            // Check timeout
            if (microtime(true) - $startTime > $options['timeout']) {
                break;
            }
            
            // Shuffle data for each epoch
            shuffle($data);
            
            $totalError = 0;
            $batchCount = 0;
            
            // Process in batches
            for ($i = 0; $i < count($data); $i += $batchSize) {
                $batchData = array_slice($data, $i, $batchSize);
                $batchError = $this->trainBatch($batchData, $t);
                $totalError += $batchError;
                $batchCount++;
                $t++;
            }
            
            $error = $totalError / $batchCount;
            $this->errorLog[] = $error;
            $iterations++;
            
            // Learning rate decay
            $this->options['learningRate'] *= $options['decayRate'];
            
            // Logging
            if ($options['log'] && $iterations % $options['logPeriod'] === 0) {
                echo "Iteration: $iterations, Error: $error\n";
            }
            
            // Early stopping
            if ($error < $bestError) {
                $bestError = $error;
                $bestWeights = $this->deepCopy($this->weights);
                $bestBiases = $this->deepCopy($this->biases);
                $patienceCounter = 0;
            } else {
                $patienceCounter++;
                if ($patienceCounter >= $patience) {
                    // Restore best weights and biases
                    $this->weights = $bestWeights;
                    $this->biases = $bestBiases;
                    break;
                }
            }
            
            // Callback
            if (isset($options['callback']) && $iterations % $options['callbackPeriod'] === 0) {
                $options['callback']([
                    'iterations' => $iterations,
                    'error' => $error
                ]);
            }
        }
        
        $trainingTime = microtime(true) - $startTime;
        
        $this->trainStats = [
            'error' => $error,
            'iterations' => $iterations,
            'time' => $trainingTime,
            'errorLog' => $this->errorLog
        ];
        
        // Reset training flag
        $this->isTraining = false;
        
        return $this->trainStats;
    }
    
    /**
     * Train the neural network with a batch of data
     */
    protected function trainBatch(array $batch, int $t): float
    {
        $totalError = 0;
        $gradients = $this->initializeGradients();
        
        foreach ($batch as $item) {
            $output = $this->forward($item['input']);
            $backpropResult = $this->backward($item['output']);
            
            // Accumulate gradients
            $this->accumulateGradients($gradients, $backpropResult);
            
            // Calculate error
            $itemError = 0;
            for ($i = 0; $i < count($output); $i++) {
                $itemError += pow($output[$i] - $item['output'][$i], 2);
            }
            $totalError += $itemError / count($output);
        }
        
        // Apply gradients using the selected optimizer
        if ($this->options['praxis'] === 'adam') {
            $this->updateWithAdam($gradients, $t, count($batch));
        } else {
            $this->updateWithSGD($gradients, count($batch));
        }
        
        return $totalError / count($batch);
    }
    
    /**
     * Initialize gradient structures
     */
    protected function initializeGradients(): array
    {
        $gradients = [
            'weights' => [],
            'biases' => []
        ];
        
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            $gradients['biases'][$i] = array_fill(0, $this->sizes[$i + 1], 0);
            $gradients['weights'][$i] = [];
            
            for ($j = 0; $j < $this->sizes[$i + 1]; $j++) {
                $gradients['weights'][$i][$j] = array_fill(0, $this->sizes[$i], 0);
            }
        }
        
        return $gradients;
    }
    
    /**
     * Accumulate gradients from backpropagation
     */
    protected function accumulateGradients(array &$gradients, array $backpropResult): void
    {
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            for ($j = 0; $j < $this->sizes[$i + 1]; $j++) {
                $gradients['biases'][$i][$j] += $backpropResult['biasGradients'][$i][$j];
                
                for ($k = 0; $k < $this->sizes[$i]; $k++) {
                    $gradients['weights'][$i][$j][$k] += $backpropResult['weightGradients'][$i][$j][$k];
                }
            }
        }
    }
    
    /**
     * Update weights and biases using Stochastic Gradient Descent
     */
    protected function updateWithSGD(array $gradients, int $batchSize): void
    {
        $learningRate = $this->options['learningRate'];
        $momentum = $this->options['momentum'];
        
        // Initialize momentum if not already done
        if (!isset($this->velocities)) {
            $this->velocities = $this->initializeGradients();
        }
        
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            for ($j = 0; $j < $this->sizes[$i + 1]; $j++) {
                // Update bias with momentum
                $this->velocities['biases'][$i][$j] = 
                    $momentum * $this->velocities['biases'][$i][$j] - 
                    $learningRate * ($gradients['biases'][$i][$j] / $batchSize);
                
                $this->biases[$i][$j] += $this->velocities['biases'][$i][$j];
                
                for ($k = 0; $k < $this->sizes[$i]; $k++) {
                    // Update weight with momentum
                    $this->velocities['weights'][$i][$j][$k] = 
                        $momentum * $this->velocities['weights'][$i][$j][$k] - 
                        $learningRate * ($gradients['weights'][$i][$j][$k] / $batchSize);
                    
                    $this->weights[$i][$j][$k] += $this->velocities['weights'][$i][$j][$k];
                }
            }
        }
    }
    
    /**
     * Update weights and biases using Adam optimizer
     */
    protected function updateWithAdam(array $gradients, int $t, int $batchSize): void
    {
        $learningRate = $this->options['learningRate'];
        $beta1 = $this->options['beta1'];
        $beta2 = $this->options['beta2'];
        $epsilon = $this->options['epsilon'];
        
        // Bias correction
        $biasCorrection1 = 1 - pow($beta1, $t);
        $biasCorrection2 = 1 - pow($beta2, $t);
        $alphat = $learningRate * sqrt($biasCorrection2) / $biasCorrection1;
        
        for ($i = 0; $i < count($this->sizes) - 1; $i++) {
            for ($j = 0; $j < $this->sizes[$i + 1]; $j++) {
                // Update bias
                $gradient = $gradients['biases'][$i][$j] / $batchSize;
                $this->mBias[$i][$j] = $beta1 * $this->mBias[$i][$j] + (1 - $beta1) * $gradient;
                $this->vBias[$i][$j] = $beta2 * $this->vBias[$i][$j] + (1 - $beta2) * pow($gradient, 2);
                
                $this->biases[$i][$j] -= $alphat * $this->mBias[$i][$j] / (sqrt($this->vBias[$i][$j]) + $epsilon);
                
                for ($k = 0; $k < $this->sizes[$i]; $k++) {
                    // Update weight
                    $gradient = $gradients['weights'][$i][$j][$k] / $batchSize;
                    $this->m[$i][$j][$k] = $beta1 * $this->m[$i][$j][$k] + (1 - $beta1) * $gradient;
                    $this->v[$i][$j][$k] = $beta2 * $this->v[$i][$j][$k] + (1 - $beta2) * pow($gradient, 2);
                    
                    $this->weights[$i][$j][$k] -= $alphat * $this->m[$i][$j][$k] / (sqrt($this->v[$i][$j][$k]) + $epsilon);
                }
            }
        }
    }

    /**
     * Forward pass through the neural network
     */
    public function forward(array $input): array
    {
        $this->layers = [];
        $this->layers[0] = $input;
        
        // Apply dropout to input if specified
        if ($this->options['dropout'] > 0 && $this->isTraining) {
            $this->dropoutMasks = [];
            $this->applyDropout(0);
        }

        // Forward propagation through each layer
        for ($i = 0; $i < count($this->weights); $i++) {
            $this->layers[$i + 1] = [];
            
            for ($j = 0; $j < count($this->weights[$i]); $j++) {
                $sum = $this->biases[$i][$j];
                
                for ($k = 0; $k < count($this->layers[$i]); $k++) {
                    $sum += $this->layers[$i][$k] * $this->weights[$i][$j][$k];
                }
                
                $this->layers[$i + 1][$j] = ($this->activation)($sum);
            }
            
            // Apply dropout to hidden layers
            if ($this->options['dropout'] > 0 && $this->isTraining && $i < count($this->weights) - 1) {
                $this->applyDropout($i + 1);
            }
        }

        return end($this->layers);
    }
    
    /**
     * Apply dropout to a layer
     */
    protected function applyDropout(int $layerIndex): void
    {
        $dropoutRate = $this->options['dropout'];
        $this->dropoutMasks[$layerIndex] = [];
        
        for ($i = 0; $i < count($this->layers[$layerIndex]); $i++) {
            // Create dropout mask (1 = keep, 0 = drop)
            $this->dropoutMasks[$layerIndex][$i] = mt_rand() / mt_getrandmax() > $dropoutRate ? 1 : 0;
            
            // Apply mask and scale
            $this->layers[$layerIndex][$i] *= $this->dropoutMasks[$layerIndex][$i] / (1 - $dropoutRate);
        }
    }

    /**
     * Backward pass through the neural network
     */
    protected function backward(array $target): array
    {
        $outputLayer = count($this->layers) - 1;
        $deltas = [];
        $weightGradients = [];
        $biasGradients = [];
        
        // Initialize gradients
        for ($i = 0; $i < count($this->weights); $i++) {
            $biasGradients[$i] = array_fill(0, count($this->biases[$i]), 0);
            $weightGradients[$i] = [];
            
            for ($j = 0; $j < count($this->weights[$i]); $j++) {
                $weightGradients[$i][$j] = array_fill(0, count($this->weights[$i][$j]), 0);
            }
        }
        
        // Calculate deltas for output layer
        $deltas[$outputLayer] = [];
        for ($i = 0; $i < count($this->layers[$outputLayer]); $i++) {
            $output = $this->layers[$outputLayer][$i];
            $error = $target[$i] - $output;
            $delta = $error * ($this->activationDerivative)($output);
            $deltas[$outputLayer][$i] = $delta;
        }
        
        // Calculate deltas for hidden layers
        for ($l = $outputLayer - 1; $l > 0; $l--) {
            $deltas[$l] = [];
            for ($i = 0; $i < count($this->layers[$l]); $i++) {
                $error = 0;
                for ($j = 0; $j < count($deltas[$l + 1]); $j++) {
                    $error += $deltas[$l + 1][$j] * $this->weights[$l][$j][$i];
                }
                
                // Apply dropout mask if using dropout
                if ($this->options['dropout'] > 0 && isset($this->dropoutMasks[$l][$i])) {
                    $error *= $this->dropoutMasks[$l][$i] / (1 - $this->options['dropout']);
                }
                
                $delta = $error * ($this->activationDerivative)($this->layers[$l][$i]);
                $deltas[$l][$i] = $delta;
            }
        }
        
        // Calculate gradients
        for ($l = 0; $l < $outputLayer; $l++) {
            for ($i = 0; $i < count($this->weights[$l]); $i++) {
                $biasGradients[$l][$i] = $deltas[$l + 1][$i];
                
                for ($j = 0; $j < count($this->weights[$l][$i]); $j++) {
                    $weightGradients[$l][$i][$j] = $deltas[$l + 1][$i] * $this->layers[$l][$j];
                }
            }
        }
        
        return [
            'weightGradients' => $weightGradients,
            'biasGradients' => $biasGradients
        ];
    }

    /**
     * Run the neural network with the provided input
     */
    public function run(array $input): array
    {
        // Save training state and disable dropout for prediction
        $wasTraining = $this->isTraining;
        $this->isTraining = false;
        
        // Normalize input if normalizer is used
        if ($this->normalizer !== null) {
            $input = $this->normalizer->transformInput($input);
        }
        
        $output = $this->forward($input);
        
        // Denormalize output if normalizer is used
        if ($this->normalizer !== null) {
            $output = $this->normalizer->inverseTransformOutput($output);
        }
        
        // Restore training state
        $this->isTraining = $wasTraining;
        
        return $output;
    }
    
    /**
     * Test the neural network with a test dataset
     */
    public function test(array $testData): array
    {
        $totalError = 0;
        $correctPredictions = 0;
        $totalPredictions = count($testData);
        
        foreach ($testData as $item) {
            $output = $this->run($item['input']);
            
            // Calculate error
            $itemError = 0;
            for ($i = 0; $i < count($output); $i++) {
                $itemError += pow($output[$i] - $item['output'][$i], 2);
            }
            $totalError += $itemError / count($output);
            
            // For classification tasks, check if prediction is correct
            if ($this->isPredictionCorrect($output, $item['output'])) {
                $correctPredictions++;
            }
        }
        
        $accuracy = $correctPredictions / $totalPredictions;
        $mse = $totalError / $totalPredictions;
        
        return [
            'error' => $mse,
            'accuracy' => $accuracy,
            'totalPredictions' => $totalPredictions,
            'correctPredictions' => $correctPredictions
        ];
    }
    
    /**
     * Check if a prediction is correct (for classification tasks)
     */
    protected function isPredictionCorrect(array $prediction, array $target): bool
    {
        // For binary classification
        if (count($prediction) === 1) {
            $predictedClass = $prediction[0] >= $this->options['binaryThresh'] ? 1 : 0;
            $actualClass = $target[0] >= $this->options['binaryThresh'] ? 1 : 0;
            return $predictedClass === $actualClass;
        }
        
        // For multi-class classification
        $predictedClass = array_search(max($prediction), $prediction);
        $actualClass = array_search(max($target), $target);
        return $predictedClass === $actualClass;
    }

    /**
     * Convert the neural network to JSON
     */
    public function toJSON(): string
    {
        $data = [
            'type' => 'NeuralNetwork',
            'options' => $this->options,
            'sizes' => $this->sizes,
            'weights' => $this->weights,
            'biases' => $this->biases,
            'trainStats' => $this->trainStats
        ];
        
        if ($this->normalizer !== null) {
            $data['normalizer'] = $this->normalizer->toArray();
        }
        
        if ($this->dataFormatter !== null) {
            $data['dataFormatter'] = $this->dataFormatter->toArray();
        }
        
        return json_encode($data);
    }

    /**
     * Create a neural network from JSON
     */
    public static function fromJSON(string $json): self
    {
        $data = json_decode($json, true);
        
        if (!isset($data['type']) || $data['type'] !== 'NeuralNetwork') {
            throw new BrainException('Invalid JSON format for NeuralNetwork');
        }
        
        $network = new self($data['options']);
        $network->sizes = $data['sizes'];
        $network->weights = $data['weights'];
        $network->biases = $data['biases'];
        $network->trainStats = $data['trainStats'] ?? [];
        $network->isInitialized = true;
        
        if (isset($data['normalizer'])) {
            $network->normalizer = Normalizer::fromArray($data['normalizer']);
        }
        
        if (isset($data['dataFormatter'])) {
            $network->dataFormatter = DataFormatter::fromArray($data['dataFormatter']);
        }
        
        return $network;
    }
    
    /**
     * Create a deep copy of an array
     */
    protected function deepCopy(array $array): array
    {
        return json_decode(json_encode($array), true);
    }
    
    /**
     * Get training statistics
     */
    public function getTrainStats(): array
    {
        return $this->trainStats;
    }
    
    /**
     * Get error log
     */
    public function getErrorLog(): array
    {
        return $this->errorLog;
    }
    
    /**
     * Export the model to a standalone PHP file
     */
    public function exportToPhp(string $className = 'ExportedNeuralNetwork'): string
    {
        $json = $this->toJSON();
        $escapedJson = addslashes($json);
        
        $code = "<?php\n";
        $code .= "/**\n";
        $code .= " * Exported Neural Network\n";
        $code .= " * \n";
        $code .= " * Developed by: Go Live Web Solutions (golive.host)\n";
        $code .= " * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)\n";
        $code .= " */\n\n";
        $code .= "class $className {\n";
        $code .= "    private \$network;\n\n";
        $code .= "    public function __construct() {\n";
        $code .= "        \$json = '$escapedJson';\n";
        $code .= "        \$this->network = json_decode(\$json, true);\n";
        $code .= "    }\n\n";
        $code .= "    public function run(array \$input): array {\n";
        $code .= "        // Normalize input if needed\n";
        $code .= "        if (isset(\$this->network['normalizer'])) {\n";
        $code .= "            \$input = \$this->normalizeInput(\$input);\n";
        $code .= "        }\n\n";
        $code .= "        // Forward pass\n";
        $code .= "        \$layers = [];\n";
        $code .= "        \$layers[0] = \$input;\n\n";
        $code .= "        for (\$i = 0; \$i < count(\$this->network['weights']); \$i++) {\n";
        $code .= "            \$layers[\$i + 1] = [];\n";
        $code .= "            \n";
        $code .= "            for (\$j = 0; \$j < count(\$this->network['weights'][\$i]); \$j++) {\n";
        $code .= "                \$sum = \$this->network['biases'][\$i][\$j];\n";
        $code .= "                \n";
        $code .= "                for (\$k = 0; \$k < count(\$layers[\$i]); \$k++) {\n";
        $code .= "                    \$sum += \$layers[\$i][\$k] * \$this->network['weights'][\$i][\$j][\$k];\n";
        $code .= "                }\n";
        $code .= "                \n";
        $code .= "                \$layers[\$i + 1][\$j] = \$this->activate(\$sum);\n";
        $code .= "            }\n";
        $code .= "        }\n\n";
        $code .= "        \$output = end(\$layers);\n\n";
        $code .= "        // Denormalize output if needed\n";
        $code .= "        if (isset(\$this->network['normalizer'])) {\n";
        $code .= "            \$output = \$this->denormalizeOutput(\$output);\n";
        $code .= "        }\n\n";
        $code .= "        return \$output;\n";
        $code .= "    }\n\n";
        $code .= "    private function activate(float \$x): float {\n";
        $code .= "        \$activation = \$this->network['options']['activation'] ?? 'sigmoid';\n";
        $code .= "        \n";
        $code .= "        switch (\$activation) {\n";
        $code .= "            case 'sigmoid':\n";
        $code .= "                return 1 / (1 + exp(-\$x));\n";
        $code .= "            case 'relu':\n";
        $code .= "                return max(0, \$x);\n";
        $code .= "            case 'leaky-relu':\n";
        $code .= "                \$alpha = \$this->network['options']['leakyReluAlpha'] ?? 0.01;\n";
        $code .= "                return \$x > 0 ? \$x : \$alpha * \$x;\n";
        $code .= "            case 'tanh':\n";
        $code .= "                return tanh(\$x);\n";
        $code .= "            default:\n";
        $code .= "                return 1 / (1 + exp(-\$x)); // Default to sigmoid\n";
        $code .= "        }\n";
        $code .= "    }\n\n";
        $code .= "    private function normalizeInput(array \$input): array {\n";
        $code .= "        \$normalizer = \$this->network['normalizer'];\n";
        $code .= "        \$result = [];\n";
        $code .= "        \n";
        $code .= "        foreach (\$input as \$i => \$value) {\n";
        $code .= "            \$min = \$normalizer['inputRanges'][\$i]['min'] ?? 0;\n";
        $code .= "            \$max = \$normalizer['inputRanges'][\$i]['max'] ?? 1;\n";
        $code .= "            \$result[\$i] = (\$value - \$min) / (\$max - \$min);\n";
        $code .= "        }\n";
        $code .= "        \n";
        $code .= "        return \$result;\n";
        $code .= "    }\n\n";
        $code .= "    private function denormalizeOutput(array \$output): array {\n";
        $code .= "        \$normalizer = \$this->network['normalizer'];\n";
        $code .= "        \$result = [];\n";
        $code .= "        \n";
        $code .= "        foreach (\$output as \$i => \$value) {\n";
        $code .= "            \$min = \$normalizer['outputRanges'][\$i]['min'] ?? 0;\n";
        $code .= "            \$max = \$normalizer['outputRanges'][\$i]['max'] ?? 1;\n";
        $code .= "            \$result[\$i] = \$value * (\$max - \$min) + \$min;\n";
        $code .= "        }\n";
        $code .= "        \n";
        $code .= "        return \$result;\n";
        $code .= "    }\n";
        $code .= "}\n";
        
        return $code;
    }
}
