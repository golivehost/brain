<?php
/**
 * Cross-validation utility for model evaluation
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

class CrossValidation
{
    /**
     * Perform k-fold cross-validation
     * 
     * @param object $model The model to train and evaluate
     * @param array $data The dataset to use for cross-validation
     * @param int $k Number of folds
     * @param array $options Additional options for training
     * @return array Results of cross-validation
     */
    public static function kFold(object $model, array $data, int $k = 5, array $options = []): array
    {
        // Shuffle data
        shuffle($data);
        
        // Split data into k folds
        $folds = array_chunk($data, ceil(count($data) / $k));
        
        $results = [];
        
        for ($i = 0; $i < $k; $i++) {
            // Create training and validation sets
            $validationSet = $folds[$i];
            $trainingSet = [];
            
            for ($j = 0; $j < $k; $j++) {
                if ($j !== $i) {
                    $trainingSet = array_merge($trainingSet, $folds[$j]);
                }
            }
            
            // Clone model to avoid state sharing between folds
            $foldModel = clone $model;
            
            // Train model
            $trainStats = $foldModel->train($trainingSet, $options);
            
            // Evaluate model
            $evalStats = self::evaluateModel($foldModel, $validationSet);
            
            // Store results
            $results[] = [
                'fold' => $i + 1,
                'trainStats' => $trainStats,
                'evalStats' => $evalStats
            ];
        }
        
        // Calculate average metrics
        $avgMetrics = self::calculateAverageMetrics($results);
        
        return [
            'folds' => $results,
            'averageMetrics' => $avgMetrics
        ];
    }

    /**
     * Perform stratified k-fold cross-validation (for classification tasks)
     * 
     * @param object $model The model to train and evaluate
     * @param array $data The dataset to use for cross-validation
     * @param int $k Number of folds
     * @param callable $labelExtractor Function to extract class label from data item
     * @param array $options Additional options for training
     * @return array Results of cross-validation
     */
    public static function stratifiedKFold(object $model, array $data, int $k = 5, callable $labelExtractor, array $options = []): array
    {
        // Group data by class
        $classBuckets = [];
        
        foreach ($data as $item) {
            $label = $labelExtractor($item);
            if (!isset($classBuckets[$label])) {
                $classBuckets[$label] = [];
            }
            $classBuckets[$label][] = $item;
        }
        
        // Shuffle each class bucket
        foreach ($classBuckets as &$bucket) {
            shuffle($bucket);
        }
        
        // Create stratified folds
        $folds = array_fill(0, $k, []);
        
        foreach ($classBuckets as $label => $items) {
            $itemsPerFold = ceil(count($items) / $k);
            $foldItems = array_chunk($items, $itemsPerFold);
            
            // Distribute items to folds
            for ($i = 0; $i < count($foldItems); $i++) {
                if ($i < $k) {
                    $folds[$i] = array_merge($folds[$i], $foldItems[$i]);
                } else {
                    // If we have more chunks than folds, distribute the remaining items
                    $folds[$i % $k] = array_merge($folds[$i % $k], $foldItems[$i]);
                }
            }
        }
        
        $results = [];
        
        for ($i = 0; $i < $k; $i++) {
            // Create training and validation sets
            $validationSet = $folds[$i];
            $trainingSet = [];
            
            for ($j = 0; $j < $k; $j++) {
                if ($j !== $i) {
                    $trainingSet = array_merge($trainingSet, $folds[$j]);
                }
            }
            
            // Clone model to avoid state sharing between folds
            $foldModel = clone $model;
            
            // Train model
            $trainStats = $foldModel->train($trainingSet, $options);
            
            // Evaluate model
            $evalStats = self::evaluateModel($foldModel, $validationSet);
            
            // Store results
            $results[] = [
                'fold' => $i + 1,
                'trainStats' => $trainStats,
                'evalStats' => $evalStats
            ];
        }
        
        // Calculate average metrics
        $avgMetrics = self::calculateAverageMetrics($results);
        
        return [
            'folds' => $results,
            'averageMetrics' => $avgMetrics
        ];
    }

    /**
     * Perform leave-one-out cross-validation
     * 
     * @param object $model The model to train and evaluate
     * @param array $data The dataset to use for cross-validation
     * @param array $options Additional options for training
     * @return array Results of cross-validation
     */
    public static function leaveOneOut(object $model, array $data, array $options = []): array
    {
        $k = count($data);
        
        // For large datasets, this can be very computationally expensive
        if ($k > 100) {
            trigger_error("Leave-one-out cross-validation with {$k} samples may be computationally expensive", E_USER_WARNING);
        }
        
        return self::kFold($model, $data, $k, $options);
    }

    /**
     * Perform train-test split
     * 
     * @param array $data The dataset to split
     * @param float $testSize Proportion of data to use for testing (0.0-1.0)
     * @param bool $shuffle Whether to shuffle the data before splitting
     * @return array Associative array with 'train' and 'test' keys
     */
    public static function trainTestSplit(array $data, float $testSize = 0.2, bool $shuffle = true): array
    {
        if ($shuffle) {
            shuffle($data);
        }
        
        $testCount = (int) ceil(count($data) * $testSize);
        $trainCount = count($data) - $testCount;
        
        $trainData = array_slice($data, 0, $trainCount);
        $testData = array_slice($data, $trainCount);
        
        return [
            'train' => $trainData,
            'test' => $testData
        ];
    }

    /**
     * Evaluate a model on a dataset
     * 
     * @param object $model The model to evaluate
     * @param array $data The dataset to evaluate on
     * @return array Evaluation metrics
     */
    public static function evaluateModel(object $model, array $data): array
    {
        $predictions = [];
        $targets = [];
        $errors = [];
        
        foreach ($data as $item) {
            $output = $model->run($item['input']);
            $predictions[] = $output;
            $targets[] = $item['output'];
            
            // Calculate error
            $itemError = 0;
            for ($i = 0; $i < count($output); $i++) {
                $itemError += pow($output[$i] - $item['output'][$i], 2);
            }
            $errors[] = $itemError / count($output);
        }
        
        // Calculate metrics
        $mse = array_sum($errors) / count($errors);
        $rmse = sqrt($mse);
        
        // Calculate R-squared (coefficient of determination)
        $r2 = self::calculateR2($targets, $predictions);
        
        // For classification tasks
        $accuracy = self::calculateAccuracy($targets, $predictions);
        $precision = self::calculatePrecision($targets, $predictions);
        $recall = self::calculateRecall($targets, $predictions);
        $f1 = self::calculateF1Score($precision, $recall);
        
        return [
            'mse' => $mse,
            'rmse' => $rmse,
            'r2' => $r2,
            'accuracy' => $accuracy,
            'precision' => $precision,
            'recall' => $recall,
            'f1' => $f1
        ];
    }

    /**
     * Calculate average metrics across all folds
     * 
     * @param array $results Results from cross-validation
     * @return array Average metrics
     */
    protected static function calculateAverageMetrics(array $results): array
    {
        $metrics = [
            'trainError' => 0,
            'trainIterations' => 0,
            'trainTime' => 0,
            'mse' => 0,
            'rmse' => 0,
            'r2' => 0,
            'accuracy' => 0,
            'precision' => 0,
            'recall' => 0,
            'f1' => 0
        ];
        
        $count = count($results);
        
        foreach ($results as $result) {
            $metrics['trainError'] += $result['trainStats']['error'];
            $metrics['trainIterations'] += $result['trainStats']['iterations'];
            $metrics['trainTime'] += $result['trainStats']['time'];
            
            $metrics['mse'] += $result['evalStats']['mse'];
            $metrics['rmse'] += $result['evalStats']['rmse'];
            $metrics['r2'] += $result['evalStats']['r2'];
            $metrics['accuracy'] += $result['evalStats']['accuracy'];
            $metrics['precision'] += $result['evalStats']['precision'];
            $metrics['recall'] += $result['evalStats']['recall'];
            $metrics['f1'] += $result['evalStats']['f1'];
        }
        
        // Calculate averages
        foreach ($metrics as &$value) {
            $value /= $count;
        }
        
        return $metrics;
    }

    /**
     * Calculate R-squared (coefficient of determination)
     * 
     * @param array $targets Actual values
     * @param array $predictions Predicted values
     * @return float R-squared value
     */
    protected static function calculateR2(array $targets, array $predictions): float
    {
        // Flatten arrays if they are multi-dimensional
        $y = [];
        $yPred = [];
        
        foreach ($targets as $i => $target) {
            foreach ($target as $j => $value) {
                $y[] = $value;
                $yPred[] = $predictions[$i][$j];
            }
        }
        
        $mean = array_sum($y) / count($y);
        
        $ssTotal = 0;
        $ssResidual = 0;
        
        for ($i = 0; $i < count($y); $i++) {
            $ssTotal += pow($y[$i] - $mean, 2);
            $ssResidual += pow($y[$i] - $yPred[$i], 2);
        }
        
        if ($ssTotal === 0) {
            return 0; // Avoid division by zero
        }
        
        return 1 - ($ssResidual / $ssTotal);
    }

    /**
     * Calculate classification accuracy
     * 
     * @param array $targets Actual values
     * @param array $predictions Predicted values
     * @return float Accuracy
     */
    protected static function calculateAccuracy(array $targets, array $predictions): float
    {
        $correct = 0;
        $total = count($targets);
        
        for ($i = 0; $i < $total; $i++) {
            $targetClass = self::getPredictedClass($targets[$i]);
            $predictedClass = self::getPredictedClass($predictions[$i]);
            
            if ($targetClass === $predictedClass) {
                $correct++;
            }
        }
        
        return $correct / $total;
    }

    /**
     * Calculate precision
     * 
     * @param array $targets Actual values
     * @param array $predictions Predicted values
     * @return float Precision
     */
    protected static function calculatePrecision(array $targets, array $predictions): float
    {
        $truePositives = 0;
        $falsePositives = 0;
        
        for ($i = 0; $i < count($targets); $i++) {
            $targetClass = self::getPredictedClass($targets[$i]);
            $predictedClass = self::getPredictedClass($predictions[$i]);
            
            if ($predictedClass === 1) {
                if ($targetClass === 1) {
                    $truePositives++;
                } else {
                    $falsePositives++;
                }
            }
        }
        
        if ($truePositives + $falsePositives === 0) {
            return 0;
        }
        
        return $truePositives / ($truePositives + $falsePositives);
    }

    /**
     * Calculate recall
     * 
     * @param array $targets Actual values
     * @param array $predictions Predicted values
     * @return float Recall
     */
    protected static function calculateRecall(array $targets, array $predictions): float
    {
        $truePositives = 0;
        $falseNegatives = 0;
        
        for ($i = 0; $i < count($targets); $i++) {
            $targetClass = self::getPredictedClass($targets[$i]);
            $predictedClass = self::getPredictedClass($predictions[$i]);
            
            if ($targetClass === 1) {
                if ($predictedClass === 1) {
                    $truePositives++;
                } else {
                    $falseNegatives++;
                }
            }
        }
        
        if ($truePositives + $falseNegatives === 0) {
            return 0;
        }
        
        return $truePositives / ($truePositives + $falseNegatives);
    }

    /**
     * Calculate F1 score
     * 
     * @param float $precision Precision value
     * @param float $recall Recall value
     * @return float F1 score
     */
    protected static function calculateF1Score(float $precision, float $recall): float
    {
        // If either precision or recall is zero, F1 score is zero
        if ($precision <= 0 || $recall <= 0) {
            return 0;
        }
        
        // Avoid potential floating-point issues by checking if the sum is very close to zero
        if (abs($precision + $recall) < 1e-10) {
            return 0;
        }
        
        return 2 * ($precision * $recall) / ($precision + $recall);
    }

    /**
     * Get predicted class from output array
     * 
     * @param array $output Model output
     * @return int Predicted class (0 or 1)
     */
    protected static function getPredictedClass(array $output): int
    {
        // For binary classification
        if (count($output) === 1) {
            return $output[0] >= 0.5 ? 1 : 0;
        }
        
        // For multi-class classification
        $maxIndex = 0;
        $maxValue = $output[0];
        
        for ($i = 1; $i < count($output); $i++) {
            if ($output[$i] > $maxValue) {
                $maxValue = $output[$i];
                $maxIndex = $i;
            }
        }
        
        return $maxIndex;
    }
}
