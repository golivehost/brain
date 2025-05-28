<?php
/**
 * Model checkpoint utility for saving and loading models during training
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

use GoLiveHost\Brain\Exceptions\BrainException;

class ModelCheckpoint
{
    protected string $directory;
    protected string $filePrefix;
    protected int $saveFrequency;
    protected bool $saveOnlyBest;
    protected string $monitorMetric;
    protected float $bestMetricValue;
    protected bool $isMaximizing;
    protected int $maxCheckpoints;
    protected array $checkpointFiles;

    /**
     * @param array $options Configuration options for model checkpointing
     */
    public function __construct(array $options = [])
    {
        $this->directory = $options['directory'] ?? './checkpoints';
        $this->filePrefix = $options['filePrefix'] ?? 'model';
        $this->saveFrequency = $options['saveFrequency'] ?? 10;
        $this->saveOnlyBest = $options['saveOnlyBest'] ?? false;
        $this->monitorMetric = $options['monitorMetric'] ?? 'error';
        $this->isMaximizing = $options['isMaximizing'] ?? false;
        $this->maxCheckpoints = $options['maxCheckpoints'] ?? 5;
        $this->bestMetricValue = $this->isMaximizing ? -INF : INF;
        $this->checkpointFiles = [];
        
        // Create directory if it doesn't exist
        if (!is_dir($this->directory)) {
            if (!mkdir($this->directory, 0755, true)) {
                throw new BrainException("Failed to create checkpoint directory: {$this->directory}");
            }
        }
    }

    /**
     * Save model checkpoint
     */
    public function save(object $model, array $metrics, int $epoch): ?string
    {
        // Check if we should save based on frequency
        if (!$this->saveOnlyBest && $epoch % $this->saveFrequency !== 0) {
            return null;
        }
        
        // Check if we should save based on metric improvement
        $currentMetric = $metrics[$this->monitorMetric] ?? null;
        if ($currentMetric === null) {
            throw new BrainException("Monitored metric '{$this->monitorMetric}' not found in metrics");
        }
        
        $shouldSave = false;
        
        if ($this->saveOnlyBest) {
            if (($this->isMaximizing && $currentMetric > $this->bestMetricValue) || 
                (!$this->isMaximizing && $currentMetric < $this->bestMetricValue)) {
                $this->bestMetricValue = $currentMetric;
                $shouldSave = true;
            }
        } else {
            $shouldSave = true;
        }
        
        if (!$shouldSave) {
            return null;
        }
        
        // Generate filename
        $timestamp = date('YmdHis');
        $metricStr = number_format($currentMetric, 4);
        $filename = "{$this->filePrefix}_epoch{$epoch}_{$this->monitorMetric}{$metricStr}_{$timestamp}.json";
        $filepath = "{$this->directory}/{$filename}";
        
        // Get model JSON
        $modelJson = $model->toJSON();
        
        // Verify JSON is valid
        $jsonData = json_decode($modelJson);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new BrainException("Failed to generate valid JSON for model: " . json_last_error_msg());
        }
        
        // Verify JSON is not empty
        if (empty($modelJson) || $modelJson === '{}' || $modelJson === '[]') {
            throw new BrainException("Generated JSON for model is empty");
        }
        
        // Save model with error checking
        $bytesWritten = file_put_contents($filepath, $modelJson);
        if ($bytesWritten === false) {
            throw new BrainException("Failed to write checkpoint to {$filepath}");
        }
        
        // Verify file was written successfully
        if ($bytesWritten === 0) {
            // Remove empty file
            if (file_exists($filepath)) {
                unlink($filepath);
            }
            throw new BrainException("Failed to write data to checkpoint file (0 bytes written)");
        }
        
        // Verify file exists and has content
        if (!file_exists($filepath) || filesize($filepath) === 0) {
            throw new BrainException("Checkpoint file is empty or does not exist: {$filepath}");
        }
        
        // Add to checkpoint files
        $this->checkpointFiles[] = $filepath;
        
        // Manage maximum number of checkpoints
        $this->manageCheckpoints();
        
        return $filepath;
    }

    /**
     * Manage the maximum number of checkpoints by removing oldest ones
     */
    protected function manageCheckpoints(): void
    {
        if (count($this->checkpointFiles) <= $this->maxCheckpoints) {
            return;
        }
        
        // Sort by creation time (oldest first)
        usort($this->checkpointFiles, function($a, $b) {
            return filemtime($a) - filemtime($b);
        });
        
        // Remove oldest files
        $filesToRemove = array_slice($this->checkpointFiles, 0, count($this->checkpointFiles) - $this->maxCheckpoints);
        foreach ($filesToRemove as $file) {
            if (file_exists($file)) {
                unlink($file);
            }
        }
        
        // Update checkpoint files list
        $this->checkpointFiles = array_slice($this->checkpointFiles, count($filesToRemove));
    }

    /**
     * Load the latest checkpoint
     */
    public function loadLatest(string $modelClass): ?object
    {
        $files = glob("{$this->directory}/{$this->filePrefix}_*.json");
        
        if (empty($files)) {
            return null;
        }
        
        // Sort by creation time (newest first)
        usort($files, function($a, $b) {
            return filemtime($b) - filemtime($a);
        });
        
        $latestFile = $files[0];
        
        return $this->load($latestFile, $modelClass);
    }

    /**
     * Load the best checkpoint
     */
    public function loadBest(string $modelClass): ?object
    {
        $files = glob("{$this->directory}/{$this->filePrefix}_*.json");
        
        if (empty($files)) {
            return null;
        }
        
        // Extract metric values from filenames
        $metricPattern = "/{$this->monitorMetric}([\d\.]+)_/";
        $filesWithMetrics = [];
        
        foreach ($files as $file) {
            if (preg_match($metricPattern, $file, $matches)) {
                $filesWithMetrics[$file] = (float) $matches[1];
            }
        }
        
        if (empty($filesWithMetrics)) {
            return null;
        }
        
        // Sort by metric value
        if ($this->isMaximizing) {
            arsort($filesWithMetrics);
        } else {
            asort($filesWithMetrics);
        }
        
        $bestFile = array_key_first($filesWithMetrics);
        
        return $this->load($bestFile, $modelClass);
    }

    /**
     * Load a specific checkpoint file
     */
    public function load(string $filepath, string $modelClass): ?object
    {
        if (!file_exists($filepath)) {
            throw new BrainException("Checkpoint file not found: {$filepath}");
        }
        
        // Check file size
        if (filesize($filepath) === 0) {
            throw new BrainException("Checkpoint file is empty: {$filepath}");
        }
        
        $json = file_get_contents($filepath);
        
        if ($json === false) {
            throw new BrainException("Failed to read checkpoint file: {$filepath}");
        }
        
        // Validate JSON format
        $data = json_decode($json, true);
        if (json_last_error() !== JSON_ERROR_NONE) {
            throw new BrainException("Invalid JSON format in checkpoint file: " . json_last_error_msg());
        }
        
        if (empty($data)) {
            throw new BrainException("Checkpoint file contains empty JSON data");
        }
        
        if (!isset($data['type'])) {
            throw new BrainException("Invalid JSON format: missing 'type' field in checkpoint file");
        }
        
        if (!method_exists($modelClass, 'fromJSON')) {
            throw new BrainException("Model class {$modelClass} does not implement fromJSON method");
        }
        
        return $modelClass::fromJSON($json);
    }

    /**
     * Get list of available checkpoints
     */
    public function getCheckpoints(): array
    {
        $files = glob("{$this->directory}/{$this->filePrefix}_*.json");
        
        $checkpoints = [];
        
        foreach ($files as $file) {
            $info = [
                'filepath' => $file,
                'filename' => basename($file),
                'created' => filemtime($file),
                'size' => filesize($file)
            ];
            
            // Extract epoch and metric from filename
            if (preg_match('/epoch(\d+)_([a-zA-Z]+)([\d\.]+)_/', $file, $matches)) {
                $info['epoch'] = (int) $matches[1];
                $info['metric'] = $matches[2];
                $info['value'] = (float) $matches[3];
            }
            
            $checkpoints[] = $info;
        }
        
        // Sort by creation time (newest first)
        usort($checkpoints, function($a, $b) {
            return $b['created'] - $a['created'];
        });
        
        return $checkpoints;
    }

    /**
     * Delete all checkpoints
     */
    public function clearCheckpoints(): void
    {
        $files = glob("{$this->directory}/{$this->filePrefix}_*.json");
        
        foreach ($files as $file) {
            if (file_exists($file)) {
                unlink($file);
            }
        }
        
        $this->checkpointFiles = [];
    }
}
