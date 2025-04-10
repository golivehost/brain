<?php
/**
 * Data Normalizer for neural networks
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

class Normalizer
{
    protected array $inputRanges = [];
    protected array $outputRanges = [];
    protected bool $isInitialized = false;
    
    /**
     * Fit the normalizer to the data
     */
    public function fit(array $data): void
    {
        $this->inputRanges = [];
        $this->outputRanges = [];
        
        // Find min and max for each input and output dimension
        foreach ($data as $item) {
            $this->updateRanges($this->inputRanges, $item['input']);
            $this->updateRanges($this->outputRanges, $item['output']);
        }
        
        $this->isInitialized = true;
    }
    
    /**
     * Update min and max ranges for each dimension
     */
    protected function updateRanges(array &$ranges, array $values): void
    {
        foreach ($values as $i => $value) {
            if (!isset($ranges[$i])) {
                $ranges[$i] = ['min' => $value, 'max' => $value];
            } else {
                $ranges[$i]['min'] = min($ranges[$i]['min'], $value);
                $ranges[$i]['max'] = max($ranges[$i]['max'], $value);
            }
        }
    }
    
    /**
     * Transform data to normalized range [0, 1]
     */
    public function transform(array $data): array
    {
        if (!$this->isInitialized) {
            throw new \RuntimeException('Normalizer must be fitted before transforming data');
        }
        
        $normalizedData = [];
        
        foreach ($data as $item) {
            $normalizedInput = $this->normalizeValues($item['input'], $this->inputRanges);
            $normalizedOutput = $this->normalizeValues($item['output'], $this->outputRanges);
            
            $normalizedData[] = [
                'input' => $normalizedInput,
                'output' => $normalizedOutput
            ];
        }
        
        return $normalizedData;
    }
    
    /**
     * Transform a single input
     */
    public function transformInput(array $input): array
    {
        if (!$this->isInitialized) {
            throw new \RuntimeException('Normalizer must be fitted before transforming data');
        }
        
        return $this->normalizeValues($input, $this->inputRanges);
    }
    
    /**
     * Inverse transform a single output
     */
    public function inverseTransformOutput(array $output): array
    {
        if (!$this->isInitialized) {
            throw new \RuntimeException('Normalizer must be fitted before inverse transforming data');
        }
        
        return $this->denormalizeValues($output, $this->outputRanges);
    }
    
    /**
     * Normalize values to range [0, 1]
     */
    protected function normalizeValues(array $values, array $ranges): array
    {
        $normalized = [];
        
        foreach ($values as $i => $value) {
            if (isset($ranges[$i])) {
                $min = $ranges[$i]['min'];
                $max = $ranges[$i]['max'];
                
                if ($max > $min) {
                    $normalized[$i] = ($value - $min) / ($max - $min);
                } else {
                    $normalized[$i] = 0.5; // Default if min equals max
                }
            } else {
                $normalized[$i] = $value; // Keep as is if no range info
            }
        }
        
        return $normalized;
    }
    
    /**
     * Denormalize values from range [0, 1]
     */
    protected function denormalizeValues(array $values, array $ranges): array
    {
        $denormalized = [];
        
        foreach ($values as $i => $value) {
            if (isset($ranges[$i])) {
                $min = $ranges[$i]['min'];
                $max = $ranges[$i]['max'];
                
                $denormalized[$i] = $value * ($max - $min) + $min;
            } else {
                $denormalized[$i] = $value; // Keep as is if no range info
            }
        }
        
        return $denormalized;
    }
    
    /**
     * Convert to array for serialization
     */
    public function toArray(): array
    {
        return [
            'inputRanges' => $this->inputRanges,
            'outputRanges' => $this->outputRanges,
            'isInitialized' => $this->isInitialized
        ];
    }
    
    /**
     * Create from array after deserialization
     */
    public static function fromArray(array $data): self
    {
        $normalizer = new self();
        $normalizer->inputRanges = $data['inputRanges'];
        $normalizer->outputRanges = $data['outputRanges'];
        $normalizer->isInitialized = $data['isInitialized'];
        
        return $normalizer;
    }
}
