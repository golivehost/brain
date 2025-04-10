<?php
/**
 * Data Formatter for neural networks
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

class DataFormatter
{
    protected array $vocabulary = [];
    protected array $indexMap = [];
    protected bool $isInitialized = false;
    
    /**
     * Format data for neural network training
     */
    public function format(array $data): array
    {
        if (!$this->isInitialized) {
            $this->buildVocabulary($data);
        }
        
        $formattedData = [];
        
        foreach ($data as $item) {
            $formattedInput = $this->formatItem($item['input']);
            $formattedOutput = $this->formatItem($item['output']);
            
            $formattedData[] = [
                'input' => $formattedInput,
                'output' => $formattedOutput
            ];
        }
        
        return $formattedData;
    }
    
    /**
     * Format sequences for recurrent neural networks
     */
    public function formatSequences(array $sequences): array
    {
        if (!$this->isInitialized) {
            $this->buildVocabularyFromSequences($sequences);
        }
        
        $formattedSequences = [];
        
        foreach ($sequences as $sequence) {
            $formattedInputs = [];
            $formattedOutputs = [];
            
            foreach ($sequence['input'] as $input) {
                $formattedInputs[] = $this->formatItem($input);
            }
            
            foreach ($sequence['output'] as $output) {
                $formattedOutputs[] = $this->formatItem($output);
            }
            
            $formattedSequences[] = [
                'input' => $formattedInputs,
                'output' => $formattedOutputs
            ];
        }
        
        return $formattedSequences;
    }
    
    /**
     * Format a single input sequence
     */
    public function formatInputSequence(array $sequence): array
    {
        if (!$this->isInitialized) {
            throw new \RuntimeException('DataFormatter must be initialized before formatting');
        }
        
        $formattedSequence = [];
        
        foreach ($sequence as $item) {
            $formattedSequence[] = $this->formatItem($item);
        }
        
        return $formattedSequence;
    }
    
    /**
     * Format a single output sequence
     */
    public function formatOutputSequence(array $sequence): array
    {
        if (!$this->isInitialized) {
            throw new \RuntimeException('DataFormatter must be initialized before formatting');
        }
        
        $formattedSequence = [];
        
        foreach ($sequence as $item) {
            $formattedSequence[] = $this->formatItem($item);
        }
        
        return $formattedSequence;
    }
    
    /**
     * Build vocabulary from training data
     */
    protected function buildVocabulary(array $data): void
    {
        $this->vocabulary = [];
        $this->indexMap = [];
        
        foreach ($data as $item) {
            $this->addToVocabulary($item['input']);
            $this->addToVocabulary($item['output']);
        }
        
        $this->isInitialized = true;
    }
    
    /**
     * Build vocabulary from sequence data
     */
    protected function buildVocabularyFromSequences(array $sequences): void
    {
        $this->vocabulary = [];
        $this->indexMap = [];
        
        foreach ($sequences as $sequence) {
            foreach ($sequence['input'] as $input) {
                $this->addToVocabulary($input);
            }
            
            foreach ($sequence['output'] as $output) {
                $this->addToVocabulary($output);
            }
        }
        
        $this->isInitialized = true;
    }
    
    /**
     * Add items to vocabulary
     */
    protected function addToVocabulary(array $item): void
    {
        foreach ($item as $value) {
            if (is_string($value) && !isset($this->indexMap[$value])) {
                $index = count($this->vocabulary);
                $this->vocabulary[$index] = $value;
                $this->indexMap[$value] = $index;
            }
        }
    }
    
    /**
     * Format a single item
     */
    protected function formatItem(array $item): array
    {
        $formatted = [];
        
        foreach ($item as $value) {
            if (is_string($value) && isset($this->indexMap[$value])) {
                // One-hot encode string values
                $oneHot = array_fill(0, count($this->vocabulary), 0);
                $oneHot[$this->indexMap[$value]] = 1;
                $formatted = array_merge($formatted, $oneHot);
            } else {
                // Keep numeric values as is
                $formatted[] = $value;
            }
        }
        
        return $formatted;
    }
    
    /**
     * Convert to array for serialization
     */
    public function toArray(): array
    {
        return [
            'vocabulary' => $this->vocabulary,
            'indexMap' => $this->indexMap,
            'isInitialized' => $this->isInitialized
        ];
    }
    
    /**
     * Create from array after deserialization
     */
    public static function fromArray(array $data): self
    {
        $formatter = new self();
        $formatter->vocabulary = $data['vocabulary'];
        $formatter->indexMap = $data['indexMap'];
        $formatter->isInitialized = $data['isInitialized'];
        
        return $formatter;
    }
}
