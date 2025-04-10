<?php
/**
 * Tensor operations utility
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

class Tensor
{
    /**
     * Apply a function to each element of a tensor
     */
    public static function map(array $tensor, callable $func): array
    {
        $result = [];
        
        foreach ($tensor as $key => $value) {
            if (is_array($value)) {
                $result[$key] = self::map($value, $func);
            } else {
                $result[$key] = $func($value);
            }
        }
        
        return $result;
    }
    
    /**
     * Element-wise addition of two tensors
     */
    public static function add(array $a, array $b): array
    {
        $result = [];
        
        foreach ($a as $key => $value) {
            if (is_array($value) && isset($b[$key]) && is_array($b[$key])) {
                $result[$key] = self::add($value, $b[$key]);
            } elseif (isset($b[$key])) {
                $result[$key] = $value + $b[$key];
            } else {
                $result[$key] = $value;
            }
        }
        
        return $result;
    }
    
    /**
     * Element-wise subtraction of two tensors
     */
    public static function subtract(array $a, array $b): array
    {
        $result = [];
        
        foreach ($a as $key => $value) {
            if (is_array($value) && isset($b[$key]) && is_array($b[$key])) {
                $result[$key] = self::subtract($value, $b[$key]);
            } elseif (isset($b[$key])) {
                $result[$key] = $value - $b[$key];
            } else {
                $result[$key] = $value;
            }
        }
        
        return $result;
    }
    
    /**
     * Element-wise multiplication of two tensors
     */
    public static function multiply(array $a, array $b): array
    {
        $result = [];
        
        foreach ($a as $key => $value) {
            if (is_array($value) && isset($b[$key]) && is_array($b[$key])) {
                $result[$key] = self::multiply($value, $b[$key]);
            } elseif (isset($b[$key])) {
                $result[$key] = $value * $b[$key];
            } else {
                $result[$key] = 0;
            }
        }
        
        return $result;
    }
    
    /**
     * Scalar multiplication
     */
    public static function scalarMultiply(array $tensor, float $scalar): array
    {
        return self::map($tensor, function ($x) use ($scalar) {
            return $x * $scalar;
        });
    }
    
    /**
     * Calculate the sum of all elements in a tensor
     */
    public static function sum(array $tensor): float
    {
        $sum = 0;
        
        foreach ($tensor as $value) {
            if (is_array($value)) {
                $sum += self::sum($value);
            } else {
                $sum += $value;
            }
        }
        
        return $sum;
    }
    
    /**
     * Calculate the mean of all elements in a tensor
     */
    public static function mean(array $tensor): float
    {
        $sum = self::sum($tensor);
        $count = self::count($tensor);
        
        return $count > 0 ? $sum / $count : 0;
    }
    
    /**
     * Count the number of elements in a tensor
     */
    public static function count(array $tensor): int
    {
        $count = 0;
        
        foreach ($tensor as $value) {
            if (is_array($value)) {
                $count += self::count($value);
            } else {
                $count++;
            }
        }
        
        return $count;
    }
    
    /**
     * Calculate the maximum value in a tensor
     */
    public static function max(array $tensor)
    {
        $max = null;
        
        foreach ($tensor as $value) {
            if (is_array($value)) {
                $subMax = self::max($value);
                if ($max === null || $subMax > $max) {
                    $max = $subMax;
                }
            } else {
                if ($max === null || $value > $max) {
                    $max = $value;
                }
            }
        }
        
        return $max;
    }
    
    /**
     * Calculate the minimum value in a tensor
     */
    public static function min(array $tensor)
    {
        $min = null;
        
        foreach ($tensor as $value) {
            if (is_array($value)) {
                $subMin = self::min($value);
                if ($min === null || $subMin < $min) {
                    $min = $subMin;
                }
            } else {
                if ($min === null || $value < $min) {
                    $min = $value;
                }
            }
        }
        
        return $min;
    }
    
    /**
     * Reshape a tensor
     */
    public static function reshape(array $tensor, array $shape): array
    {
        // Flatten the tensor
        $flat = self::flatten($tensor);
        
        // Check if the shapes are compatible
        $totalElements = array_product($shape);
        if (count($flat) !== $totalElements) {
            throw new \InvalidArgumentException("Cannot reshape tensor of size " . count($flat) . " into shape " . implode('x', $shape));
        }
        
        // Reshape
        return self::reshapeFlat($flat, $shape);
    }
    
    /**
     * Flatten a tensor into a 1D array
     */
    public static function flatten(array $tensor): array
    {
        $result = [];
        
        foreach ($tensor as $value) {
            if (is_array($value)) {
                $result = array_merge($result, self::flatten($value));
            } else {
                $result[] = $value;
            }
        }
        
        return $result;
    }
    
    /**
     * Reshape a flat array into the specified shape
     */
    protected static function reshapeFlat(array $flat, array $shape): array
    {
        if (empty($shape)) {
            return $flat[0];
        }
        
        $result = [];
        $currentDim = array_shift($shape);
        $subSize = empty($shape) ? 1 : array_product($shape);
        
        for ($i = 0; $i < $currentDim; $i++) {
            $subArray = array_slice($flat, $i * $subSize, $subSize);
            
            if (empty($shape)) {
                $result[$i] = $subArray[0];
            } else {
                $result[$i] = self::reshapeFlat($subArray, $shape);
            }
        }
        
        return $result;
    }
}
