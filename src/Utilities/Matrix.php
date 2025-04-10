<?php
/**
 * Matrix operations utility
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

namespace GoLiveHost\Brain\Utilities;

class Matrix
{
    /**
     * Multiply two matrices
     */
    public static function multiply(array $a, array $b): array
    {
        $aRows = count($a);
        $aCols = count($a[0]);
        $bRows = count($b);
        $bCols = count($b[0]);
        
        if ($aCols !== $bRows) {
            throw new \InvalidArgumentException("Matrix dimensions do not match for multiplication");
        }
        
        $result = [];
        
        for ($i = 0; $i < $aRows; $i++) {
            $result[$i] = [];
            
            for ($j = 0; $j < $bCols; $j++) {
                $sum = 0;
                
                for ($k = 0; $k < $aCols; $k++) {
                    $sum += $a[$i][$k] * $b[$k][$j];
                }
                
                $result[$i][$j] = $sum;
            }
        }
        
        return $result;
    }
    
    /**
     * Add two matrices
     */
    public static function add(array $a, array $b): array
    {
        $aRows = count($a);
        $aCols = count($a[0]);
        $bRows = count($b);
        $bCols = count($b[0]);
        
        if ($aRows !== $bRows || $aCols !== $bCols) {
            throw new \InvalidArgumentException("Matrix dimensions do not match for addition");
        }
        
        $result = [];
        
        for ($i = 0; $i < $aRows; $i++) {
            $result[$i] = [];
            
            for ($j = 0; $j < $aCols; $j++) {
                $result[$i][$j] = $a[$i][$j] + $b[$i][$j];
            }
        }
        
        return $result;
    }
    
    /**
     * Subtract matrix b from matrix a
     */
    public static function subtract(array $a, array $b): array
    {
        $aRows = count($a);
        $aCols = count($a[0]);
        $bRows = count($b);
        $bCols = count($b[0]);
        
        if ($aRows !== $bRows || $aCols !== $bCols) {
            throw new \InvalidArgumentException("Matrix dimensions do not match for subtraction");
        }
        
        $result = [];
        
        for ($i = 0; $i < $aRows; $i++) {
            $result[$i] = [];
            
            for ($j = 0; $j < $aCols; $j++) {
                $result[$i][$j] = $a[$i][$j] - $b[$i][$j];
            }
        }
        
        return $result;
    }
    
    /**
     * Transpose a matrix
     */
    public static function transpose(array $matrix): array
    {
        $rows = count($matrix);
        $cols = count($matrix[0]);
        
        $result = [];
        
        for ($j = 0; $j < $cols; $j++) {
            $result[$j] = [];
            
            for ($i = 0; $i < $rows; $i++) {
                $result[$j][$i] = $matrix[$i][$j];
            }
        }
        
        return $result;
    }
    
    /**
     * Element-wise multiplication (Hadamard product)
     */
    public static function elementMultiply(array $a, array $b): array
    {
        $aRows = count($a);
        $aCols = count($a[0]);
        $bRows = count($b);
        $bCols = count($b[0]);
        
        if ($aRows !== $bRows || $aCols !== $bCols) {
            throw new \InvalidArgumentException("Matrix dimensions do not match for element-wise multiplication");
        }
        
        $result = [];
        
        for ($i = 0; $i < $aRows; $i++) {
            $result[$i] = [];
            
            for ($j = 0; $j < $aCols; $j++) {
                $result[$i][$j] = $a[$i][$j] * $b[$i][$j];
            }
        }
        
        return $result;
    }
    
    /**
     * Scalar multiplication
     */
    public static function scalarMultiply(array $matrix, float $scalar): array
    {
        $rows = count($matrix);
        $cols = count($matrix[0]);
        
        $result = [];
        
        for ($i = 0; $i < $rows; $i++) {
            $result[$i] = [];
            
            for ($j = 0; $j < $cols; $j++) {
                $result[$i][$j] = $matrix[$i][$j] * $scalar;
            }
        }
        
        return $result;
    }
    
    /**
     * Create a zero matrix
     */
    public static function zeros(int $rows, int $cols): array
    {
        $matrix = [];
        
        for ($i = 0; $i < $rows; $i++) {
            $matrix[$i] = array_fill(0, $cols, 0);
        }
        
        return $matrix;
    }
    
    /**
     * Create an identity matrix
     */
    public static function identity(int $size): array
    {
        $matrix = self::zeros($size, $size);
        
        for ($i = 0; $i < $size; $i++) {
            $matrix[$i][$i] = 1;
        }
        
        return $matrix;
    }
    
    /**
     * Calculate the determinant of a matrix
     */
    public static function determinant(array $matrix): float
    {
        $n = count($matrix);
        
        if ($n !== count($matrix[0])) {
            throw new \InvalidArgumentException("Matrix must be square to calculate determinant");
        }
        
        if ($n === 1) {
            return $matrix[0][0];
        }
        
        if ($n === 2) {
            return $matrix[0][0] * $matrix[1][1] - $matrix[0][1] * $matrix[1][0];
        }
        
        $det = 0;
        
        for ($j = 0; $j < $n; $j++) {
            $submatrix = [];
            
            for ($i = 1; $i < $n; $i++) {
                $row = [];
                
                for ($k = 0; $k < $n; $k++) {
                    if ($k !== $j) {
                        $row[] = $matrix[$i][$k];
                    }
                }
                
                $submatrix[] = $row;
            }
            
            $det += pow(-1, $j) * $matrix[0][$j] * self::determinant($submatrix);
        }
        
        return $det;
    }
    
    /**
     * Calculate the inverse of a matrix
     */
    public static function inverse(array $matrix): array
    {
        $n = count($matrix);
        
        if ($n !== count($matrix[0])) {
            throw new \InvalidArgumentException("Matrix must be square to calculate inverse");
        }
        
        $det = self::determinant($matrix);
        
        if (abs($det) < 1e-10) {
            throw new \InvalidArgumentException("Matrix is singular, cannot calculate inverse");
        }
        
        if ($n === 1) {
            return [[1 / $matrix[0][0]]];
        }
        
        if ($n === 2) {
            $result = [
                [$matrix[1][1], -$matrix[0][1]],
                [-$matrix[1][0], $matrix[0][0]]
            ];
            
            return self::scalarMultiply($result, 1 / $det);
        }
        
        // For larger matrices, use the adjugate method
        $cofactors = [];
        
        for ($i = 0; $i < $n; $i++) {
            $cofactors[$i] = [];
            
            for ($j = 0; $j < $n; $j++) {
                $submatrix = [];
                
                for ($k = 0; $k < $n; $k++) {
                    if ($k !== $i) {
                        $row = [];
                        
                        for ($l = 0; $l < $n; $l++) {
                            if ($l !== $j) {
                                $row[] = $matrix[$k][$l];
                            }
                        }
                        
                        $submatrix[] = $row;
                    }
                }
                
                $cofactors[$i][$j] = pow(-1, $i + $j) * self::determinant($submatrix);
            }
        }
        
        $adjugate = self::transpose($cofactors);
        
        return self::scalarMultiply($adjugate, 1 / $det);
    }
}
