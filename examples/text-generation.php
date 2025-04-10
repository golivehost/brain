<?php
/**
 * Text generation example using the GoLiveHost Brain library
 * 
 * Developed by: Go Live Web Solutions (golive.host)
 * Author: Shubhdeep Singh (GitHub.com/shubhdeepdev)
 */

require_once __DIR__ . '/../vendor/autoload.php';

use GoLiveHost\Brain\Brain;
use GoLiveHost\Brain\Utilities\DataFormatter;

// This example demonstrates how to use LSTM for character-level text generation

// Sample text for training (a short poem)
$text = "
The Road Not Taken
by Robert Frost

Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could
To where it bent in the undergrowth;

Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,

And both that morning equally lay
In leaves no step had trodden black.
Oh, I kept the first for another day!
Yet knowing how way leads on to way,
I doubted if I should ever come back.

I shall be telling this with a sigh
Somewhere ages and ages hence:
Two roads diverged in a wood, and I—
I took the one less traveled by,
And that has made all the difference.
";

// Function to prepare training data for character-level text generation
function prepareTrainingData($text, $sequenceLength = 10) {
    $chars = str_split($text);
    $uniqueChars = array_unique($chars);
    sort($uniqueChars);
    
    $charToIndex = [];
    $indexToChar = [];
    
    foreach ($uniqueChars as $i => $char) {
        $charToIndex[$char] = $i;
        $indexToChar[$i] = $char;
    }
    
    $sequences = [];
    
    for ($i = 0; $i < count($chars) - $sequenceLength; $i++) {
        $inputSeq = array_slice($chars, $i, $sequenceLength);
        $outputChar = $chars[$i + $sequenceLength];
        
        $inputIndices = array_map(function($char) use ($charToIndex) {
            return $charToIndex[$char];
        }, $inputSeq);
        
        $outputIndex = $charToIndex[$outputChar];
        
        // One-hot encode input and output
        $input = [];
        $output = [];
        
        foreach ($inputIndices as $index) {
            $oneHot = array_fill(0, count($uniqueChars), 0);
            $oneHot[$index] = 1;
            $input[] = $oneHot;
        }
        
        $oneHot = array_fill(0, count($uniqueChars), 0);
        $oneHot[$outputIndex] = 1;
        $output[] = $oneHot;
        
        $sequences[] = [
            'input' => $input,
            'output' => $output
        ];
    }
    
    return [
        'sequences' => $sequences,
        'charToIndex' => $charToIndex,
        'indexToChar' => $indexToChar,
        'vocabSize' => count($uniqueChars)
    ];
}

// Function to generate text
function generateText($lstm, $seed, $length, $charToIndex, $indexToChar, $vocabSize) {
    $chars = str_split($seed);
    $result = $seed;
    
    // Convert seed to one-hot encoded sequences
    $sequence = [];
    foreach ($chars as $char) {
        $oneHot = array_fill(0, $vocabSize, 0);
        $oneHot[$charToIndex[$char]] = 1;
        $sequence[] = $oneHot;
    }
    
    // Generate characters
    for ($i = 0; $i < $length; $i++) {
        $prediction = $lstm->run($sequence);
        $lastPrediction = end($prediction);
        
        // Sample from the distribution (with temperature)
        $index = sampleFromDistribution($lastPrediction, 0.5);
        $nextChar = $indexToChar[$index];
        
        $result .= $nextChar;
        
        // Update sequence for next prediction
        array_shift($sequence);
        $oneHot = array_fill(0, $vocabSize, 0);
        $oneHot[$charToIndex[$nextChar]] = 1;
        $sequence[] = $oneHot;
    }
    
    return $result;
}

// Sample from a probability distribution with temperature
function sampleFromDistribution($distribution, $temperature = 1.0) {
    // Apply temperature
    $distribution = array_map(function($p) use ($temperature) {
        return pow($p, 1 / $temperature);
    }, $distribution);
    
    // Normalize
    $sum = array_sum($distribution);
    $distribution = array_map(function($p) use ($sum) {
        return $p / $sum;
    }, $distribution);
    
    // Sample
    $r = mt_rand() / mt_getrandmax();
    $cumulativeProb = 0;
    
    for ($i = 0; $i < count($distribution); $i++) {
        $cumulativeProb += $distribution[$i];
        if ($r <= $cumulativeProb) {
            return $i;
        }
    }
    
    return count($distribution) - 1; // Fallback
}

// Prepare training data
echo "Preparing training data...\n";
$sequenceLength = 25;
$data = prepareTrainingData($text, $sequenceLength);

echo "Vocabulary size: {$data['vocabSize']}\n";
echo "Number of sequences: " . count($data['sequences']) . "\n\n";

// Create LSTM
$lstm = Brain::lstm([
    'inputSize' => $data['vocabSize'],
    'hiddenLayers' => [128],
    'outputSize' => $data['vocabSize'],
    'activation' => 'tanh',
    'learningRate' => 0.01,
    'iterations' => 50,
    'batchSize' => 32,
    'log' => true,
    'logPeriod' => 5
]);

// Train LSTM
echo "Training LSTM...\n";
$result = $lstm->train($data['sequences']);
echo "Training completed with error {$result['error']}\n\n";

// Generate text
$seedText = "Two roads diverged in a";
echo "Generating text with seed: \"$seedText\"\n\n";

$generatedText = generateText(
    $lstm, 
    $seedText, 
    200, 
    $data['charToIndex'], 
    $data['indexToChar'], 
    $data['vocabSize']
);

echo $generatedText . "\n\n";

// Save the model
$json = $lstm->toJSON();
file_put_contents(__DIR__ . '/text-generation-model.json', $json);
echo "Model saved to text-generation-model.json\n";
