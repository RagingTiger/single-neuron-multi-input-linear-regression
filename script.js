/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {TRAINING_DATA} from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/training.js';

// Input feature pairs (House size, Number of Bedrooms)
const INPUTS = TRAINING_DATA.inputs;

// Current listed house prices in dollars given their features above (target output values you want to predict).
const OUTPUTS = TRAINING_DATA.outputs;

// Ingest 1D input feature arrays as 2D tensors so that you can combine them later.
const INPUTS_TENSOR = tf.tensor2d(INPUTS);

// Output can stay 1 dimensional.
const OUTPUTS_TENSOR = tf.tensor1d(OUTPUTS);

// Function to take a Tensor and normalize values
// based on all values contained in that Tensor.
function normalize(tensor, min, max) {
  const result = tf.tidy(function() {
    // Find the minimum value contained in the Tensor.
    const MIN_VALUES = min || tf.min(tensor, 0);

    // Find the maximum value contained in the Tensor.
    const MAX_VALUES = max || tf.max(tensor, 0);

    // Now calculate subtract the MIN_VALUE from every value in the Tensor
    // And store the results in a new Tensor.
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);

    // Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);

    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
    // Return the adjusted values divided by the range size as a new Tensor.
    return {NORMALIZED_VALUES, MIN_VALUES, MAX_VALUES};
  });
  return result;
}


// Normalize all input feature arrays and then dispose of the original non normalized Tensors.
const FEATURE_RESULTS = normalize(INPUTS);
console.log('Normalized Values:');
FEATURE_RESULTS.NORMALIZED_VALUES.print();
console.log('Min Values:');
FEATURE_RESULTS.MIN_VALUES.print();
console.log('Max Values:');
FEATURE_RESULTS.MAX_VALUES.print();
INPUTS_TENSOR.dispose();


// Now actually create and define model architecture.
const model = tf.sequential();

// We will use one dense layer with 1 neuron (units) and an input of 
// 2 input feaature values (representing house size and number of rooms)
model.add(tf.layers.dense({inputShape: [2], units: 7}));
model.add(tf.layers.dense({units: 5}));
model.add(tf.layers.dense({units: 3}));
model.add(tf.layers.dense({units: 2}));
model.add(tf.layers.dense({units: 1}));

model.summary();

train();


async function train() {
  // Choose a learning rate that is suitable for the data we are using.
  const LEARNING_RATE = 0.0000001;
  
  // Compile the model with the defined learning rate and specify
  // our loss function to use.
  model.compile({
    optimizer: tf.train.sgd(LEARNING_RATE),
    loss: 'meanSquaredError'
  });

  // Finally do the training itself over 500 iterations of the data.
  // As we have so little training data we use batch size of 1.
  // We also set for the data to be shuffled each time we try 
  // and learn from it.
  let results = await model.fit(FEATURE_RESULTS.NORMALIZED_VALUES, OUTPUTS_TENSOR, {
    epochs: 100,
    validationSplit: 0.15, // TODO - define test/val/train split.
    batchSize: 20, 
    shuffle: true
  });
  
  OUTPUTS_TENSOR.dispose();
  FEATURE_RESULTS.NORMALIZED_VALUES.dispose();
  
  console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
  console.log("Average validation error loss: " + Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));
    
  // Once trained we can evaluate the model.
  evaluate();
}


function evaluate() {
  // Predict answer for a single piece of data.
  tf.tidy(function() {
    let newInput = normalize(tf.tensor2d([[750, 1]]), FEATURE_RESULTS.MIN_VALUES, FEATURE_RESULTS.MAX_VALUES);

    let output = model.predict(newInput.NORMALIZED_VALUES);
    output.print();
  });
  
  // Should show 7 Tensors left in memory incase you want to perform more predictions.
  // 4 Tensors store the min/max values for each of the 2 input features which you will need to normalize new inputs.
  // 3 Tensors make up the model itself that was trained.
  
  FEATURE_RESULTS.MIN_VALUES.dispose();
  FEATURE_RESULTS.MAX_VALUES.dispose();
  model.dispose();
  
  console.log(tf.memory().numTensors);
}
