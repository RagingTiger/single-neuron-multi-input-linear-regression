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

const status = document.getElementById('status');
if (status) {
  status.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
}

// An array of house sizes (first input feature).
const HOUSE_SIZES = [720, 863, 674, 600, 760, 982, 1513, 1073, 1185, 1222, 1060, 1575, 1440, 1787, 1551, 1653, 1575, 2522];

// An array of house bedrooms (second input feature).
const HOUSE_BEDROOMS = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3];

// Current listed house prices in dollars given their features above (target output values you want to predict).
const HOUSE_PRICES = [971000, 875000, 620000, 590000, 710000, 849000, 1995000, 1199000, 1380000, 1398888, 1650000, 1498000, 1782000, 1987888, 1688000, 1850000, 1498000, 5900000];

// Ingest 1D input feature arrays as 2D tensors so that you can combine them later.
const HOUSE_SIZES_TENSOR = tf.tensor2d(HOUSE_SIZES, [HOUSE_SIZES.length, 1]);
const HOUSE_BEDROOMS_TENSOR = tf.tensor2d(HOUSE_BEDROOMS, [HOUSE_BEDROOMS.length, 1]);

// Output can stay 1 dimensional.
const HOUSE_PRICES_TENSOR = tf.tensor1d(HOUSE_PRICES);


// Function to take a Tensor and normalize values
// based on all values contained in that Tensor.
function normalize(tensor, min, max) {
  const result = tf.tidy(function() {
    // Find the minimum value contained in the Tensor.
    const MIN_VALUE = min || tf.min(tensor);

    // Find the maximum value contained in the Tensor.
    const MAX_VALUE = max || tf.max(tensor);

    // Now calculate subtract the MIN_VALUE from every value in the Tensor
    // And store the results in a new Tensor.
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUE);

    // Calculate the range size of possible values.
    const RANGE_SIZE = tf.sub(MAX_VALUE, MIN_VALUE);

    // Return the adjusted values divided by the range size as a new Tensor.
    return tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
  });

  return result;
}


// Normalize all input feature arrays and then dispose of the original non normalized Tensors.
const HOUSE_SIZES_MIN = tf.min(HOUSE_SIZES_TENSOR);
const HOUSE_SIZES_MAX = tf.max(HOUSE_SIZES_TENSOR);
const HOUSE_SIZES_TENSOR_NORMALIZED = normalize(HOUSE_SIZES_TENSOR, HOUSE_SIZES_MIN, HOUSE_SIZES_MAX);
HOUSE_SIZES_TENSOR.dispose();

const HOUSE_BEDROOMS_MIN = tf.min(HOUSE_BEDROOMS_TENSOR);
const HOUSE_BEDROOMS_MAX = tf.max(HOUSE_BEDROOMS_TENSOR);
const HOUSE_BEDROOMS_TENSOR_NORMALIZED = normalize(HOUSE_BEDROOMS_TENSOR);
HOUSE_BEDROOMS_TENSOR.dispose();

// Print normalized Tensors to console to view contents.
console.log('Normalized House Sizes:');
HOUSE_SIZES_TENSOR_NORMALIZED.print();
console.log('Normalized Bedroom Sizes:');
HOUSE_BEDROOMS_TENSOR_NORMALIZED.print();

// Finally merge all the input feature 2D tensors using their 2nd axis (Axis 1 as zero indexed).
// This will combine the input features such that each item is an array of input features.
// For example:
// Feature 1 Tensor: [[1], [2]]
// Feature 2 Tensor: [[3], [4]]
// Returned Merged 2D Tensor: [[1,3], [2,4]]
const AXIS = 1;
const NORMALIZED_INPUT_FEATURES_COMBINED = tf.concat([HOUSE_SIZES_TENSOR_NORMALIZED, HOUSE_BEDROOMS_TENSOR_NORMALIZED], AXIS);
HOUSE_SIZES_TENSOR_NORMALIZED.dispose();
HOUSE_BEDROOMS_TENSOR_NORMALIZED.dispose();
NORMALIZED_INPUT_FEATURES_COMBINED.print();


// Now actually create and define model architecture.
const model = tf.sequential();

// We will use one dense layer with 1 neuron (units) and an input of 
// 2 input feaature values (representing house size and number of rooms)
model.add(tf.layers.dense({inputShape: [2], units: 1}));

// Choose a learning rate that is suitable for the data we are using.
const LEARNING_RATE = 0.001;

train();

async function train() {
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
  let results = await model.fit(NORMALIZED_INPUT_FEATURES_COMBINED, HOUSE_PRICES_TENSOR, {
    epochs: 100,
    validationSplit: 0.15, // TODO - define test/val/train split.
    batchSize: 1, 
    shuffle: true
  });
  
  NORMALIZED_INPUT_FEATURES_COMBINED.dispose();
  HOUSE_PRICES_TENSOR.dispose();
  
  model.summary();
  
  // Once trained we can evaluate the model.
  evaluate();
}

function evaluate() {
    // Predict answer for a single piece of data.
  const INPUTS = tf.tidy(function() {
    const NEW_SIZE = tf.tensor2d([[1000]]);
    const NEW_BEDROOMS = tf.tensor2d([[2]]);

    const NEW_SIZE_NORMALIZED = normalize(NEW_SIZE, HOUSE_SIZES_MIN, HOUSE_SIZES_MAX);
    const NEW_BEDROOMS_NORMALIZED = normalize(NEW_BEDROOMS, HOUSE_BEDROOMS_MIN, HOUSE_BEDROOMS_MAX);

    return NEW_HOUSE_INPUT_TENSOR = tf.concat([NEW_SIZE_NORMALIZED, NEW_BEDROOMS_NORMALIZED], 1);
  });

  let output = model.predict(INPUTS);
  INPUTS.dispose();
  output.print();
  output.dispose();
  
  
  // Should show 7 Tensors left in memory incase you want to perform more predictions.
  // 4 Tensors store the min/max values for each of the 2 input features which you will need to normalize new inputs.
  // 2 Tensors make up the model itself that was trained.
  console.log(tf.memory().numTensors);
}
