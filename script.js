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

// An array of input data
const HOUSE_FEATURES = [
  // Size, Bedrooms
  [720, 1],
  [863, 1],
  [674, 1],
  [600, 1],
  [760, 1],
  [982, 1],
  [1513, 2],
  [1073, 2],
  [1185, 2],
  [1222, 2],
  [1060, 2],
  [1575, 2],
  [1440, 3],
  [1787, 3],
  [1551, 3],
  [1653, 3],
  [1575, 3],
  [2522, 3]
];


// Current listed house prices in dollars given their features above.
const HOUSE_PRICES = [971000, 875000, 620000, 590000, 710000, 849000, 1995000, 1199000, 1380000, 1398888, 1650000, 1498000, 1782000, 1987888, 1688000, 1850000, 1498000, 5900000];

const HOUSE_FEATURES_TENSOR = tf.tensor2d(HOUSE_FEATURES);
const HOUSE_PRICES_TENSOR = tf.tensor1d(HOUSE_PRICES);

function normalize(tensor) {
  let offsetValue = tf.sub(tensor, tf.min(tensor));
  let range = tf.sub(tf.max(tensor), tf.min(tensor));
  return tf.div(offsetValue, range);
}

const HOUSE_FEATURES_TENSOR_NORMALIZED = normalize(HOUSE_FEATURES_TENSOR);
HOUSE_FEATURES_TENSOR_NORMALIZED.print();


// Now actually create and define model architecture.
const model = tf.sequential();

// We will use one dense layer with 1 neuron and an input of 
// 2 values (representing house size and number of rooms)
model.add(tf.layers.dense({inputShape: [2], units: 1}));

// Choose a learning rate that is suitable for the data we are using.
const LEARNING_RATE = 0.0001;

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
  let results = await model.fit(HOUSE_FEATURES_TENSOR_NORMALIZED, HOUSE_PRICES_TENSOR, {
    epochs: 200,
    validationSplit: 0.15,
    batchSize: 1, 
    shuffle: true
  });
  
  // Once trained we can evaluate the model.
  evaluate();
}

async function evaluate(stuff) {
  // Predict answer for a single piece of data.
  model.predict(tf.tensor2d([[0.4, 0.5]])).print();
}
