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
const houseFeatures = [
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


// Estimated dollar cost of house for each piece of data above (1000x square footage).
const housePrice = [800000, 850000, 900000, 950000, 980000, 1000000, 1050000, 1075000, 1100000, 1150000, 1200000,  1250000 , 1300000, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000, 2000000];
