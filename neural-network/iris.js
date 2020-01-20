const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
const iris = require("./iris.json");
const irisTesting = require("./iris-testing.json");

// convert/setup our data
const trainingData = tf.tensor2d(
  iris.map(item => [
    item.sepalLength,
    item.sepalWidth,
    item.petalLength,
    item.petalWidth
  ])
);

const outputData = tf.tensor2d(
  iris.map(item => [
    item === "setosa" ? 1 : 0,
    item === "versicolor" ? 1 : 0,
    item === "virginica" ? 1 : 0
  ])
);

const testingData = tf.tensor2d(
  irisTesting.map(item => [
    item.sepalLength,
    item.sepalWidth,
    item.petalLength,
    item.petalWidth
  ])
);

// build neural network
const model = tf.sequential();

model.add(
  tf.layers.dense({
    inputShape: [4],
    activation: "sigmoid",
    units: 5
  })
);
model.add(
  tf.layers.dense({
    inputShape: [5],
    activation: "sigmoid",
    units: 3
  })
);
model.add(
  tf.layers.dense({
    activation: "sigmoid",
    units: 3
  })
);
model.compile({
  loss: "meanSquaredError",
  optimizer: tf.train.adam(0.06)
});
// train/fit our network
const startTime = Date.now();
model.fit(trainingData, outputData, { epochs: 100 }).then(history => {
    console.log(history);
  //   console.log("DONE", Date.now() - startTime);
  model.predict(testingData).print();
});
// test network
