// Create a model
const model = tf.sequential();

// Create a hidden layer
const hidden = tf.layers.dense({
  units: 4, // number of nodes
  inputShape: [2], // input shape
  activation: "sigmoid" // activation function, got a lot
});
// Add the hidden layer
model.add(hidden);

// Create another output layer
const output = tf.layers.dense({
  units: 3,
  //   here the input shape is 'inferred from the previous layer'
  activation: "sigmoid"
});
// Add output layer
model.add(output);

// An optimizer using gradient desc
const sgdOpt = tf.train.sgd(0.1);

// Done configure the model, then compile it
model.compile({
  optimizer: sgdOpt,
  loss: tf.losses.meanSquaredError
});

const xs = tf.tensor2d([
  [0.25, 0.92],
  [0.51, 0.25],
  [0.45, 0.92],
  [0.29, 0.82]
]);

const ys = tf.tensor2d([
  [0.25, 0.92, 0.54],
  [0.51, 0.13, 0.25],
  [0.67, 0.45, 0.92],
  [0.29, 0.82, 0.33]
]);

async function train() {
  for (let i = 0; i < 10; i++) {
    const res = await model.fit(xs, ys, {
      epochs: 5,
      shuffle: true
    });
    console.log(res.history.loss[0]);
  }
}

train().then(() => {
  console.log("training complete");
  let outputs = model.predict(xs);
  outputs.print();
});
