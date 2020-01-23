function setup() {
  noCanvas();
  frameRate(1);
}

function draw() {
  const tensor = tf.tensor(
    [0, 0, 1920, 1080, 0, 0, 1920, 1080.5],
    [2, 2, 2],
    "int32"
  );
  const val = [];
  for (let i = 0; i < 30; i++) {
    val[i] = random(0, 100);
  }
  const shape = [2, 5, 3];

  const a = tf.tensor3d(val, shape, "int32");
  const b = tf.tensor3d(val, shape, "int32");

  const addition = a.mul(b);

  const tenseA = tf.variable(a);

  // console.log(tensor.toString());
  // console.log(a.data());
  // a.data().then(res => console.log(res));
  // tenseA.print();
  // addition.print();
  // console.log(a.dataSync());
  // console.log(a.get(1));
  // console.log(tenseA);
  tenseA.dispose();

  console.log(tf.memory().numTensors);
}
