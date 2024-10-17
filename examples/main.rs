use rustflow;

fn main() {
    let inputs = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
        vec![1.0, 1.0, 1.0],
    ];
    let outputs = vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
    ];

    let layers: Vec<Box<dyn rustflow::layer::Layer>> = vec![
        Box::new(rustflow::layer::DenseLayer::new(3, 2, Box::new(rustflow::activation_function::LeakyRelu))),
        Box::new(rustflow::layer::DenseLayer::new(2, 2, Box::new(rustflow::activation_function::Sigmoid))),
        Box::new(rustflow::layer::SoftmaxLayer::new(2)),
    ];

    let mut model = rustflow::network::Network::new(layers, Box::new(rustflow::error_function::LogErr));
    let _ = model.train(&inputs, &outputs, 10000, 0.01).unwrap();

    for i in 0..inputs.len() {
        let pred = model.forward(&inputs[i]).unwrap();
        if pred[0] >= pred[1] {
            println!("1.0");
        } else {
            println!("0.0");
        }
    }
}
