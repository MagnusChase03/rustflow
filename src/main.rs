pub mod activation_function;
pub mod error_function;
pub mod layer;
pub mod network;

use error_function::ErrorFunction;

fn main() {
    let layers: Vec<Box<dyn layer::Layer>> = vec![
        Box::new(layer::DenseLayer::new(
            3,
            2,
            Box::new(activation_function::LeakyRelu),
        )),
        Box::new(layer::DenseLayer::new(
            2,
            2,
            Box::new(activation_function::Sigmoid),
        )),
        Box::new(layer::SoftmaxLayer::new(2)),
    ];

    let err = Box::new(error_function::LogErr {});
    let mut n = network::Network::new(layers, err);

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

    let _ = n.train(&inputs, &outputs, 10000, 0.01).unwrap();

    for i in 0..inputs.len() {
        let pred = n.forward(&inputs[i]).unwrap();
        if pred[0] >= 0.5 {
            println!("1.0");
        } else {
            println!("0.0");
        }
    }
}
