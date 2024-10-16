pub mod layer;
pub mod activation_function;
pub mod error_function;
pub mod network;

use error_function::ErrorFunction;

fn main() {
    let layers: Vec<Box<dyn layer::Layer>> = vec![
        Box::new(layer::DenseLayer::new(3, 2, Box::new(activation_function::LeakyRelu))),
        Box::new(layer::DenseLayer::new(2, 2, Box::new(activation_function::Sigmoid))),
        Box::new(layer::SoftmaxLayer::new(2))
    ];

    let mut n = network::Network::new(layers);

    let inputs = vec![
        vec![ 0.0,  0.0,  0.0],
        vec![ 0.0,  0.0,  1.0],
        vec![ 0.0,  1.0,  0.0],
        vec![ 0.0,  1.0,  1.0],
        vec![ 1.0,  0.0,  0.0],
        vec![ 1.0,  0.0,  1.0],
        vec![ 1.0,  1.0,  0.0],
        vec![ 1.0,  1.0,  1.0],
    ];
    let outputs = vec![
         0.0,
         1.0,
         1.0,
         0.0,
         1.0,
         0.0,
         0.0,
         1.0
    ];

    let log_err = error_function::LogErr{};
    for _ in 0..10000 {
        let mut correct = 0.0;
        for i in 0..inputs.len() {
            let pred = n.forward(&inputs[i]).unwrap();
            if outputs[i] == 1.0 {
                let errors = vec![
                    log_err.derivative(pred[0], 1.0),
                    log_err.derivative(pred[1], 0.0)
                ];
                let _ = n.backward(&errors, 0.001).unwrap();

                if pred[0] >= pred[1] {
                    correct += 1.0;
                }
            } else {
                let errors = vec![
                    log_err.derivative(pred[0], 0.0),
                    log_err.derivative(pred[1], 1.0)
                ];
                let _ = n.backward(&errors, 0.001).unwrap();

                if pred[0] < pred[1] {
                    correct += 1.0;
                }
            }
        }
        println!("{}%", correct / inputs.len() as f64 * 100.0);
    }

    for i in 0..inputs.len() {
        let pred = n.forward(&inputs[i]).unwrap();
        if pred[0] >= pred[1] {
            println!("1.0");
        } else {
            println!("0.0");
        }
    }
}
