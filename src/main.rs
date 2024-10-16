pub mod layer;
pub mod activation_function;
pub mod network;

fn main() {
    let sigmoid = Box::new(activation_function::Sigmoid);
    let shape: Vec<usize> = vec![3, 2, 2];
    let mut n = network::Network::new(&shape, sigmoid);

    let inputs = vec![
        vec![-1.0, -1.0, -1.0],
        vec![-1.0, -1.0,  1.0],
        vec![-1.0,  1.0, -1.0],
        vec![-1.0,  1.0,  1.0],
        vec![ 1.0, -1.0, -1.0],
        vec![ 1.0, -1.0,  1.0],
        vec![ 1.0,  1.0, -1.0],
        vec![ 1.0,  1.0,  1.0],
    ];
    let outputs = vec![
        -1.0,
         1.0,
         1.0,
        -1.0,
         1.0,
        -1.0,
        -1.0,
         1.0
    ];

    for _ in 0..1000 {
        for i in 0..inputs.len() {
            let _ = n.forward(&inputs[i]).unwrap();
            let pred = &n.outputs;
            if outputs[i] == 1.0 {
                let errors = vec![-1.0 / pred[0], 1.0 / (1.0 - pred[1])];
                let _ = n.backward(&errors, 0.01).unwrap();
            } else {
                let errors = vec![1.0 / (1.0 - pred[0]), -1.0 / pred[1]];
                let _ = n.backward(&errors, 0.01).unwrap();
            } 
        }
    }

    for i in 0..inputs.len() {
        let _ = n.forward(&inputs[i]).unwrap();
        let pred = &n.outputs;
        if pred[0] >= pred[1] {
            println!("1.0");
        } else {
            println!("-1.0");
        }
    }
}
