pub mod layer;
pub mod activation_function;
pub mod network;

fn main() {
    let sigmoid = Box::new(activation_function::Sigmoid);
    let shape: Vec<usize> = vec![3, 2, 2];
    let mut n = network::Network::new(&shape, sigmoid);

    let inputs = vec![1.0, 2.0, 3.0];
    let _ = n.forward(&inputs).unwrap();
    println!("{:?}", &n.outputs);

    let errors = vec![-0.1, 0.1];
    let _ = n.backward(&errors, 0.01).unwrap();
    let _ = n.forward(&inputs).unwrap();
    println!("{:?}", &n.outputs);
}
