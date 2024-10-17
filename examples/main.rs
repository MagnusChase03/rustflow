use rustflow;

use std::io;
use std::fs;
use csv;

use rand;
use rand::seq::SliceRandom;

fn load_iris(filepath: &str) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), io::Error> {
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut outputs: Vec<Vec<f64>> = Vec::new();

    let f = fs::File::open(filepath)?;
    let mut rdr = csv::ReaderBuilder::new().from_reader(f);

    let mut row: usize = 0;
    for line in rdr.records() {
        let mut col = 0;
        let record = line?;
        inputs.push(Vec::new());
        outputs.push(vec![0.0, 0.0, 0.0]);
        for value in record.iter() {
            let num = value.parse::<f64>().unwrap();
            if col == 4 {
                outputs[row][num as usize] = 1.0;
            } else {
                inputs[row].push(num);
            }
            col += 1;
        }
        row += 1;
    }

    return Ok((inputs, outputs));
}

fn arg_max(values: Vec<f64>) -> usize {
    let mut max = values[0];
    let mut maxi: usize = 0;

    for i in 1..values.len() {
        if values[i] > max {
            max = values[i];
            maxi = i;
        }
    }

    return maxi;
}

fn main() {
    let (inputs, outputs) = load_iris("examples/iris.csv").unwrap();
    let layers: Vec<Box<dyn rustflow::layer::Layer>> = vec![
        Box::new(rustflow::layer::DenseLayer::new(4, 3, Box::new(rustflow::activation_function::LeakyRelu))),
        Box::new(rustflow::layer::DenseLayer::new(3, 3, Box::new(rustflow::activation_function::Sigmoid))),
        Box::new(rustflow::layer::SoftmaxLayer::new(3)),
    ];

    let mut model = rustflow::network::Network::new(layers, Box::new(rustflow::error_function::LogErr));
    let _ = model.train(&inputs, &outputs, 1000, 0.01).unwrap();

    let mut correct = 0;
    for i in 0..inputs.len() {
        let pred = model.forward(&inputs[i]).unwrap();
        if arg_max(pred) == arg_max(outputs[i].clone()) {
            correct += 1;
        }
    }
    println!("Acc: {}", correct as f64 / outputs.len() as f64);
}
