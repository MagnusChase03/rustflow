use std::io;

use super::error_function;
use super::layer;

pub struct Network {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,

    layers: Vec<Box<dyn layer::Layer>>,
    error_function: Box<dyn error_function::ErrorFunction>,
}

impl Network {
    pub fn new(
        layers: Vec<Box<dyn layer::Layer>>,
        error_function: Box<dyn error_function::ErrorFunction>,
    ) -> Self {
        return Network {
            inputs: vec![0.0; layers[0].get_input_size()],
            outputs: vec![0.0; layers[layers.len() - 1].get_output_size()],
            layers,
            error_function,
        };
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Result<Vec<f64>, io::Error> {
        if inputs.len() != self.layers[0].get_input_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape",
            ));
        }

        self.inputs = inputs.clone();
        let mut prev_outputs = self.layers[0].forward(inputs)?;
        for i in 1..self.layers.len() {
            prev_outputs = self.layers[i].forward(&prev_outputs)?;
        }
        self.outputs = prev_outputs;

        return Ok(self.outputs.clone());
    }

    pub fn backward(
        &mut self,
        errors: &Vec<f64>,
        learning_rate: f64,
    ) -> Result<Vec<f64>, io::Error> {
        if errors.len() != self.layers[self.layers.len() - 1].get_output_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid error shape",
            ));
        }

        let end = self.layers.len() - 1;
        let mut delta = self.layers[end].backward(errors, learning_rate)?;
        for i in 1..self.layers.len() {
            let index: usize = self.layers.len() - 1 - i;
            delta = self.layers[index].backward(&delta, learning_rate)?;
        }

        return Ok(delta);
    }

    pub fn train(
        &mut self,
        inputs: &Vec<Vec<f64>>,
        outputs: &Vec<Vec<f64>>,
        epochs: usize,
        learning_rate: f64,
    ) -> Result<(), io::Error> {
        if inputs.len() < 1 || inputs[0].len() != self.layers[0].get_input_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape",
            ));
        }

        if outputs.len() < 1
            || outputs[0].len() != self.layers[self.layers.len() - 1].get_output_size()
            || outputs.len() != inputs.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid output shape",
            ));
        }

        for e in 0..epochs {
            let mut total_error = 0.0;
            for i in 0..inputs.len() {
                let pred = self.forward(&inputs[i])?;
                let mut errors: Vec<f64> = Vec::new();
                for j in 0..outputs[i].len() {
                    errors.push(self.error_function.derivative(pred[j], outputs[i][j]));
                    total_error += self.error_function.normal(pred[j], outputs[i][j]);
                }
                let _ = self.backward(&errors, learning_rate)?;
            }
            println!("[Epoch {}] Error: {}", e, total_error);
        }

        return Ok(());
    }
}
