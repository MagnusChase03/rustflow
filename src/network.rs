use std::io;

use super::layer;
use super::activation_function;

pub struct Network {
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,

    layers: Vec<Box<dyn layer::Layer>>,
}

impl Network {
    pub fn new(
        layers: Vec<Box<dyn layer::Layer>>
    ) -> Self {
        return Network{
            inputs: vec![0.0; layers[0].get_input_size()],
            outputs: vec![0.0; layers[layers.len() - 1].get_output_size()],
            layers,
        };
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Result<Vec<f64>, io::Error> {
        if inputs.len() != self.layers[0].get_input_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape"
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
        learning_rate: f64
    ) -> Result<Vec<f64>, io::Error> {
        if errors.len() != self.layers[self.layers.len() - 1].get_output_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid error shape"
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
}
