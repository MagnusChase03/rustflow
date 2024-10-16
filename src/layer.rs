use rand;
use rand::Rng;

use std::io;
use super::activation_function;

pub struct Layer {
    pub input_size: usize,
    pub output_size: usize,
    
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    cache: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,

    activation_function: Box<dyn activation_function::ActivationFunction>
}

impl Layer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation_function: Box<dyn activation_function::ActivationFunction>
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights: Vec<f64> = Vec::new();
        for _ in 0..output_size * input_size {
            weights.push(rng.gen());
        }

        return Layer{
            input_size,
            output_size,
            inputs: vec![0.0; input_size],
            outputs: vec![0.0; output_size],
            cache: vec![0.0; output_size],
            weights,
            bias: vec![0.0; output_size],
            activation_function
        }; 
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Result<(), io::Error> {
        if inputs.len() != self.input_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape."
            ));
        }

        for i in 0..self.input_size {
            self.inputs[i] = inputs[i];
        }

        for i in 0..self.output_size {
            self.outputs[i] = self.bias[i];
            for j in 0..self.input_size {
                self.outputs[i] += inputs[j] *
                    self.weights[i * self.input_size + j];
            }
            self.cache[i] = self.outputs[i];
            self.outputs[i] = self.activation_function.normal(self.outputs[i]);
        }

        return Ok(());
    }

    pub fn backward(
        &mut self,
        errors: &Vec<f64>,
        learning_rate: f64
    ) -> Result<Vec<f64>, io::Error> {
        if self.output_size != errors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid error shape."
            ));
        }

        let mut result = vec![0.0; self.input_size];
        for i in 0..self.output_size {
            self.bias[i] -= errors[i] *
                    self.activation_function.derivative(self.cache[i]) *
                    learning_rate;

            for j in 0..self.input_size {
                result[j] += errors[i] * 
                    self.activation_function.derivative(self.cache[i]) *
                    self.weights[i * self.input_size + j];

                self.weights[i * self.input_size + j] -= errors[i] *
                    self.activation_function.derivative(self.cache[i]) *
                    self.inputs[j] *
                    learning_rate;
            }
        }

        return Ok(result);
    }
}
