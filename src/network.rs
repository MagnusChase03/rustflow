use std::io;

use super::layer;
use super::activation_function;

pub struct Network {
    pub shape: Vec<usize>,
    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,

    layers: Vec<layer::Layer>
}

impl Network {
    pub fn new(
        shape: &Vec<usize>,
        output_activation_function: Box<dyn activation_function::ActivationFunction>
    ) -> Self {
        let mut layers: Vec<layer::Layer> = vec![];
        for i in 0..shape.len() - 2 {
            layers.push(layer::Layer::new(
                shape[i],
                shape[i + 1],
                Box::new(activation_function::LeakyRelu) 
            ));
        }
        layers.push(layer::Layer::new(
            shape[shape.len() - 2],
            shape[shape.len() - 1],
            output_activation_function
        ));

        return Network{
            shape: shape.clone(),
            inputs: vec![0.0; shape[0]],
            outputs: vec![0.0; shape[shape.len() - 1]],
            layers
        };
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) -> Result<(), io::Error> {
        if inputs.len() != self.shape[0] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape"
            ));
        }

        self.inputs = inputs.clone();
        self.layers[0].forward(inputs)?;
        for i in 1..self.shape.len() - 1 {
            let prev_outputs = self.layers[i - 1].outputs.clone();
            self.layers[i].forward(&prev_outputs)?;
        }
        self.outputs = self.layers[self.shape.len() - 2].outputs.clone();

        return Ok(());
    }

    pub fn backward(
        &mut self,
        errors: &Vec<f64>,
        learning_rate: f64
    ) -> Result<Vec<f64>, io::Error> {
        if errors.len() != self.shape[self.shape.len() - 1] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid error shape"
            ));
        }

        let mut delta = self.layers[self.shape.len() - 2].backward(errors, learning_rate)?;
        for i in 1..self.shape.len() - 1 {
            let index: usize = self.shape.len() - 2 - i;
            delta = self.layers[index].backward(&delta, learning_rate)?;
        }

        return Ok(delta);
    }
}
