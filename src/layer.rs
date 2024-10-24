use rand;
use rand::Rng;

use super::activation_function;
use std::io;

pub trait Layer {
    fn get_input_size(&self) -> usize;
    fn get_output_size(&self) -> usize;

    fn forward(&mut self, inputs: &Vec<f64>) -> Result<Vec<f64>, io::Error>;
    fn backward(&mut self, errors: &Vec<f64>, learning_rate: f64) -> Result<Vec<f64>, io::Error>;
}

pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,

    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
    cache: Vec<f64>,
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,

    activation_function: Box<dyn activation_function::ActivationFunction>,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation_function: Box<dyn activation_function::ActivationFunction>,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights: Vec<f64> = Vec::new();
        for _ in 0..output_size * input_size {
            weights.push(rng.gen::<f64>() / 1000.0);
        }

        return DenseLayer {
            input_size,
            output_size,
            inputs: vec![0.0; input_size],
            outputs: vec![0.0; output_size],
            cache: vec![0.0; output_size],
            weights,
            bias: vec![0.0; output_size],
            activation_function,
        };
    }
}

impl Layer for DenseLayer {
    fn get_input_size(&self) -> usize {
        return self.input_size;
    }

    fn get_output_size(&self) -> usize {
        return self.output_size;
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> Result<Vec<f64>, io::Error> {
        if inputs.len() != self.input_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape.",
            ));
        }
        self.inputs = inputs.clone();

        for i in 0..self.output_size {
            self.outputs[i] = self.bias[i];
            for j in 0..self.input_size {
                self.outputs[i] += inputs[j] * self.weights[i * self.input_size + j];
            }
            self.cache[i] = self.outputs[i];
            self.outputs[i] = self.activation_function.normal(self.outputs[i]);

            if self.outputs[i].is_nan() || self.cache[i].is_nan() {
                panic!("[ERROR] NaN result {:?} {:?}", self.weights, self.inputs);
            }
        }

        return Ok(self.outputs.clone());
    }

    fn backward(&mut self, errors: &Vec<f64>, learning_rate: f64) -> Result<Vec<f64>, io::Error> {
        if self.output_size != errors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid error shape.",
            ));
        }

        let mut result = vec![0.0; self.input_size];
        for i in 0..self.output_size {
            let gradient = errors[i] * self.activation_function.derivative(self.cache[i]);
            self.bias[i] -= gradient * learning_rate;
            for j in 0..self.input_size {
                result[j] += gradient * self.weights[i * self.input_size + j];

                self.weights[i * self.input_size + j] -= gradient
                    * self.inputs[j]
                    * learning_rate;

                if self.weights[i * self.input_size + j].is_nan() {
                    panic!("[ERROR] NaN result {:?} {:?} {:?} {:?}", gradient, errors, self.cache, self.inputs);
                }
            }
        }

        return Ok(result);
    }
}

pub struct SoftmaxLayer {
    pub input_size: usize,

    pub inputs: Vec<f64>,
    pub outputs: Vec<f64>,
}

impl SoftmaxLayer {
    pub fn new(input_size: usize) -> Self {
        return SoftmaxLayer {
            input_size,
            inputs: vec![0.0; input_size],
            outputs: vec![0.0; input_size],
        };
    }
}

impl Layer for SoftmaxLayer {
    fn get_input_size(&self) -> usize {
        return self.input_size;
    }

    fn get_output_size(&self) -> usize {
        return self.input_size;
    }

    fn forward(&mut self, inputs: &Vec<f64>) -> Result<Vec<f64>, io::Error> {
        if inputs.len() != self.input_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid input shape.",
            ));
        }
        self.inputs = inputs.clone();

        let mut total: f64 = 0.0;
        for i in 0..self.input_size {
            total += inputs[i].exp();
        }

        for i in 0..self.input_size {
            self.outputs[i] = inputs[i].exp() / total;
        }

        return Ok(self.outputs.clone());
    }

    fn backward(&mut self, errors: &Vec<f64>, _learning_rate: f64) -> Result<Vec<f64>, io::Error> {
        if self.input_size != errors.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid error shape.",
            ));
        }

        let mut result = vec![0.0; self.input_size];
        for i in 0..self.input_size {
            result[i] += errors[i] * self.outputs[i] * (1.0 - self.outputs[i]);
            if result[i].is_nan() {
                panic!("[ERROR] NaN result {:?} {:?}", errors, self.inputs);
            }
        }

        return Ok(result);
    }
}
