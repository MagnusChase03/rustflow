pub trait ActivationFunction {
    fn normal(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

pub struct LeakyRelu;
impl ActivationFunction for LeakyRelu {
    fn normal(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.01 * x;
        }
        return x;
    }

    fn derivative(&self, x: f64) -> f64 {
        if x < 0.0 {
            return 0.01;
        }
        return 1.0;
    }
}

pub struct Sigmoid;
impl ActivationFunction for Sigmoid {
    fn normal(&self, x: f64) -> f64 {
        return 1.0 / (1.0 + (-x).exp());
    }

    fn derivative(&self, x: f64) -> f64 {
        return self.normal(x) * (1.0 - self.normal(x));
    }
}
