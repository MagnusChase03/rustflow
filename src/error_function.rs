pub trait ErrorFunction {
    fn normal(&self, x: f64, y: f64) -> f64;
    fn derivative(&self, x: f64, y: f64) -> f64;
}

pub struct MSE;
impl ErrorFunction for MSE {
    fn normal(&self, x: f64, y: f64) -> f64 {
        return (x - y) * (x - y);
    }

    fn derivative(&self, x: f64, y: f64) -> f64 {
        return 2.0 * (x - y);
    }
}

pub struct LogErr;
impl ErrorFunction for LogErr {
    fn normal(&self, x: f64, y: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            panic!("Invalid input");
        }

        if y == 1.0 {
            return -x.ln();
        } else if y == 0.0 {
            return -(1.0 - x).ln();
        }
        panic!("Invalid output class"); 
    }

    fn derivative(&self, x: f64, y: f64) -> f64 {
        if x < 0.0 || x > 1.0 {
            panic!("Invalid input");
        }

        if y == 1.0 {
            return -1.0 / x;
        } else if y == 0.0 {
            return 1.0 / (1.0 - x);
        }
        panic!("Invalid output class"); 
    }
}
