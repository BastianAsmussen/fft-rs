use numbers::complex::Complex;
use std::f64::consts::PI;

pub mod numbers;

/// A Fast Fourier Transform (FFT) implementation.
#[derive(Debug, Default, Clone)]
pub struct FFT {
    twiddle_cache: Vec<Complex>,
    current_size: usize,
}

impl FFT {
    /// Creates a new FFT instance.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            twiddle_cache: Vec::new(),
            current_size: 0,
        }
    }

    /// Computes the FFT of a complex signal.
    ///
    /// The input will be padded to the next power of 2 if necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// use fft_rs::FFT;
    /// use fft_rs::numbers::complex::Complex;
    ///
    /// let mut fft = FFT::new();
    /// let signal = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    /// let spectrum = fft.transform(&signal);
    ///
    /// assert_eq!(spectrum.len(), 2);
    /// assert!((spectrum[0] - Complex::new(1.0, 0.0)).norm() < 1e-10);
    /// ```
    #[must_use]
    pub fn transform(&mut self, signal: &[Complex]) -> Vec<Complex> {
        let n = signal.len().next_power_of_two();
        let mut padded = signal.to_vec();
        padded.resize(n, Complex::new(0.0, 0.0));

        self.fft_inplace(&mut padded);
        padded
    }

    /// Computes the FFT of a real signal.
    #[must_use]
    pub fn transform_real(&mut self, signal: &[f64]) -> Vec<Complex> {
        let n = signal.len().next_power_of_two();
        let mut data = Vec::with_capacity(n);
        data.extend(signal.iter().map(|&x| Complex::new(x, 0.0)));
        data.resize(n, Complex::new(0.0, 0.0));

        self.fft_inplace(&mut data);
        data
    }

    /// Pre-computes twiddle factors for a given size.
    #[expect(clippy::cast_precision_loss)]
    pub fn compute_twiddle_factors(&mut self, size: usize) {
        if size == self.current_size {
            return;
        }

        self.twiddle_cache.clear();
        self.twiddle_cache.reserve(size);

        // Compute base angle once.
        let base_angle = -2.0 * PI / (size as f64);

        // Generate factors using multiplication instead of repeated cos/sin.
        let mut factor = Complex::new(1.0, 0.0);
        let step = Complex::from_polar(1.0, base_angle);

        for _ in 0..size {
            self.twiddle_cache.push(factor);
            factor *= step;
        }

        self.current_size = size;
    }

    fn fft_inplace(&mut self, data: &mut [Complex]) {
        let n = data.len();
        debug_assert!(n.is_power_of_two());

        if n != self.current_size {
            self.compute_twiddle_factors(n);
        }

        Self::bit_reverse_permutation(data);

        // Butterfly computations.
        let mut size = 2;
        let mut step = n / 2;

        while size <= n {
            let half = size / 2;

            // Process butterflies in blocks for better cache locality.
            for i in (0..n).step_by(size) {
                for j in 0..half {
                    let twiddle = self.twiddle_cache[j * step];
                    let even = data[i + j];
                    let odd = data[i + j + half] * twiddle;

                    data[i + j] = even + odd;
                    data[i + j + half] = even - odd;
                }
            }

            size *= 2;
            step /= 2;
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    fn bit_reverse_permutation(data: &mut [Complex]) {
        let n = data.len();
        let bits = (n as f64).log2() as u32;

        for i in 1..n - 1 {
            let mut rev = 0;
            let mut j = i;

            for _ in 0..bits {
                rev = (rev << 1) | (j & 1);
                j >>= 1;
            }

            if rev > i {
                data.swap(i, rev);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numbers::complex::Complex;

    fn setup(samples: &[f64]) -> Vec<Complex> {
        let mut fft = FFT::new();
        fft.transform_real(samples)
    }

    #[test]
    fn test_power_of_two() {
        let result = setup(&[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        // DC component should be sum of all samples.
        assert!((result[0].re() - 4.0).abs() < 1e-10);
        assert!(result[0].im().abs() < 1e-10);

        // Test Nyquist frequency component.
        assert!((result[4].re() - 0.0).abs() < 1e-10);
        assert!(result[4].im().abs() < 1e-10);
    }

    #[test]
    fn test_non_power_of_two() {
        let result = setup(&[1.0, 1.0, 1.0]);

        // Result should be padded to length 4.
        assert_eq!(result.len(), 4);

        // DC component should be sum of all samples.
        assert!((result[0].re() - 3.0).abs() < 1e-10);
    }
}
