/*
* A complex number can be visually represented as a pair of numbers (a, b)
* forming a vector on a diagram called an Argand diagram, representing the
* complex plane. Re is the real axis, Im is the imaginary axis, and i is the
* "imaginary unit", that satisfies i2 = −1.
*
* Definition and Usage:
* A complex number is an expression of the form a + bi, where a and b are real
* numbers, and i is an abstract symbol, the so-called imaginary unit, whose
* meaning will be explained further below. For example, 2 + 3i is a complex
* number.
*
* For a complex number a + bi, the real number a is called its real part, and
* the real number b (not the complex number bi) is its imaginary part. The real
* part of a complex number z is denoted Re(z), the imaginary part is Im(z) for
* example, Re(2 + 3i) = 2, Im(2 + 3i) = 3.
*
* A complex number z can be identified with the ordered pair of real numbers
* (Re (z),Im (z)), which may be interpreted as coordinates of a point in a
* Euclidean plane with standard coordinates, which is then called the complex
* plane or Argand diagram,[6][a].[7] The horizontal axis is generally used to
* display the real part, with increasing values to the right, and the imaginary
* part marks the vertical axis, with increasing values upwards.
*
* A real number a can be regarded as a complex number a + 0i, whose imaginary
* part is 0. A purely imaginary number bi is a complex number 0 + bi, whose real
* part is zero. It is common to write a + 0i = a, 0 + bi = bi, and
* a + (−b)i = a − bi; for example, 3 + (−4)i = 3 − 4i.
*
* Addition and Subtraction:
* Two complex numbers a = x + yi and b = u + vi are added by separately adding
* their real and imaginary parts. That is to say:
*   a + b = (x + yi) + (u + vi) = (x + u) + (y + v)i.
* Similarly, subtraction can be performed as:
*   a - b = (x + yi) - (u + vi) = (x - u) + (y - v)i.
*
* The addition can be geometrically visualized as follows: the sum of two
* complex numbers a and b, interpreted as points in the complex plane, is the
* point obtained by building a parallelogram from the three vertices O, and the
* points of the arrows labeled a and b (provided that they are not on a line).
* Equivalently, calling these points A, B, respectively and the fourth point of
* the parallelogram X the triangles OAB and XBA are congruent.
*
* Multiplication:
* The product of two complex numbers is computed as follows:
*   (a + bi) * (c + di) = ac − bd + (ad + bc)i.
* For example,
*   (3 + 2i)(4 − i) = 3 * 4 − (2 * (−1)) + (3 * (−1) + 2 * 4) i = 14 + 5i.
* In particular, this includes as a special case the fundamental formula
*   i^2 = i * i = −1.
* This formula distinguishes the complex number i from any real number, since
* the square of any (negative or positive) real number is always a non-negative
* real number.
*
* With this definition of multiplication and addition, familiar rules for the
* arithmetic of rational or real numbers continue to hold for complex numbers.
* More precisely, the distributive property, the commutative properties
* (of addition and multiplication) hold. Therefore, the complex numbers form an
* algebraic structure known as a field, the same way as the rational or real
* numbers do.
*/

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    #[must_use]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[must_use]
    #[inline]
    pub const fn i() -> Self {
        Self::new(0.0, 1.0)
    }

    #[must_use]
    pub const fn re(&self) -> f64 {
        self.re
    }

    #[must_use]
    pub const fn im(&self) -> f64 {
        self.im
    }

    #[must_use]
    pub fn sin(self) -> Self {
        Self::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    #[must_use]
    pub fn cos(self) -> Self {
        Self::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    #[must_use]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    #[must_use]
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl Add for Complex {
    type Output = Self;

    /// a + b = (x + yi) + (u + vi) = (x + u) + (y + v)i
    fn add(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        };
    }
}

impl Sub for Complex {
    type Output = Self;

    /// a - b = (x + yi) - (u + vi) = (x - u) + (y - v)i
    fn sub(self, rhs: Self) -> Self::Output {
        Self::Output::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        };
    }
}

impl Mul for Complex {
    type Output = Self;

    /// (a + bi) * (c + di) = ac − bd + (ad + bc)i
    fn mul(self, rhs: Self) -> Self::Output {
        let re = self.re.mul_add(rhs.re, -(self.im * rhs.im));
        let im = self.re.mul_add(rhs.im, self.im * rhs.re);

        Self::Output::new(re, im)
    }
}

impl MulAssign for Complex {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        };
    }
}

impl Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::Output::new(self.re / rhs, self.im / rhs)
    }
}

impl DivAssign<f64> for Complex {
    fn div_assign(&mut self, rhs: f64) {
        *self = Self {
            re: self.re / rhs,
            im: self.im / rhs,
        };
    }
}

impl Rem<f64> for Complex {
    type Output = Self;

    fn rem(self, rhs: f64) -> Self::Output {
        Self::Output::new(self.re % rhs, self.im % rhs)
    }
}

impl RemAssign<f64> for Complex {
    fn rem_assign(&mut self, rhs: f64) {
        *self = Self {
            re: self.re % rhs,
            im: self.im % rhs,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn setup() -> (Complex, Complex) {
        let a = Complex::new(5.0, 3.0);
        let b = Complex::new(2.0, 7.0);

        (a, b)
    }

    #[test]
    fn complex_add() {
        let (a, b) = setup();

        let expected = Complex::new(7.0, 10.0);
        let actual = a + b;

        assert_eq!(expected, actual);
    }

    #[test]
    fn complex_sub() {
        let (a, b) = setup();

        let expected = Complex::new(3.0, -4.0);
        let actual = a - b;

        assert_eq!(expected, actual);
    }

    #[test]
    fn complex_mul() {
        let (a, b) = setup();

        let expected = Complex::new(-11.0, 41.0);
        let actual = a * b;

        assert_eq!(expected, actual);
    }

    #[test]
    fn complex_div() {
        let (a, _) = setup();
        let b = 2.0;

        let expected = Complex::new(2.5, 1.5);
        let actual = a / b;

        assert_eq!(expected, actual);
    }

    #[test]
    fn complex_mod() {
        let (a, _) = setup();
        let b = 2.0;

        let expected = Complex::new(1.0, 1.0);
        let actual = a % b;

        assert_eq!(expected, actual);
    }
}
