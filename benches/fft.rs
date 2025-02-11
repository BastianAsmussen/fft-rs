use criterion::{
    criterion_group, criterion_main, AxisScale, BenchmarkId, Criterion, PlotConfiguration,
};
use fft_rs::numbers::complex::Complex;
use rand::Rng;
use std::f64::consts::PI;
use std::hint::black_box;

fn generate_sine_wave(freq: f64, samples: usize) -> Vec<Complex> {
    (0..samples)
        .map(|i| {
            let t = i as f64 / samples as f64;

            Complex::new((2.0 * PI * freq * t).sin(), 0.0)
        })
        .collect()
}

fn generate_noise(num_samples: usize) -> Vec<Complex> {
    let mut rng = rand::rng();

    (0..num_samples)
        .map(|_| Complex::new(rng.random_range(-1.0..1.0), 0.0))
        .collect()
}

fn bench_fft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Size Scaling");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let mut fft = fft_rs::FFT::new();

    for size in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] {
        let sine = generate_sine_wave(1.0, size);

        group.bench_with_input(BenchmarkId::new("sine_wave", size), &sine, |b, input| {
            b.iter(|| fft.transform(black_box(input)));
        });
    }

    group.finish();
}

fn bench_fft_signals(c: &mut Criterion) {
    // Fixed size for signal type comparison.
    const SIZE: usize = 1024;

    let mut group = c.benchmark_group("FFT Signal Types");
    let mut fft = fft_rs::FFT::new();

    // Single frequency.
    let sine = generate_sine_wave(1.0, SIZE);

    // Multiple frequencies.
    let complex_signal: Vec<Complex> = generate_sine_wave(1.0, SIZE)
        .iter()
        .zip(generate_sine_wave(10.0, SIZE))
        .map(|(a, b)| *a + b)
        .collect();

    // White noise.
    let noise = generate_noise(SIZE);

    // DC signal (all same value).
    let dc: Vec<Complex> = vec![Complex::new(1.0, 0.0); SIZE];

    group.bench_function("sine_1hz", |b| b.iter(|| fft.transform(black_box(&sine))));
    group.bench_function("complex_signal", |b| {
        b.iter(|| fft.transform(black_box(&complex_signal)));
    });

    group.bench_function("white_noise", |b| {
        b.iter(|| fft.transform(black_box(&noise)));
    });

    group.bench_function("dc_signal", |b| b.iter(|| fft.transform(black_box(&dc))));

    group.finish();
}

criterion_group!(benches, bench_fft_sizes, bench_fft_signals);
criterion_main!(benches);
