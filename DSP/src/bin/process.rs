
use std::fs::File;
use std::fs::read;
use std::fs::write;

use bytemuck::cast_slice;
use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_traits::Pow;

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        .collect()
}

fn main() {
    
    let mut bytes = read("megalovania.wav").unwrap();


    // read metadata
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    println!("Sample rate = {}", sample_rate);

    // WAV header is typically 44 bytes for PCM
    let pcm_bytes = &bytes[44..];

    // reinterpret bytes as i16
    let sound_data: &[i16] = cast_slice(pcm_bytes);
    let mut count: usize = 0;
    let window_size: usize = (sample_rate as f32 * 0.125) as usize;
    let hop_size = (window_size as f32 * 0.5) as usize;

    // create our planner, whatever that is
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_size);

    let scale_x = 4.0;

    println!("setup done");

    let data_start = bytes.windows(4)
    .position(|w| w == b"data")
    .unwrap() + 8; // skip 'data' + 4-byte size


    println!("{}", data_start);
    let mut idx = data_start;
    let window = hann_window(window_size);

    while idx + window_size < sound_data.len() {
        let mut buffer: Vec<Complex<f32>> = (0..window_size)
            .map(|i| Complex { re: sound_data[i + idx] as f32 * window[i], im: 0.0 })
            .collect();
        fft.process(&mut buffer);
        idx += hop_size;

        if idx % 1000000 <= 78 {
            let progress = idx as f32 / sound_data.len() as f32;
            let filled = (progress * 10.0) as usize;
            print!("\r[{}{}] {:.0}%", "#".repeat(filled), ".".repeat(10 - filled), progress * 100.0);
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }

    let progress = idx as f32 / sound_data.len() as f32;
    let filled = (progress * 10.0) as usize;
    print!("\r[{}{}] {:.0}%", "#".repeat(filled), ".".repeat(10 - filled), progress * 100.0);
    use std::io::Write;
    std::io::stdout().flush().unwrap();

   
    write("output.wav", &bytes).unwrap();

}