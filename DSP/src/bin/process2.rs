use std::fs::read;
use std::fs::write;
use std::time::{Instant};
use bytemuck::cast_slice;
use rustfft::{FftPlanner, num_complex::Complex};
use std::env;         

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / size as f32).cos())
        .collect()
}

fn update_bar(idx: usize, window_size: usize, precision: i32, sound_data_len: usize) {
    if idx % window_size <= precision as usize {
        let progress = idx as f32 / sound_data_len as f32;
        let filled = (progress * 10.0) as usize;
        print!(
            "\r[{}{}] {:.0}%",
            "#".repeat(filled),
            ".".repeat(10 - filled),
            progress * 100.0
        );
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: {} <input.wav> <output.wav> <semitones>", args[0]);
        std::process::exit(1);
    }

    let input_file = &args[1];
    let output_file = &args[2];
    let semitones: f32 = args[3].parse().expect("Semitones must be a number (e.g. -6 or 3.5)");

    let mut bytes = read(input_file).expect("Failed to read input WAV");

    // read metadata
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    println!("Sample rate = {}", sample_rate);

    let data_start = bytes
        .windows(4)
        .position(|w| w == b"data")
        .unwrap()
        + 8; // skip 'data' + 4-byte size
    println!("data_start = {}", data_start);

    let pcm_bytes = &bytes[data_start..];
    let sound_data: &[i16] = cast_slice(pcm_bytes);

    let window_size: usize = (sample_rate as f32 * 0.05) as usize;
    let hop_size = (window_size as f32 * 0.125) as usize; 

    // Pitch shift in semitones 
    let pitch_ratio: f32 = 2.0_f32.powf(semitones / 12.0);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_size);
    let ifft = planner.plan_fft_inverse(window_size);
    let window = hann_window(window_size);

    // OLA normalization accumulator — prevents amplitude drop from windowing
    let mut norm_acc = vec![0.0f32; sound_data.len()];
    let mut sound_data_out = vec![0.0f32; sound_data.len()];

    let two_pi = 2.0 * std::f32::consts::PI;
    let mut prev_phase: Vec<f32> = vec![0.0; window_size];
    let mut phase_acc: Vec<f32> = vec![0.0; window_size];

    println!("setup done");
    let now = Instant::now();

    let mut idx = 0usize;
    while idx + window_size < sound_data.len() {
        // Analysis frame
        let mut buffer: Vec<Complex<f32>> = (0..window_size)
            .map(|i| Complex {
                re: sound_data[i + idx] as f32 * window[i],
                im: 0.0,
            })
            .collect();
        fft.process(&mut buffer);

        // --- Pitch shifting ---
        let mut new_buffer = vec![Complex { re: 0.0f32, im: 0.0 }; window_size];
        // Only loop through positive frequencies (up to Nyquist)
        let half_window = window_size / 2;

        for k in 0..=half_window {
            let mag = buffer[k].norm();
            let phase = buffer[k].arg();

            let expected_advance = two_pi * k as f32 * hop_size as f32 / window_size as f32;
            let delta = phase - prev_phase[k] - expected_advance;
            let delta_wrapped = delta - two_pi * (delta / two_pi).round();

            let dst = (k as f32 * pitch_ratio).round() as usize;
            
            // Ignore if it exceeds the positive frequency boundary
            if dst <= half_window {
                let syn_advance = two_pi * dst as f32 * hop_size as f32 / window_size as f32
                    + delta_wrapped * pitch_ratio;
                phase_acc[dst] += syn_advance;

                // Scale amplitude by pitch_ratio to compensate for gaps/sparsity
                let amplitude_scale = if pitch_ratio > 1.0 { pitch_ratio } else { 1.0 };
                let complex_val = Complex::from_polar(mag * amplitude_scale, phase_acc[dst]);
                prev_phase[k] = phase;

                // pos bin
                new_buffer[dst] += complex_val;

                // neg bin
                if dst > 0 && dst < half_window {
                    let conjugate_dst = window_size - dst;
                    new_buffer[conjugate_dst] += Complex {
                        re: complex_val.re,
                        im: -complex_val.im, // Conjugate phase
                    };
                }
            }
        }
        buffer = new_buffer;
        ifft.process(&mut buffer);
        let scale = 1.0 / window_size as f32;

        // OLA: accumulate output and normalization window
        let end = (idx + window_size).min(sound_data_out.len());
        for i in 0..(end - idx) {
            sound_data_out[i + idx] += buffer[i].re * scale * window[i];
            norm_acc[i + idx] += window[i] * window[i];
        }

        idx += hop_size;
        update_bar(idx, window_size, 10000, sound_data.len());
    }

    println!(); // newline after progress bar

    // Normalize OLA output to compensate for windowing
    for i in 0..sound_data_out.len() {
        if norm_acc[i] > 1e-8 {
            sound_data_out[i] /= norm_acc[i];
        }
    }

    // Convert back to i16 and patch into original WAV bytes
    let out_i16: Vec<i16> = sound_data_out
        .iter()
        .map(|&s| s.clamp(-32768.0, 32767.0) as i16)
        .collect();
    let out_bytes: &[u8] = cast_slice(&out_i16);
    let max_len = bytes.len() - data_start;
    let copy_len = out_bytes.len().min(max_len);
    bytes[data_start..data_start + copy_len].copy_from_slice(&out_bytes[..copy_len]);

    write("output.wav", &bytes).unwrap();
    println!("Done → output.wav");
    let elapsed_time = now.elapsed();
    println!("Running took {} seconds.", elapsed_time.as_secs());
}