
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

fn update_bar(idx: usize, window_size: usize, precision: i32, sound_data_len: usize){
    if idx % window_size <= precision as usize{ 
        let progress = idx as f32 / sound_data_len as f32;
        let filled = (progress * 10.0) as usize;
        print!("\r[{}{}] {:.0}%", "#".repeat(filled), ".".repeat(10 - filled), progress * 100.0);
        use std::io::Write;
        std::io::stdout().flush().unwrap();
    }
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

    // thisll hold the output
    let mut sound_data_out = vec![0i16; sound_data.len()];

    let mut count: usize = 0;
    let window_size: usize = (sample_rate as f32 * 0.125) as usize;
    let hop_size = (window_size as f32 * 0.5) as usize;

    // create our planner, whatever that is
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(window_size);
    let ifft = planner.plan_fft_inverse(window_size); // each of these is its own obj

    let mut prev_phase = vec![0.0f32; window_size];
    let mut phase_acc = vec![0.0f32; window_size];
    let mut phase_acc_out = vec![0.0f32; window_size];

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

        // --- processing happens here ---


        // phase vocoding
        let mut mag = vec![0.0f32; window_size];
        let mut phase = vec![0.0f32; window_size];

        let omega = 2.0 * std::f32::consts::PI * hop_size as f32 / window_size as f32;

        for k in 0..window_size {
            mag[k] = buffer[k].norm();
            phase[k] = buffer[k].arg();
        }

        let mut new_buffer = vec![Complex::new(0.0, 0.0); window_size];
        new_buffer.fill(Complex::new(0.0, 0.0)); // once before
        for k in 0..window_size {
            let mut delta = phase[k] - prev_phase[k];
            prev_phase[k] = phase[k];

            delta -= k as f32 * omega;
            delta -= 2.0 * std::f32::consts::PI * (delta / (2.0 * std::f32::consts::PI)).round();

            phase_acc[k] += k as f32 * omega + delta;
            let true_freq = k as f32 * omega + delta;
            let pitch = 2.0;
            let target = (k as f32 * pitch) as usize;
            if target < window_size {
                phase_acc_out[target] += true_freq * pitch;
                new_buffer[target] += Complex::from_polar(mag[k], phase_acc_out[target]);            
            }

            // dump frame 0 only
            if idx == 0 {
                let mut f = std::fs::File::create("debug_frame.csv").unwrap();
                use std::io::Write;
                writeln!(f, "bin,mag_in,phase_in,mag_out,phase_out").unwrap();
                for k in 0..window_size {
                    writeln!(f, "{},{},{},{},{}", k,
                        mag[k], phase[k],
                        new_buffer[k].norm(), new_buffer[k].arg()
                    ).unwrap();
                }
            }
        }
        buffer.copy_from_slice(&new_buffer); // once after
        // --- ---

        ifft.process(&mut buffer);

        let scale = 1.0 / window_size as f32; // the ifft call doesnt scale by default

        for i in 0..window_size {
            sound_data_out[i + idx] += (buffer[i].re * scale) as i16;
        }

        
        idx += hop_size;

        update_bar(idx, window_size, 10000, sound_data.len());
    }

    update_bar(idx, window_size, 10000, sound_data.len());


   
    let out_bytes: &[u8] = cast_slice(&sound_data_out); // i16 back to bytes
    let max_len = bytes.len() - data_start;
    let copy_len = out_bytes.len().min(max_len);

    bytes[data_start..data_start + copy_len]
    .copy_from_slice(&out_bytes[..copy_len]); // dont overwrite header, wouldnt be nice

    write("output.wav", &bytes).unwrap();
}