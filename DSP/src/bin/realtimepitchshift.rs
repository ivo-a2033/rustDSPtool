use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::thread;
use std::time::Duration;
use std::fs::read;
use std::fs::write;
use std::time::{Instant};
use bytemuck::cast_slice;
use rustfft::{FftPlanner, num_complex::Complex};
use rtrb::{Consumer, Producer, RingBuffer};
use macroquad::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
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

fn run_processing(
    pitch_semitones: Arc<AtomicU32>,
    input_file: String
) {

    //  CPAL SETUP

    // --- Setup: host, device, config ---
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .expect("no output device available");

    let config = device.default_output_config().unwrap();
    let sample_format = config.sample_format();
    let config: cpal::StreamConfig = config.into();

    let channels = config.channels as usize;
    let sample_rate = config.sample_rate.0;
    println!("Output device: {}", device.name().unwrap());
    println!("Sample rate: {sample_rate}, channels: {channels}");


    // need this here, even tho its wav-file and not output stream
    // need to know wav channel number to decide buffer
    let mut bytes = read(input_file).unwrap();
    let wav_channels = u16::from_le_bytes([bytes[22], bytes[23]]) as usize;
    println!("WAV channels = {}", wav_channels);

    let capacity = (config.sample_rate.0 as f32 * wav_channels as f32 * 0.1)  as usize ; // ~1 second
    let (mut producer, consumer) = RingBuffer::<f32>::new(capacity);

    // Build the stream for the actual sample format the device wants.
    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_stream::<f32>(&device, &config, channels, consumer).unwrap(),
        cpal::SampleFormat::I16 => build_stream::<i16>(&device, &config, channels, consumer).unwrap(),
        cpal::SampleFormat::U16 => build_stream::<u16>(&device, &config, channels, consumer).unwrap(),
        _ => panic!("unsupported sample format"),
    };

    stream.play().unwrap();

    // PROCESSING SETUP

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
    let sound_data_raw: &[i16] = cast_slice(pcm_bytes);

    let sound_data: Vec<i16> = if wav_channels == 2 {
        sound_data_raw.chunks(2).map(|pair| pair[0]).collect() // just take left channel
    } else {
        sound_data_raw.to_vec()
    };

    

    let window_size: usize = (sample_rate as f32 * 0.05) as usize;
    let hop_size = (window_size as f32 * 0.125) as usize; 

    // Pitch shift in semitones 
    let semitones: f32 = 0.0; 
    let mut pitch_ratio: f32 = 2.0_f32.powf(semitones / 12.0);

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
    let mut flush_pos = 0usize; // everything before this index is finalized + pushed
    while idx + window_size < sound_data.len() {

        let semitones = f32::from_bits(pitch_semitones.load(Ordering::Relaxed)) as f32;
        pitch_ratio = 2.0_f32.powf(semitones / 12.0);


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


        let flush_end = idx.min(sound_data_out.len());
        while flush_pos < flush_end {
            let mut v = sound_data_out[flush_pos];
            if norm_acc[flush_pos] > 1e-8 {
                v /= norm_acc[flush_pos];
            }
            let sample = (v / i16::MAX as f32).clamp(-1.0, 1.0);

            // backpressure: wait for room instead of dropping samples
            while producer.slots() < channels {
                thread::sleep(Duration::from_millis(1));
            }
            for _ in 0..channels {
                let _ = producer.push(sample);
            }
            flush_pos += 1;
        }

        update_bar(idx, window_size, 10000, sound_data.len());

         
    }

      let elapsed_time = now.elapsed();
    println!("Running took {} seconds.", elapsed_time.as_secs());
   
}



#[macroquad::main("Real time pitch shift")]
async fn main(){

    let args: Vec<String> = env::args().collect();
    if args.len() < 1 {
        eprintln!("Usage: {} <input.wav>", args[0]);
        std::process::exit(1);
    }
    let input_file = args[1].clone();
  

    let pitch_semitones = Arc::new(AtomicU32::new(0.0f32.to_bits()));

    let pitch_for_processing = pitch_semitones.clone();
    let handle = thread::spawn(move || {
        run_processing(pitch_for_processing, input_file);
    });

    loop {
        clear_background(Color::new(0.0,0.0,0.0,0.0));

        let (mouse_x, mouse_y) = mouse_position();
        let new_value = mouse_x/screen_width() * 24.0 - 12.0;
        pitch_semitones.store(new_value.to_bits(), Ordering::Relaxed);

        next_frame().await;
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
    }

    
}


fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: usize,
    mut consumer: Consumer<f32>,
) -> Result<cpal::Stream, Box<dyn std::error::Error>>
where
    T: cpal::Sample + cpal::SizedSample + cpal::FromSample<f32>,
{
    let err_fn = |err| eprintln!("stream error: {err}");

    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            // ---- REAL-TIME CALLBACK: this runs on cpal's audio thread ----
            // `data` is interleaved: frame0[ch0..chN], frame1[ch0..chN], ...
            for out in data.iter_mut() {
                let sample_value = consumer.pop().unwrap_or(0.0); // underrun -> silence
                if (sample_value == 0.0) {println!("I underrun");}
                *out = T::from_sample(sample_value);
            }
        },
        err_fn,
        None, // timeout
    ).unwrap();

    Ok(stream)
}