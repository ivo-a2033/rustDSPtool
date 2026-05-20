use macroquad::prelude::*;
use macroquad::audio::*;
use std::fs::File;
use std::fs::read;
use std::fs::write;

use bytemuck::cast_slice;
use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_traits::Pow;

#[macroquad::main("BasicShapes")]
async fn main() {

    
    let mut bytes = read("megalovania.wav").unwrap();
    let sound = load_sound("megalovania.wav").await.unwrap();
    play_sound(&sound, PlaySoundParams { looped: false, volume: 1.0 });

    // read metadata
    let sample_rate = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    println!("Sample rate = {}", sample_rate);

    // WAV header is typically 44 bytes for PCM
    let pcm_bytes = &bytes[44..];

    // reinterpret bytes as i16
    let sound_data: &[i16] = cast_slice(pcm_bytes);
    let mut count: usize = 0;
    let window_size: usize = (48000.0 * 0.25) as usize;

    // create our planner, whatever that is
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    let start_time = get_time();

    let scale_x = 2.0;

    println!("setup done");

    loop {
        clear_background(Color::new(0.0,0.0,0.0,0.0));

        let t = (get_time() - start_time) as f32;
        count = (t * 48000.0 * 2.0) as usize;

        // read from wherever we are window_size samples
        let mut buffer = vec![Complex { re: 0.0, im: 0.0 }; window_size];
        for i in 0..window_size {
        let w = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / window_size as f32).cos();
            buffer[i].re = sound_data[i+count] as f32 * w;
        }

        // do fft on it
        fft.process(&mut buffer);
        
        
        let notes = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"];

        for octave_shift in 0..3 {
            let base = 880.0 / 2_f32.powf(octave_shift as f32); // A5, A4, A3

            for j in 0..12 {
                let freq = base * 2_f32.powf(j as f32 / 12.0);

                let bin = (freq * window_size as f32 / sample_rate as f32) as usize;
                let x = bin as f32;

                draw_text(notes[j], x*scale_x, 350.0, 16.0, BLUE);
                draw_line(x*scale_x, 350.0, x*scale_x, 0.0, 1.0, RED);
            }
        }

        for i in (1..buffer.len()) {
            let x1 = (i as isize) as f32;
            let magnitude = -buffer[i].norm();

            draw_line(x1*scale_x, 300.0,
            x1*scale_x, 300.0 + (magnitude as f32) / 100000.0, 2.0, BLUE);
        } 
        
       
        let minimum_frame_time = 1. / 60.; // 60 FPS
        let frame_time = get_frame_time();
        if frame_time < minimum_frame_time {
            let time_to_sleep = (minimum_frame_time - frame_time) * 1000.;
            std::thread::sleep(std::time::Duration::from_millis(time_to_sleep as u64));
        }

        next_frame().await;
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
    }
    let data_start = bytes.windows(4)
    .position(|w| w == b"data")
    .unwrap() + 8; // skip 'data' + 4-byte size

    println!("{}", data_start);

    let factor = 4; // hold each sample for N samples
    let mut i = data_start;
    while i + 1 < bytes.len() {
        let sample_index = (i - data_start) / 2;
        // snap to nearest held sample
        let held_index = (sample_index / factor) * factor;
        let held_byte = data_start + held_index * 2;
        bytes[i]     = bytes[held_byte];
        bytes[i + 1] = bytes[held_byte + 1];
        i += 2;
    }

    write("output.wav", &bytes).unwrap();

}