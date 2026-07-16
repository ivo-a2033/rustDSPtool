# rustDSPtool

Simple Rust tool for pitch shifting audio files (change pitch without changing duration) using FFT phase vocoder.

**Features**
- Offline pitch shifting for WAV files
- Real-time pitch shifting with live mouse control
- Basic FFT spectrum visualizer

**What it doesn't do**
- Only supports WAV files (no MP3, etc.)
- File-based only (no live mic input)
- Single effect: pitch shift

**Tech**
Uses `rustfft` for FFT, `cpal` for audio, `macroquad` for visuals, and Overlap-Add phase vocoder.

# Build instructions

(Rust and cargo must be installed)

**1. Be in DSP/src directory**
cd DSP/src

**2. Build**
cargo build --release

**Examples:**
cargo run --release --bin process2 input.wav output.wav -6 (writes the input audio to output file, pitch shifted by -6 semitones, same duration)

cargo run --release --bin realtimepitchshift example.wav (spawns the window that reads the mouse, the mouse x coordinate dictates the real time semitones amount by which to shift the audio)

cargo run --release  --bin rustDSP example.wav (spawns the window with a visualization of the fft bins of the audio)