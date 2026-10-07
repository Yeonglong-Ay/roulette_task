# sounds.py
"""
Generates and plays sounds for the roulette task.

Three sounds
------------
Win  — bright ascending major arpeggio (C5 → E5 → G5 → C6, ~0.4 s)
Loss — dull descending minor arpeggio  (G3 → Eb3 → C3, ~0.4 s)
Spin — white-noise burst with rising pitch sweep (~1.8 s), played at
       wheel spin start and looped until the wheel stops.

Volume balance
--------------
High-frequency tones are perceived as louder than low-frequency ones
at the same digital amplitude (equal-loudness contours / Fletcher-Munson).
Win notes (523–1047 Hz) are attenuated relative to loss notes (131–196 Hz)
so both feel equally prominent at the same system volume.
    Win  amplitude : 0.35  (reduced from 0.45 — compensates for HF loudness)
    Loss amplitude : 0.65  (raised from 0.55 — compensates for LF quietness)

Playback
--------
Uses `afplay` (macOS built-in) via subprocess in a daemon thread so sounds
never block the Tkinter event loop.  Falls back to `aplay` (Linux) silently.
"""

import math
import os
import random
import struct
import subprocess
import tempfile
import threading
import wave
from pathlib import Path


# ---------------------------------------------------------------------------
# WAV synthesis helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 44_100


def _sine_samples(freq: float, duration: float, amplitude: float = 0.6) -> list:
    """16-bit PCM samples for a sine wave with a short fade-in/fade-out."""
    n    = int(SAMPLE_RATE * duration)
    fade = int(SAMPLE_RATE * 0.015)
    samples = []
    for i in range(n):
        t     = i / SAMPLE_RATE
        value = amplitude * math.sin(2 * math.pi * freq * t)
        if i < fade:
            value *= i / fade
        elif i > n - fade:
            value *= (n - i) / fade
        samples.append(int(value * 32767))
    return samples


def _mix(a: list, b: list) -> list:
    """Mix (sum + clip) two sample lists of potentially different lengths."""
    length = max(len(a), len(b))
    out = []
    for i in range(length):
        sa = a[i] if i < len(a) else 0
        sb = b[i] if i < len(b) else 0
        out.append(max(-32768, min(32767, sa + sb)))
    return out


def _write_wav(path: str, samples: list):
    """Write 16-bit PCM mono samples to a WAV file."""
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))


def _build_win_wav(path: str):
    """
    Ascending major arpeggio: C5 → E5 → G5 → C6
    Amplitude 0.38 — slightly reduced from pure sine to compensate for
    high-frequency perceptual loudness (Fletcher-Munson effect).
    """
    notes    = [523.25, 659.25, 783.99, 1046.50]   # C5 E5 G5 C6
    note_dur = 0.14
    note_gap = 0.09
    total_n  = int(SAMPLE_RATE * (note_gap * (len(notes) - 1) + note_dur))
    mixed    = [0] * total_n

    for idx, freq in enumerate(notes):
        offset   = int(idx * note_gap * SAMPLE_RATE)
        primary  = _sine_samples(freq,     note_dur, amplitude=0.38)
        harmonic = _sine_samples(freq * 2, note_dur, amplitude=0.10)
        note     = _mix(primary, harmonic)
        for i, s in enumerate(note):
            pos = offset + i
            if pos < total_n:
                mixed[pos] = max(-32768, min(32767, mixed[pos] + s))

    _write_wav(path, mixed)


def _build_loss_wav(path: str):
    """
    Descending minor arpeggio: G3 → Eb3 → C3
    Amplitude 0.50 — boosted relative to win to compensate for
    low-frequency perceptual quietness (Fletcher-Munson effect).
    Balanced so win and loss feel equally prominent at the same volume.
    """
    notes    = [196.00, 155.56, 130.81]   # G3 Eb3 C3
    note_dur = 0.18
    note_gap = 0.12
    total_n  = int(SAMPLE_RATE * (note_gap * (len(notes) - 1) + note_dur))
    mixed    = [0] * total_n

    for idx, freq in enumerate(notes):
        offset = int(idx * note_gap * SAMPLE_RATE)
        note   = _sine_samples(freq,        note_dur, amplitude=0.50)
        beat   = _sine_samples(freq * 0.98, note_dur, amplitude=0.12)
        note   = _mix(note, beat)
        for i, s in enumerate(note):
            pos = offset + i
            if pos < total_n:
                mixed[pos] = max(-32768, min(32767, mixed[pos] + s))

    _write_wav(path, mixed)


def _build_spin_wav(path: str):
    """
    Roulette wheel spin sound (~1.8 s).

    Design: filtered white noise + a rising pitch sweep on a sine carrier.

    - White noise gives the tactile "ball rolling on wood" texture.
    - A sine tone sweeps from 180 Hz → 600 Hz over 1.8 s, mimicking the
      rising pitch of a slowing wheel (faster = higher pitch in the brain).
    - Both are amplitude-enveloped: fast attack (0.05 s), sustain, then a
      gradual 0.4 s decay at the end so it doesn't cut off abruptly.
    - Volume kept moderate (noise 0.25, sweep 0.20) so it doesn't compete
      with win/loss sounds.
    """
    duration  = 1.8
    n         = int(SAMPLE_RATE * duration)
    attack    = int(SAMPLE_RATE * 0.05)
    decay     = int(SAMPLE_RATE * 0.40)

    rng = random.Random(42)   # deterministic seed for reproducibility

    samples = []
    for i in range(n):
        t = i / SAMPLE_RATE

        # Amplitude envelope
        if i < attack:
            env = i / attack
        elif i > n - decay:
            env = (n - i) / decay
        else:
            env = 1.0

        # White noise
        noise = (rng.random() * 2 - 1) * 0.25 * env

        # Rising sine sweep: freq rises linearly from 180 to 600 Hz
        freq      = 180 + (600 - 180) * (t / duration)
        # Phase accumulation for smooth sweep (avoid phase discontinuity)
        # integral of freq(t) dt = 180t + (600-180)/(2*duration) * t^2
        phase     = 2 * math.pi * (180 * t + (420 / (2 * duration)) * t * t)
        sweep     = math.sin(phase) * 0.20 * env

        sample = noise + sweep
        samples.append(max(-32768, min(32767, int(sample * 32767))))

    _write_wav(path, samples)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SoundPlayer:
    """
    Generates sound files once per session and plays them non-blockingly.

    Usage:
        player = SoundPlayer()
        player.play_spin()    # at wheel spin start
        player.play_win()     # at outcome reveal — win
        player.play_loss()    # at outcome reveal — loss
    """

    def __init__(self):
        self._tmp_dir   = Path(tempfile.mkdtemp(prefix='roulette_sounds_'))
        self._win_path  = str(self._tmp_dir / 'win.wav')
        self._loss_path = str(self._tmp_dir / 'loss.wav')
        self._spin_path = str(self._tmp_dir / 'spin.wav')
        self._player    = self._detect_player()

        # Generate all WAVs up front — no latency on first trial
        _build_win_wav(self._win_path)
        _build_loss_wav(self._loss_path)
        _build_spin_wav(self._spin_path)

        if self._player:
            print(f"[Sound] Initialised — player: {self._player}  "
                  f"sounds: {self._tmp_dir}")
        else:
            print("[Sound] No audio player found — sounds disabled.")

    # ------------------------------------------------------------------

    def play_win(self):
        """Play the win chime (non-blocking)."""
        self._play(self._win_path)

    def play_loss(self):
        """Play the loss thud (non-blocking)."""
        self._play(self._loss_path)

    def play_spin(self):
        """Play the wheel spin sound (non-blocking, ~1.8 s)."""
        self._play(self._spin_path)

    # ------------------------------------------------------------------

    def _detect_player(self):
        for cmd in ('afplay', 'aplay', 'ffplay'):
            try:
                subprocess.run([cmd, '--help'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return cmd
            except FileNotFoundError:
                continue
        return None

    def _play(self, path: str):
        """Launch playback in a daemon thread so Tkinter is never blocked."""
        if not self._player:
            return

        def _run():
            try:
                args = [self._player, path]
                if self._player == 'ffplay':
                    args = ['ffplay', '-nodisp', '-autoexit',
                            '-loglevel', 'quiet', path]
                subprocess.run(args,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except Exception:
                pass

        threading.Thread(target=_run, daemon=True).start()

