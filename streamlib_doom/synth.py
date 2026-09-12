"""Renders a MUS score the way a 1993 Sound Blaster did: two-operator FM
through the GENMIDI patches in the WAD, with the sound effects mixed in at the
moments the demo script fires them.

Not a cycle-exact OPL2. Same operator layout, same waveforms, same envelope
shape and the same instrument bytes, approximated in floating point — which is
enough for "At Doom's Gate" to sound like the Adlib version everyone remembers.
"""
from __future__ import annotations

import math

import numpy

from .wad import MUS_TICKS_PER_SECOND, Wad

SAMPLE_RATE = 48_000
PERCUSSION_CHANNEL = 15


def _rate_seconds(rate: int) -> float:
    """An OPL rate nibble as a time constant: 15 is instant, 0 is several seconds."""
    return 3.0 * 2.0 ** (-rate * 0.8)


def _waveform(phase: numpy.ndarray, select: int) -> numpy.ndarray:
    s = numpy.sin(phase)
    if select == 1:
        return numpy.maximum(s, 0.0)
    if select == 2:
        return numpy.abs(s)
    if select == 3:
        quarter = (phase % math.pi) < (math.pi / 2)
        return numpy.abs(s) * quarter
    return s


def _envelope(n: int, op: dict, held_samples: int) -> numpy.ndarray:
    """Attack to 1, decay to the sustain level, hold while the key is down (or
    keep falling for a percussive operator), then release."""
    t = numpy.arange(n, dtype=numpy.float32) / SAMPLE_RATE
    attack = max(_rate_seconds(op["attack"]), 0.0005)
    decay = max(_rate_seconds(op["decay"]), 0.002)
    release = max(_rate_seconds(op["release"]), 0.002)
    sustain_gain = 2.0 ** (-op["sustain"] * 0.5)
    held = held_samples / SAMPLE_RATE
    env = numpy.empty(n, dtype=numpy.float32)
    rising = t < attack
    env[rising] = t[rising] / attack
    after = ~rising
    ta = t[after] - attack
    decayed = sustain_gain + (1.0 - sustain_gain) * numpy.exp(-ta / decay)
    if not op["sustaining"]:
        decayed = decayed * numpy.exp(-numpy.maximum(ta - decay * 3.0, 0.0) / (release * 2.0))
    env[after] = decayed
    releasing = t > held
    env[releasing] *= numpy.exp(-(t[releasing] - held) / release)
    return env


def _fm_note(frequency: float, voice: dict, velocity: float, held_samples: int) -> numpy.ndarray:
    modulator, carrier = voice["modulator"], voice["carrier"]
    release_tail = int(SAMPLE_RATE * min(1.5, _rate_seconds(carrier["release"]) * 4.0 + 0.05))
    n = held_samples + release_tail
    t = numpy.arange(n, dtype=numpy.float32) / SAMPLE_RATE
    mod_multiplier = max(modulator["multiplier"], 0.5)
    car_multiplier = max(carrier["multiplier"], 0.5)
    mod_index = 3.5 * 10.0 ** (-0.75 * modulator["level"] / 20.0)
    mod_phase = 2.0 * math.pi * frequency * mod_multiplier * t
    feedback = voice["feedback"] / 7.0 * 1.2
    if feedback > 0.0:
        mod_phase = mod_phase + feedback * numpy.sin(mod_phase)
    modulation = _envelope(n, modulator, held_samples) * _waveform(mod_phase, modulator["wave"]) * mod_index
    car_phase = 2.0 * math.pi * frequency * car_multiplier * t
    if voice["additive"]:
        signal = _waveform(car_phase, carrier["wave"]) + modulation / max(mod_index, 1e-6) * 0.5
    else:
        signal = _waveform(car_phase + modulation, carrier["wave"])
    gain = 10.0 ** (-0.75 * carrier["level"] / 20.0)
    return (signal * _envelope(n, carrier, held_samples) * gain * velocity).astype(numpy.float32)


def render_score(wad: Wad, lump_name: str, seconds: float) -> numpy.ndarray:
    """(n, 2) float32 stereo at 48 kHz: the first `seconds` of the score."""
    events = wad.mus_events(lump_name)
    instruments = wad.genmidi()
    total = int(seconds * SAMPLE_RATE)
    mix = numpy.zeros((total + SAMPLE_RATE * 2, 2), dtype=numpy.float32)
    program = {c: 0 for c in range(16)}
    volume = {c: 1.0 for c in range(16)}
    pan = {c: 0.5 for c in range(16)}
    last_velocity = {c: 100 for c in range(16)}
    active: dict[tuple[int, int], tuple[int, int, float]] = {}  # (channel, note) -> (start sample, program, velocity)

    def finish(channel: int, note: int, end_sample: int) -> None:
        key = (channel, note)
        if key not in active:
            return
        start, prog, velocity = active.pop(key)
        if start >= total:
            return
        held = max(end_sample - start, int(SAMPLE_RATE * 0.03))
        if channel == PERCUSSION_CHANNEL:
            if not 35 <= note <= 81:
                return
            instrument = instruments[128 + note - 35]
            pitch_note = instrument["fixed_note"] if instrument["fixed_note"] else note
        else:
            instrument = instruments[prog]
            pitch_note = note
        signal = numpy.zeros(0, dtype=numpy.float32)
        voice_count = 2 if instrument["flags"] & 4 else 1
        for v in range(voice_count):
            voice = instrument["voices"][v]
            frequency = 440.0 * 2.0 ** ((pitch_note + voice["note_offset"] - 69) / 12.0)
            if frequency < 20.0 or frequency > 12000.0:
                continue
            rendered = _fm_note(frequency, voice, velocity * volume[channel], held)
            if len(rendered) > len(signal):
                rendered[: len(signal)] += signal
                signal = rendered
            else:
                signal[: len(rendered)] += rendered
        if len(signal) == 0:
            return
        end = min(start + len(signal), len(mix))
        left, right = math.cos(pan[channel] * math.pi / 2), math.sin(pan[channel] * math.pi / 2)
        mix[start:end, 0] += signal[: end - start] * left
        mix[start:end, 1] += signal[: end - start] * right

    for tick, channel, kind, a, b in events:
        sample = int(tick / MUS_TICKS_PER_SECOND * SAMPLE_RATE)
        if sample > total:
            break
        if kind == 4:
            if a == 0:
                program[channel] = b
            elif a == 3:
                volume[channel] = b / 127.0
            elif a == 4:
                pan[channel] = b / 127.0
        elif kind == 1:
            if b >= 0:
                last_velocity[channel] = b
            finish(channel, a, sample)
            active[(channel, a)] = (sample, program[channel], last_velocity[channel] / 127.0)
        elif kind == 0:
            finish(channel, a, sample)
    for (channel, note) in list(active):
        finish(channel, note, total)
    mix = mix[:total]
    peak = float(numpy.abs(mix).max()) or 1.0
    return mix / peak * 0.5


def resample_effect(rate: int, mono: numpy.ndarray) -> numpy.ndarray:
    """An 11 kHz effect brought up to 48 kHz by linear interpolation."""
    n = int(len(mono) * SAMPLE_RATE / rate)
    positions = numpy.arange(n, dtype=numpy.float64) * rate / SAMPLE_RATE
    return numpy.interp(positions, numpy.arange(len(mono)), mono).astype(numpy.float32)


def mix_effect(track: numpy.ndarray, effect: numpy.ndarray, at_seconds: float, gain: float = 0.8, pan: float = 0.5) -> None:
    start = int(at_seconds * SAMPLE_RATE)
    if start >= len(track):
        return
    end = min(len(track), start + len(effect))
    left, right = math.cos(pan * math.pi / 2), math.sin(pan * math.pi / 2)
    track[start:end, 0] += effect[: end - start] * gain * left
    track[start:end, 1] += effect[: end - start] * gain * right
