import midi

from ..models import music


OUTPUT_PATH = "output.mid"


def write_midi(score: music.Score, path: str = OUTPUT_PATH) -> str:
    """
    Write a Score to a MIDI file.

    Args:
        score: The complete musical score to render.
        path: Where to write the file.

    Returns:
        A message describing what was written.
    """
    pattern = midi.Pattern(resolution=music.PPQ)

    meta_track = midi.Track()
    pattern.append(meta_track)
    meta_track.append(midi.SetTempoEvent(tick=0, bpm=score.tempo_bpm))
    ts_event = midi.TimeSignatureEvent(tick=0)
    ts_event.numerator = score.time_signature.numerator
    ts_event.denominator = score.time_signature.denominator
    ts_event.metronome = 24
    ts_event.thirtyseconds = 8
    meta_track.append(ts_event)
    meta_track.append(midi.EndOfTrackEvent(tick=0))

    total_notes = 0
    for track in score.tracks:
        t = midi.Track()
        pattern.append(t)

        if track.name:
            t.append(midi.TrackNameEvent(tick=0, text=track.name))
        t.append(midi.ProgramChangeEvent(tick=0, channel=track.channel, value=track.program))

        timeline: list[tuple[int, int, int, int]] = []
        for note in track.notes:
            timeline.append((note.start_tick, 1, note.pitch, note.velocity))
            timeline.append((note.start_tick + note.duration, 0, note.pitch, 0))

        timeline.sort(key=lambda e: (e[0], e[1]))

        # One channel can't sound the same pitch twice. When notes on a pitch overlap (two voices
        # sharing it), keep it sounding until the last of them ends instead of letting the first
        # note-off cut the others short; a later attack on a held pitch re-strikes it.
        sounding: dict[int, int] = {}
        struck_at: dict[int, int] = {}
        prev_tick = 0
        for abs_tick, kind, pitch, velocity in timeline:
            events = []
            if kind == 1:
                held = sounding.get(pitch, 0)
                if not held or struck_at[pitch] < abs_tick:
                    if held:
                        events.append(midi.NoteOffEvent(channel=track.channel, pitch=pitch, velocity=0))
                    events.append(midi.NoteOnEvent(channel=track.channel, pitch=pitch, velocity=velocity))
                    struck_at[pitch] = abs_tick
                sounding[pitch] = held + 1
            else:
                sounding[pitch] -= 1
                if not sounding[pitch]:
                    events.append(midi.NoteOffEvent(channel=track.channel, pitch=pitch, velocity=0))

            for ev in events:
                ev.tick = abs_tick - prev_tick
                t.append(ev)
                prev_tick = abs_tick

        t.append(midi.EndOfTrackEvent(tick=0))
        total_notes += len(track.notes)

    midi.write_midifile(path, pattern)

    return (
        f"Wrote {path}: {len(score.tracks)} track(s), {total_notes} note(s), "
        f"{score.tempo_bpm:g} BPM, {score.time_signature.numerator}/{score.time_signature.denominator}."
    )
