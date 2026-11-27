"""
Turn a bar-by-bar transcription into a playable Score, and check transcriptions for mistakes.

Everything that is arithmetic rather than reading happens here (key signatures, accidentals,
ties, tuplets, repeats, ticks), so the model only has to report what is printed.
"""
from dataclasses import dataclass
from fractions import Fraction

from .models import music
from .models.transcription import Event, Measure, Meter, PageLayout, SystemTranscription, Voice


STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = "FCGDAEB"
ACCIDENTAL_ALTER = {"sharp": 1, "flat": -1, "natural": 0, "double-sharp": 2, "double-flat": -2}
ACCIDENTAL_SIGN = {"sharp": "#", "flat": "b", "natural": "n", "double-sharp": "##", "double-flat": "bb"}

# Length of each written note value, in quarter notes.
NOTE_VALUES = {
    "w": Fraction(4), "h": Fraction(2), "q": Fraction(1), "8": Fraction(1, 2),
    "16": Fraction(1, 4), "32": Fraction(1, 8), "64": Fraction(1, 16),
}
# n notes played in the time of m: duplets and quadruplets borrow from compound meters.
TUPLET_RATIOS = {
    2: Fraction(3, 2), 3: Fraction(2, 3), 4: Fraction(3, 4), 5: Fraction(4, 5),
    6: Fraction(4, 6), 7: Fraction(4, 7), 9: Fraction(8, 9),
}
DYNAMIC_VELOCITY = {"ppp": 20, "pp": 30, "p": 45, "mp": 60, "mf": 80, "f": 100, "ff": 115, "fff": 127}
DEFAULT_VELOCITY = 80


def key_alter(key_fifths: int, step: str) -> int:
    """Semitone shift the key signature gives to a letter name."""
    if key_fifths > 0 and step in SHARP_ORDER[:key_fifths]:
        return 1
    if key_fifths < 0 and step in SHARP_ORDER[::-1][:-key_fifths]:
        return -1
    return 0


def event_length(event: Event) -> Fraction:
    """Length of an event in quarter notes."""
    length = NOTE_VALUES[event.duration] * (2 - Fraction(1, 2 ** event.dots))
    if event.tuplet:
        length *= TUPLET_RATIOS[event.tuplet]
    return length


def bar_length(meter: Meter) -> Fraction:
    return Fraction(4 * meter.numerator, meter.denominator)


def is_bar_rest(voice: Voice) -> bool:
    """A lone rest fills the whole bar whatever its written value (a whole rest in 3/4, say)."""
    return len(voice.events) == 1 and not voice.events[0].pitches


def voice_length(voice: Voice) -> Fraction:
    return sum((event_length(event) for event in voice.events), Fraction(0))


def _quarters(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{float(value):g}"


# --- Checking ---------------------------------------------------------------------------------

def check_system(
    system: SystemTranscription,
    meter: Meter,
    staff_names: list[str],
    expected_measures: int | None = None,
) -> list[str]:
    """
    Find mistakes in a transcribed line that can be detected without looking at the image.

    Args:
        system: The transcription to check.
        meter: Time signature in effect at the start of the line.
        staff_names: Names of the staves in each system, top to bottom.
        expected_measures: Bar count measured from the barlines in the image, if it should be enforced.

    Returns:
        One readable message per problem; empty if none were found.
    """
    errors = []
    if system.measure_count != len(system.measures):
        errors.append(
            f"measure_count is {system.measure_count} but {len(system.measures)} bars were transcribed."
        )
    if expected_measures is not None and len(system.measures) != expected_measures:
        errors.append(
            f"Counting the barlines in the image gives {expected_measures} bars on this line but you "
            f"transcribed {len(system.measures)}. Recount; if you are sure of your count, keep it."
        )

    for index, measure in enumerate(system.measures, start=1):
        if measure.time_change:
            meter = measure.time_change
        if measure.number != index:
            errors.append(f"Bar {index} is numbered {measure.number}; number the bars 1, 2, 3... in order.")
        if len(measure.staves) != len(staff_names):
            errors.append(
                f"Bar {index} has {len(measure.staves)} staves but each line has {len(staff_names)} "
                f"({', '.join(staff_names)}): give exactly one entry per staff, top to bottom."
            )
            continue

        expected = bar_length(meter)
        for staff_name, staff in zip(staff_names, measure.staves):
            for voice_number, voice in enumerate(staff.voices, start=1):
                if is_bar_rest(voice):
                    continue
                length = voice_length(voice)
                if length > expected or (length < expected and not measure.incomplete):
                    errors.append(
                        f"Bar {index}, {staff_name}, voice {voice_number}: the note and rest values add up "
                        f"to {_quarters(length)} quarter notes but a {meter.numerator}/{meter.denominator} "
                        f"bar holds {_quarters(expected)}. Re-read the rhythms: dots, rests, beams, tuplets."
                    )
    return errors


# --- Describing -------------------------------------------------------------------------------

def _describe_event(event: Event) -> str:
    duration = event.duration + "." * event.dots + (f"/{event.tuplet}" if event.tuplet else "")
    if not event.pitches:
        return f"rest {duration}"
    pitches = "+".join(
        f"{p.step}{ACCIDENTAL_SIGN[p.accidental] if p.accidental else ''}{p.octave}{'~' if p.tie else ''}"
        for p in event.pitches
    )
    return f"{pitches} {duration}"


def describe_system(system: SystemTranscription, staff_names: list[str]) -> str:
    """
    Render a transcription as compact text, for the model to review.

    Pitches are shown as written (accidentals only where printed: # b n), "~" marks a tie,
    "/3" a triplet, "//" separates voices.
    """
    lines = [f"Key signature: {system.key_fifths:+d}. {len(system.measures)} bars."]
    for measure in system.measures:
        marks = [
            "|:" if measure.repeat_start else "",
            ":|" if measure.repeat_end else "",
            f"ending {measure.volta}" if measure.volta else "",
            f"key {measure.key_change:+d}" if measure.key_change is not None else "",
            f"time {measure.time_change.numerator}/{measure.time_change.denominator}"
            if measure.time_change else "",
            measure.dynamic or "",
            "pickup/incomplete" if measure.incomplete else "",
            "uncertain" if measure.uncertain else "",
        ]
        header = ", ".join(mark for mark in marks if mark)
        lines.append(f"Bar {measure.number}" + (f" ({header})" if header else ""))
        for index, staff in enumerate(measure.staves):
            name = staff_names[index] if index < len(staff_names) else f"Staff {index + 1}"
            voices = " // ".join(", ".join(_describe_event(e) for e in voice.events) for voice in staff.voices)
            ottava = f" [ottava {staff.ottava:+d}]" if staff.ottava else ""
            lines.append(f"  {name}{ottava}: {voices or 'full-bar rest'}")
    return "\n".join(lines)


# --- Converting -------------------------------------------------------------------------------

def play_order(measures: list[Measure]) -> list[int]:
    """Indexes of the written bars in the order they are played, with repeats and endings expanded."""
    order = []
    index = start = 0
    pass_number = 1
    in_endings = False
    repeated: set[int] = set()

    while index < len(measures):
        measure = measures[index]
        if in_endings and measure.volta is None:
            # Past the 1st/2nd endings: a new section begins.
            in_endings = False
            start, pass_number = index, 1
        if measure.repeat_start and index != start:
            start, pass_number = index, 1
        if measure.volta is not None:
            in_endings = True
            if measure.volta != pass_number:
                index += 1
                continue

        order.append(index)
        if measure.repeat_end:
            if index not in repeated:
                repeated.add(index)
                index, pass_number = start, pass_number + 1
                in_endings = False
                continue
            if measure.volta is None:
                start, pass_number = index + 1, 1
        index += 1

    return order


@dataclass
class _Sounding:
    pitch: int
    start: Fraction
    end: Fraction
    velocity: int


def to_score(layout: PageLayout, systems: list[SystemTranscription]) -> tuple[music.Score, list[str]]:
    """
    Resolve the transcribed lines of all pages into absolute MIDI notes.

    Args:
        layout: The page overview (tempo, starting time signature, staves).
        systems: Transcribed lines in reading order, across all pages.

    Returns:
        The score, and warnings about anything that had to be skipped.
    """
    warnings: list[str] = []
    staves = layout.staves[:15]
    if len(layout.staves) > len(staves):
        warnings.append(f"Only the first {len(staves)} staves fit on MIDI channels; the rest were dropped.")

    # Flatten the lines into one list of written bars, with the key and meter in effect for each.
    measures: list[Measure] = []
    keys: list[int] = []
    meters: list[Meter] = []
    meter = layout.time_signature
    for system in systems:
        key = system.key_fifths
        for measure in system.measures:
            key = measure.key_change if measure.key_change is not None else key
            meter = measure.time_change or meter
            measures.append(measure)
            keys.append(key)
            meters.append(meter)

    notes: list[list[_Sounding]] = [[] for _ in staves]
    # Per (staff, voice): ties left open by the previous event, keyed by (step, octave).
    open_ties: dict[tuple[int, int], dict[tuple[str, int], tuple[_Sounding, int]]] = {}
    velocity = DEFAULT_VELOCITY
    position = Fraction(0)

    for index in play_order(measures):
        measure = measures[index]
        if measure.dynamic:
            velocity = DYNAMIC_VELOCITY[measure.dynamic]

        length = bar_length(meters[index])
        if measure.incomplete:
            played = [
                voice_length(voice)
                for staff in measure.staves for voice in staff.voices if not is_bar_rest(voice)
            ]
            length = min(max(played, default=length), length)

        for staff_index, staff in enumerate(measure.staves[:len(staves)]):
            # Accidentals hold for the rest of the bar on the whole staff, so walk all voices in time order.
            timed = []
            for voice_index, voice in enumerate(staff.voices):
                onset = position
                for event in voice.events:
                    timed.append((onset, voice_index, event))
                    onset += event_length(event)
            timed.sort(key=lambda item: (item[0], item[1]))

            bar_accidentals: dict[tuple[str, int], int] = {}
            for onset, voice_index, event in timed:
                ties = open_ties.get((staff_index, voice_index), {})
                still_open = {}
                end = onset + event_length(event)

                for written in event.pitches:
                    spot = (written.step, written.octave)
                    if written.accidental:
                        alter = ACCIDENTAL_ALTER[written.accidental]
                        bar_accidentals[spot] = alter
                    elif spot in ties:
                        alter = ties[spot][1]
                    elif spot in bar_accidentals:
                        alter = bar_accidentals[spot]
                    else:
                        alter = key_alter(keys[index], written.step)

                    if spot in ties:
                        sounding = ties[spot][0]
                        sounding.end = end
                    else:
                        octave = written.octave + staff.ottava
                        pitch = 12 * (octave + 1) + STEP_SEMITONES[written.step] + alter
                        if not 0 <= pitch <= 127:
                            warnings.append(f"Skipped an out-of-range note ({written.step}{written.octave}).")
                            continue
                        sounding = _Sounding(pitch, onset, end, velocity)
                        notes[staff_index].append(sounding)

                    if written.tie:
                        still_open[spot] = (sounding, alter)

                open_ties[(staff_index, voice_index)] = still_open

        position += length

    def ticks(quarters: Fraction) -> int:
        return round(quarters * music.PPQ)

    tracks = [
        music.Track(
            name=info.name,
            program=min(max(info.program, 0), 127),
            channel=index if index < 9 else index + 1,  # channel 9 is percussion
            notes=[
                music.Note(
                    pitch=n.pitch,
                    start_tick=ticks(n.start),
                    duration=max(1, ticks(n.end) - ticks(n.start)),
                    velocity=n.velocity,
                )
                for n in sounding_notes
            ],
        )
        for index, (info, sounding_notes) in enumerate(zip(staves, notes))
    ]

    score = music.Score(
        tempo_bpm=layout.tempo_bpm if layout.tempo_bpm > 0 else 120.0,
        time_signature=music.TimeSignature(
            numerator=layout.time_signature.numerator,
            denominator=layout.time_signature.denominator,
        ),
        tracks=tracks,
    )
    return score, warnings
