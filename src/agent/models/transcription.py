"""
What the model reports when it reads a page: notation exactly as printed, bar by bar.

The model never computes MIDI numbers or ticks. Key signatures, accidentals, ties, tuplets
and repeats are resolved in code (see `notation.py`). Every field is required and has no
range constraints so the schemas work with OpenAI's strict structured outputs; ranges are
expressed as Literals or checked in code.
"""
from typing import Literal

from pydantic import BaseModel, Field


Step = Literal["C", "D", "E", "F", "G", "A", "B"]
Octave = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]
Accidental = Literal["sharp", "flat", "natural", "double-sharp", "double-flat"]
NoteValue = Literal["w", "h", "q", "8", "16", "32", "64"]
KeyFifths = Literal[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
Clef = Literal["treble", "bass", "alto", "tenor", "treble-8vb", "percussion"]
Dynamic = Literal["ppp", "pp", "p", "mp", "mf", "f", "ff", "fff"]


class Meter(BaseModel):
    """A time signature."""

    numerator: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = Field(
        description="Top number: beats per bar."
    )
    denominator: Literal[1, 2, 4, 8, 16, 32] = Field(description="Bottom number: the beat unit.")


class WrittenPitch(BaseModel):
    """One notehead, as printed."""

    step: Step = Field(description="Letter name of the notehead's staff position, read with the clef in effect.")
    octave: Octave = Field(
        description=(
            "Octave of the staff position. Middle C = C4. Treble clef: bottom line E4, middle line B4. "
            "Bass clef: bottom line G2, middle line D3, top line A3. Count ledger lines one by one."
        ),
    )
    accidental: Accidental | None = Field(
        description=(
            "Accidental printed directly in front of THIS notehead, else null. Do not apply the key "
            "signature or accidentals from earlier notes yourself."
        ),
    )
    tie: bool = Field(
        description=(
            "True if a tie starts at this notehead: a curve to the SAME pitch in the next note of this voice. "
            "A slur between different pitches is not a tie."
        ),
    )


class Event(BaseModel):
    """A single note, a chord (noteheads on one stem) or a rest."""

    pitches: list[WrittenPitch] = Field(description="Noteheads struck together. Empty list for a rest.")
    duration: NoteValue = Field(
        description="Written note value: w=whole, h=half, q=quarter, 8=eighth, 16=sixteenth, 32, 64.",
    )
    dots: Literal[0, 1, 2] = Field(description="Number of augmentation dots after the notehead.")
    tuplet: Literal[2, 3, 4, 5, 6, 7, 9] | None = Field(
        description="Number on the tuplet bracket this note belongs to (3 = triplet), else null.",
    )


class Voice(BaseModel):
    """One rhythmic layer of a staff."""

    events: list[Event] = Field(description="Notes, chords and rests in time order. Together they fill the bar.")


class StaffMeasure(BaseModel):
    """What one staff holds in one bar."""

    voices: list[Voice] = Field(
        description=(
            "Usually exactly one. Use two only when the staff shows two independent rhythms at once "
            "(stems up and stems down). Empty list if the staff only has a full-bar rest."
        ),
    )
    ottava: Literal[-2, -1, 0, 1, 2] = Field(
        description="Octave line over this staff in this bar: 1 = 8va, 2 = 15ma, -1 = 8vb, -2 = 15mb, 0 = none.",
    )


class Measure(BaseModel):
    """One bar, across all staves."""

    number: int = Field(description="Position of the bar on this line: 1, 2, 3...")
    staves: list[StaffMeasure] = Field(description="Exactly one entry per staff, top to bottom.")
    repeat_start: bool = Field(description="A start-repeat barline (|: with two dots) opens this bar.")
    repeat_end: bool = Field(description="An end-repeat barline (:| with two dots) closes this bar.")
    volta: Literal[1, 2] | None = Field(
        description="1 or 2 if the bar is under a 1st/2nd ending bracket, else null.",
    )
    key_change: KeyFifths | None = Field(
        description="New key signature printed inside this line at the start of this bar, else null.",
    )
    time_change: Meter | None = Field(
        description="New time signature printed at the start of this bar, else null.",
    )
    dynamic: Dynamic | None = Field(description="Dynamic mark printed in this bar, else null.")
    incomplete: bool = Field(
        description=(
            "True only for a bar that is intentionally shorter than the time signature: "
            "a pickup (anacrusis) bar, or a bar split around a repeat sign."
        ),
    )
    uncertain: bool = Field(description="True if part of this bar was hard to read and you had to guess.")


class SystemTranscription(BaseModel):
    """One line (system) of the score, bar by bar."""

    key_fifths: KeyFifths = Field(
        description=(
            "Key signature printed at the start of this line: number of sharps (positive) or flats "
            "(negative), 0 if none."
        ),
    )
    measure_count: int = Field(description="Number of bars on this line. Count the barlines before transcribing.")
    measures: list[Measure] = Field(description="Every bar on this line, left to right.")


class StaffInfo(BaseModel):
    """One staff of a system."""

    name: str = Field(description="Short label, e.g. 'Piano RH', 'Piano LH', 'Violin'.")
    clef: Clef = Field(description="Clef at the start of the piece.")
    program: int = Field(
        description=(
            "General MIDI program 0-127 for the instrument: 0 = piano, 24 = nylon guitar, "
            "40 = violin, 56 = trumpet, 73 = flute."
        ),
    )


class PageLayout(BaseModel):
    """The page's overall layout, read before any notes are transcribed."""

    is_music_sheet: bool = Field(description="False if the image is not printed music notation.")
    tempo_bpm: float = Field(
        description=(
            "Quarter-note tempo. Use a metronome mark if printed (e.g. ♩ = 96). For a tempo word only "
            "(Andante, Moderato, Allegro...) use a typical value for it. 120 if there is no marking."
        ),
    )
    time_signature: Meter = Field(description="Time signature at the start of the piece. 4/4 if none is printed.")
    staves: list[StaffInfo] = Field(
        description="The staves of ONE system, top to bottom. A piano grand staff is two staves (RH, LH).",
    )
