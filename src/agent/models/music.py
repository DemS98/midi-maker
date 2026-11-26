from pydantic import BaseModel, Field, field_validator

PPQ = 480


class Note(BaseModel):
    """One pitched note. Use absolute MIDI pitch (middle C = 60)."""

    pitch: int = Field(
        ge=0, le=127,
        description="MIDI pitch number 0-127. Middle C = 60. C4 = 60, A4 = 69.",
    )
    start_tick: int = Field(
        ge=0,
        description=(
            "Absolute tick offset from the start of the piece. "
            f"Resolution is {PPQ} PPQ, so a quarter note is {PPQ} ticks. "
            "Notes that begin on the same beat (i.e. a chord) MUST share the same start_tick."
        ),
    )
    duration: int = Field(
        gt=0,
        description=(
            f"Length of the note in ticks (PPQ={PPQ}). "
            f"Whole={PPQ * 4}, half={PPQ * 2}, quarter={PPQ}, eighth={PPQ // 2}, sixteenth={PPQ // 4}."
        ),
    )
    velocity: int = Field(
        ge=1, le=127, default=80,
        description="How hard the note is struck (1-127). Use ~80 for mf unless dynamics say otherwise.",
    )


class TimeSignature(BaseModel):
    """ Time signature for the piece of music. """
    numerator: int = Field(ge=1, le=64, default=4, description="Beats per bar (top number).")
    denominator: int = Field(
        default=4,
        description="Beat unit (bottom number). Must be a power of 2: 1, 2, 4, 8, 16, 32.",
    )

    @field_validator("denominator")
    @classmethod
    def _power_of_two(cls, value: int) -> int:
        # MIDI stores the denominator as a power of 2; anything else would be silently rounded.
        if value not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"denominator must be a power of 2 (1-32), got {value}")
        return value


class Track(BaseModel):
    """A single instrument line."""

    name: str | None = Field(default=None, description="Human-readable track name (e.g. 'Piano RH').")
    program: int = Field(
        ge=0, le=127, default=0,
        description=(
            "General MIDI program (instrument) number 0-127. "
            "0=Acoustic Grand Piano, 24=Nylon Guitar, 40=Violin, 56=Trumpet, 73=Flute."
        ),
    )
    channel: int = Field(
        ge=0, le=15, default=0,
        description="MIDI channel 0-15. Use a different channel per track. Channel 9 is reserved for percussion.",
    )
    notes: list[Note] = Field(description="Notes in the track. Order does not matter; ticks are absolute.")


class Score(BaseModel):
    """A complete piece of music — global metadata plus one or more tracks."""

    tempo_bpm: float = Field(
        gt=0, default=120.0,
        description="Tempo in beats per minute. If the sheet has no marking, default to 120.",
    )
    time_signature: TimeSignature = Field(default_factory=TimeSignature)
    tracks: list[Track] = Field(min_length=1, description="At least one track.")
