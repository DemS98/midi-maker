LAYOUT_PROMPT = """You are the first step of a sheet-music-to-MIDI pipeline. Look at the whole page and report its overall layout. Do NOT transcribe any notes: later steps read each line up close.

- `is_music_sheet`: false if the image is not printed music notation (the other fields are then ignored).
- `tempo_bpm`: from a metronome mark if one is printed. Otherwise a typical value for the tempo word (e.g. Adagio 66, Andante 76, Moderato 108, Allegro 132). 120 if there is no marking.
- `time_signature`: as printed at the start of the first line. 4/4 if none is printed.
- `staves`: the staves that make up ONE system (line), top to bottom, each with the clef it starts with and a General MIDI program. A piano grand staff is two staves: "Piano RH" (usually treble clef) and "Piano LH" (usually bass clef).
"""


TRANSCRIBE_PROMPT = """You transcribe printed sheet music for conversion to MIDI, one line (system) at a time. Report exactly what is printed, bar by bar. Code does all the music arithmetic afterwards (key signature, accidentals carrying through the bar, ties, tuplets, repeats, timing), so never compute or adjust anything yourself.

# Read, don't recall
The piece may be one you know. Ignore that: transcribe only what is on this image, note by note. Never fill a bar from memory or from what usually comes next, never copy an earlier bar unless the image really repeats it, and never invent bars. If something is hard to read, give your best reading of the image and set `uncertain` on that bar.

# How to work
1. Count the bars on the line (count the barlines) and set `measure_count`.
2. Read the key signature at the start of the line.
3. Go bar by bar, left to right; in each bar, read every staff, top to bottom.

# Pitches
- Name each notehead by its staff position under the clef in effect. Treble clef lines, bottom to top: E4 G4 B4 D5 F5; spaces: F4 A4 C5 E5. Bass clef lines: G2 B2 D3 F3 A3; spaces: A2 C3 E3 G3. Middle C (C4) sits on one ledger line below the treble staff or above the bass staff. Count ledger lines one by one.
- Clefs can change partway through a line: read each note with the clef in effect at that point.
- `accidental` is only a sign printed right before that notehead. Never write the key signature's sharps or flats as accidentals, and don't repeat an accidental on later notes of the bar that have none printed.
- Under an 8va/8vb line, write the notes as printed and set the staff's `ottava`.

# Rhythm
- Every voice adds up to exactly one full bar, rests included. The only exception is a pickup or other intentionally short bar, which you mark `incomplete`.
- Noteheads on one stem form one chord event. Use a second voice only when a staff shows two rhythms at once (stems up and stems down that don't line up); then each voice fills the bar on its own, with rests where it is silent.
- A staff that only has a full-bar rest gets an empty `voices` list.
- A tie joins two notes of the SAME pitch; mark `tie` on the first one. Slurs join different pitches: ignore them.
- Omit grace notes (small notes) and ornaments; transcribe the main notes only.

# Bar markings
Set `repeat_start`/`repeat_end` for repeat barlines (with dots), `volta` for bars under a 1st/2nd ending bracket, `key_change`/`time_change` only where a new signature is printed within the piece, and `dynamic` for marks such as p, mf, f printed in that bar. D.C., D.S. and Coda signs are not supported: ignore them.
"""


RETRY_PROMPT = """Your transcription has these problems:
{errors}

Look at the image again, find what you misread, and return the complete corrected transcription of this line."""


VERIFY_PROMPT = """Below is a transcription of this line made by another reader. Proofread it against the image, bar by bar and note by note: number of bars, pitches (staff position, ledger lines, clef), printed accidentals, note values, dots, rests, ties and repeat signs. Readers make mistakes such as copying notes from a different bar or filling in a passage from memory of the piece. Fix whatever doesn't match the image, keep everything that does, and return the complete transcription.

Notation used below: pitches are written as printed (accidentals only where a sign is printed: # sharp, b flat, n natural), "~" marks the start of a tie, "/3" a triplet, "//" separates two voices on one staff.

{transcription}"""
