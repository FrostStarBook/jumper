# Material for `jumper.breakdance`

The generated `breakdance.npz` is built from the repository's recorded high jump
and the opening six seconds of its crab-dance reference. It is an original
simulation choreography, not a motion capture or transcription of the linked
video.

Build or rebuild it from the repository root:

```bash
python tools/build_breakdance_reference.py
```

The builder time-stretches the jump, maps joints by name, aligns the dance at the
landing pose, computes support contacts with the current Jumper model, and runs
the same joint-limit, continuity, and forward-kinematics checks used at training
startup. It installs the file only after those checks pass. The optional
`support_contact` channel marks the four support sites in `LM`, `RM`, `LR`, `RR`
order; airborne frames are not treated as ground-contact poses.