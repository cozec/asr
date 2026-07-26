# Forced Alignment with Gentle (Kaldi)

[Gentle](https://github.com/strob/gentle) is a forced aligner built on **Kaldi** — a
classical HMM/GMM+DNN pipeline with a pronunciation lexicon. It is the counterpart to
[`../wave2vec2_forced_alignment/`](../wave2vec2_forced_alignment/), which does the same
job with a CTC lattice over wav2vec2 emissions. Same task, two eras of technology.

```bash
cd gentle && python3 align.py examples/data/lucier.mp3 examples/data/lucier.txt \
    -o examples/lucier_alignment.json
cd .. && .venv/bin/python plot_alignment.py
```

## Result

Aligned Alvin Lucier's *I Am Sitting in a Room* — 96 seconds, 105 words:

```
aligned  : 101/105
span     : 6.77s - 96.06s
unaligned: ['my', 'speech', 'with', 'perhaps']

   6.77 -   6.98  I          [ay]
   6.98 -   7.18  am         [ae,m]
   7.22 -   7.98  sitting    [s,ih,t,ih,ng]
   7.98 -   8.10  in         [ih,n]
   8.10 -   8.20  a          [ah]
   8.20 -   8.74  room       [r,uw,m]
```

Gentle's two-pass strategy took 21 unaligned words down to 4. Word **and** phone
timings come out of a single pass, and the 6.8 s of leading silence is correctly
assigned to no word.

## The alignment, in three figures

### 1. Words over the spectrogram

![Waveform with shaded word regions above a spectrogram with hatched word boundaries, spanning 6 to 26 seconds. Shaded regions coincide with speech bursts and skip the silences between phrases.](plots/1_words_over_spectrogram.png)

The check that the alignment is real rather than merely self-consistent: shaded regions
land on the energy bursts, and the multi-second gaps between phrases are correctly left
empty. This recording is unusually demanding — Lucier speaks slowly with long pauses,
which is exactly where a lenient aligner earns its keep.

### 2. Phone-level detail

![Spectrogram of "I am sitting in a room" above a coloured bar strip, one bar per phone, labelled ay, ae, m, s, ih, t, ih, ng, and so on.](plots/2_phone_detail.png)

Gentle returns a phone sequence per word with a duration for each. Phone durations sum
**exactly** to the word duration, so the strip tiles without gaps.

One honest observation from this plot: the `/s/` in "sitting" is **490 ms**, which is far
too long for a fricative (typical is 50–150 ms). Gentle absorbed the short pause before
the word into its boundary phone. Word boundaries stay trustworthy; individual boundary
phones are less so. That is characteristic of forced alignment generally, not a Gentle
defect.

### 3. Coverage and duration distribution

![Top: green bars marking every aligned word across the full 96 seconds, clustered into phrases with clear gaps. Bottom: histogram of word durations peaking near 0.15s with a median of 0.28s and a tail out to 1.7s.](plots/3_coverage.png)

Coverage is even across the whole recording rather than degrading over time, which is
what you want from a lenient aligner on a long file. The duration histogram peaks around
0.15 s with a median of **0.28 s** and a tail to 1.7 s — Lucier's drawn-out delivery.

## Getting Gentle to run

The demo needs `ext/k3` and `ext/m3`, compiled Kaldi binaries that the repo does not
ship. Building Kaldi from source takes 1–3 hours and `install_kaldi.sh` passes
`--static-math=yes`, which fights Apple's Accelerate framework on ARM.

**The shortcut: the official `.dmg` already contains them.**

```bash
curl -L -o gentle.dmg https://github.com/strob/gentle/releases/download/0.11.0/gentle-0.11.0.dmg
hdiutil attach gentle.dmg -nobrowse -mountpoint /tmp/gentle_dmg
cp /tmp/gentle_dmg/gentle.app/Contents/Resources/ext/{k3,m3} gentle/ext/
hdiutil detach /tmp/gentle_dmg
```

Notes:

* The binaries are **x86_64 only** and run through **Rosetta 2** on Apple Silicon. They
  link nothing beyond Accelerate, libSystem and libc++, so there are no bundled-dylib
  problems. Without Rosetta, the source build becomes unavoidable.
* They are gitignored (~42 MB), so **a fresh clone must repeat this step**.
* Models in `gentle/exp/` and `ffmpeg` are also required; the `.dmg` bundles models too.
* `lowerquality/gentle` now 301-redirects to `strob/gentle` — same repo, renamed.

## Gentle vs. the wav2vec2 aligner

| | Gentle (here) | [wav2vec2](../wave2vec2_forced_alignment/) |
|---|---|---|
| Acoustic model | Kaldi TDNN chain | wav2vec2 CTC emissions |
| Units | phones, via a pronunciation lexicon | characters, no lexicon |
| Out-of-vocabulary words | needs a lexicon entry | handled, characters always spell |
| Unalignable words | flagged `not-found-in-audio` | forced onto the path regardless |
| Setup | Kaldi binaries + models (~250 MB) | `pip install torchaudio` |

The sharpest difference is the last two rows. Gentle can **refuse** — it marked 4 words
`not-found-in-audio` rather than inventing timings. The CTC aligner has no such escape
hatch: its trellis forces a monotonic path through every character, so a word the speaker
never said still receives a start and end time. Gentle's "lenient" claim is about
tolerating mismatch between transcript and audio, which is the common case with real
transcripts.

## Layout

```
plot_alignment.py           turns the JSON into the three figures
plots/                      the figures
results/lucier_alignment.json   the alignment output (tracked)
gentle/                     upstream strob/gentle -- NOT tracked here
```

`gentle/` is gitignored: it is a third-party repo with its own history and a Kaldi
submodule, so it is cloned rather than vendored. To reproduce from a fresh checkout:

```bash
git clone https://github.com/strob/gentle.git
# then install the Kaldi binaries as above, and run align.py
```
