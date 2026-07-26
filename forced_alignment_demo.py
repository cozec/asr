"""
Forced Alignment Demo
Version: 1.0.0
"""

import torch
import torchaudio
import matplotlib.pyplot as plt
from dataclasses import dataclass
from gtts import gTTS
import os

__version__ = "1.0.0"

@dataclass
class Point:
    """Represents a point in the alignment path"""
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    """Represents a segment of aligned text"""
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def get_trellis(emission, tokens, blank_id=0):
    """Generate alignment trellis matrix.

    Two transitions are allowed per frame: stay on the current token, which emits a
    blank, or advance to the next token, which emits that token. Each branch must be
    scored with the emission it actually produces. Scoring "stay" with the token's own
    probability makes the max degenerate into picking the better predecessor, models no
    blanks at all, and leaves backtrack recovering a path this trellis never scored.
    """
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Initialize trellis with infinity
    trellis = torch.full((num_frame, num_tokens), -float("inf"))
    trellis[0, 0] = emission[0, tokens[0]]

    for t in range(1, num_frame):
        trellis[t, 0] = trellis[t-1, 0] + emission[t, blank_id]
        for j in range(1, num_tokens):
            trellis[t, j] = torch.max(
                trellis[t-1, j] + emission[t, blank_id],        # stay -> emit blank
                trellis[t-1, j-1] + emission[t, tokens[j]]      # advance -> emit token
            )

    return trellis

def backtrack(trellis, emission, tokens, blank_id=0):
    """Backtrack to find the optimal path.

    Scores the two transitions exactly as get_trellis did, so the recovered path is the
    Viterbi path of that trellis. The loop guards on t > 0 because emission[t-1] at t=0
    silently wraps to the *last* frame -- a wrong value rather than an error.
    """
    t, j = trellis.size(0) - 1, trellis.size(1) - 1
    path = [Point(j, t, emission[t, blank_id].exp().item())]

    while j > 0 and t > 0:
        # Score for staying at the same token (emits a blank)
        stay = emission[t-1, blank_id]
        # Score for changing to the next token (emits the token)
        change = emission[t-1, tokens[j]]

        stayed = trellis[t-1, j] + stay
        changed = trellis[t-1, j-1] + change

        t -= 1
        if changed > stayed:
            j -= 1
        path.append(Point(j, t, (change if changed > stayed else stay).exp().item()))

    # Any frames left over belong to the first token, held across leading silence.
    while t > 0:
        t -= 1
        path.append(Point(j, t, emission[t, blank_id].exp().item()))

    return path[::-1]

def merge_repeats(path, transcript):
    """Merge repeated tokens into segments"""
    i1, i2 = 0, 0
    segments = []
    
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2-1].time_index + 1,
                score
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    """Merge segments into words"""
    words = []
    i1, i2 = 0, 0
    
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(
                    Segment(word, segments[i1].start, segments[i2-1].end, score)
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def visualize_emission(emission):
    """Visualize emission matrix"""
    plt.figure(figsize=(10, 5))
    plt.imshow(emission.T, origin="lower")
    plt.colorbar(location="bottom")
    plt.title("Frame-wise class probability")
    plt.xlabel("Time")
    plt.ylabel("Labels")
    plt.tight_layout()
    plt.show()

def visualize_trellis_with_path(trellis, path):
    """Visualize trellis matrix with alignment path"""
    trellis_with_path = trellis.clone()
    for p in path:
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("Alignment path")
    plt.xlabel("Time")
    plt.ylabel("Labels")
    plt.tight_layout()
    plt.show()

def visualize_alignments(trellis, segments, word_segments, waveform, sample_rate):
    """Visualize alignments with word segments"""
    trellis_with_path = trellis.clone()
    for i, seg in enumerate(segments):
        if seg.label != "|":
            trellis_with_path[seg.start:seg.end, i] = float("nan")
    
    plt.figure(figsize=(10, 5))
    
    # Plot trellis with segments
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("Alignment path with word segments")
    
    # Add word boundaries
    for word in word_segments:
        plt.axvspan(word.start - 0.5, word.end - 0.5, alpha=0.1, color="white")
    
    # Add labels with scores
    for i, seg in enumerate(segments):
        if seg.label != "|":
            plt.annotate(f"{seg.label}\n({seg.score:.2f})", 
                        (seg.start, i),
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='left',
                        va='bottom')
    
    plt.tight_layout()
    plt.show()

def generate_demo_audio(text, output_path="demo_audio.wav", target_rate=16000):
    """Generate audio from text with gTTS, resampled to the model's rate.

    gTTS emits 24 kHz MP3 and wav2vec2 expects 16 kHz. Nothing downstream resamples, so
    skipping this feeds the model audio at the wrong rate: "HELLO" decodes as "HOW LOW",
    and every frame index lands on a ~74 fps time base instead of 50 fps. Forced
    alignment still returns segments, with even higher confidence, so it fails silently.
    """
    # Remove separator characters for speech generation
    clean_text = text.replace("|", " ").strip()

    # gTTS writes MP3 regardless of the extension, so name it honestly.
    mp3_path = os.path.splitext(output_path)[0] + ".mp3"
    tts = gTTS(text=clean_text, lang='en', slow=False)
    tts.save(mp3_path)

    waveform, sample_rate = torchaudio.load(mp3_path)
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_rate)
    torchaudio.save(output_path, waveform, target_rate)

    return output_path

def main():
    print(f"Forced Alignment Demo v{__version__}")
    # torchaudio.set_audio_backend() was removed in torchaudio 2.1; the backend is
    # selected automatically now. The old call raised AttributeError, which the bare
    # except turned into a bogus "install soundfile sox" message on a fully working
    # install, and returned before doing any work.

    try:
        # Prepare transcript (3 words)
        transcript = "|HELLO|WORLD|TODAY|"

        # Load model first: the audio has to be generated at the rate it expects.
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model()

        # Generate demo audio
        print("Generating demo audio...")
        audio_path = generate_demo_audio(transcript, target_rate=bundle.sample_rate)

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
            sample_rate = bundle.sample_rate

        # Move model to available device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        waveform = waveform.to(device)
        
        # Get emissions
        with torch.inference_mode():
            emissions, _ = model(waveform)
            emissions = torch.log_softmax(emissions, dim=-1)
        
        emission = emissions[0].cpu().detach()
        
        # Get tokens
        labels = bundle.get_labels()
        dictionary = {c: i for i, c in enumerate(labels)}
        tokens = [dictionary[c] for c in transcript]
        
        # Generate trellis
        trellis = get_trellis(emission, tokens)
        
        # Find path
        path = backtrack(trellis, emission, tokens)
        
        # Merge repeats and words
        segments = merge_repeats(path, transcript)
        word_segments = merge_words(segments)
        
        # Visualizations
        print("\nGenerating visualizations...")
        visualize_emission(emission)
        visualize_trellis_with_path(trellis, path)
        visualize_alignments(trellis, segments, word_segments, waveform, sample_rate)
        
        # Print results. Raw frame indices mean little on their own, so convert them
        # back to seconds using the audio-to-frame ratio the model actually produced.
        ratio = waveform.size(1) / trellis.size(0) / sample_rate
        print("\nAlignment Results:")
        print("=================")
        for word in word_segments:
            print(f"{word}  ->  {word.start * ratio:5.2f}s - {word.end * ratio:5.2f}s")

    except Exception as e:
        # Show the real failure instead of guessing at a cause. The old handler
        # blamed missing packages for every error, which was wrong and misleading.
        import traceback
        print(f"Error occurred: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 