"""
Scans manually-downloaded, manually-extracted GRID corpus speakers and builds the
per-utterance manifest that every other script in this module reads.

Expected layout per speaker (matches how GRID's normal-quality video zip and 25kHz
audio zip unpack), under Speech/data/grid_raw/:

    grid_raw/
      s1/
        video/s1/*.mpg     (e.g. bbaf2n.mpg — one 3s clip per utterance)
        audio/s1/*.wav
      s2/
        video/s2/*.mpg
        audio/s2/*.wav
      ...

If your unzip produced a different nesting, either move files to match the above or
pass --video-glob / --audio-glob to override the search pattern per speaker.

Also assigns each speaker to exactly one of three disjoint splits — sync_train,
ar_train, heldout_real — following the paper's recipe of dividing the training
corpus into separate splits for the synchronization model, the autoregressive model,
and (here, since the paper's held-out set was a *different* labeled dataset we don't
have) a set of clean real videos reserved for threshold calibration and sanity-check
evaluation (Sec. 4.2).

Usage:
    python manifest.py --speakers s1,s2,s3,s4,s5,s6,s7,s8,s9
"""

import argparse
import csv
import random
from pathlib import Path

from config import GRID_RAW_DIR, MANIFEST_PATH, SPLIT_NAMES

DEFAULT_VIDEO_GLOB = "video/{speaker}/*.mpg"
DEFAULT_AUDIO_GLOB = "audio/{speaker}/*.wav"


def find_utterances(speaker, video_glob, audio_glob):
    """Matches video/audio files for one speaker by shared filename stem."""
    speaker_dir = GRID_RAW_DIR / speaker
    videos = {p.stem: p for p in speaker_dir.glob(video_glob.format(speaker=speaker))}
    audios = {p.stem: p for p in speaker_dir.glob(audio_glob.format(speaker=speaker))}

    stems = sorted(set(videos) & set(audios))
    missing_audio = sorted(set(videos) - set(audios))
    missing_video = sorted(set(audios) - set(videos))
    if missing_audio:
        print(f"  {speaker}: {len(missing_audio)} videos have no matching audio file (skipped)")
    if missing_video:
        print(f"  {speaker}: {len(missing_video)} audio files have no matching video (skipped)")

    return [(stem, videos[stem], audios[stem]) for stem in stems]


def assign_splits(speakers, seed):
    """Deterministic, speaker-disjoint 3-way split (as even as len(speakers) allows)."""
    speakers = list(speakers)
    random.Random(seed).shuffle(speakers)
    groups = {name: [] for name in SPLIT_NAMES}
    for i, speaker in enumerate(speakers):
        groups[SPLIT_NAMES[i % len(SPLIT_NAMES)]].append(speaker)
    return groups


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the utterance manifest + speaker split for the GRID corpus subset."
    )
    parser.add_argument(
        "--speakers", type=str, required=True,
        help="Comma-separated speaker ids present under Speech/data/grid_raw/, e.g. s1,s2,s3,s4,s5,s6,s7,s8,s9",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for the speaker split.")
    parser.add_argument("--video-glob", type=str, default=DEFAULT_VIDEO_GLOB)
    parser.add_argument("--audio-glob", type=str, default=DEFAULT_AUDIO_GLOB)
    return parser.parse_args()


def main():
    args = parse_args()
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]

    missing = [s for s in speakers if not (GRID_RAW_DIR / s).is_dir()]
    if missing:
        raise SystemExit(
            f"Missing speaker directories under {GRID_RAW_DIR}: {missing}. "
            f"Download+extract them from the GRID corpus first (see Speech/README.md)."
        )

    splits = assign_splits(speakers, args.seed)
    speaker_to_split = {sp: name for name, sps in splits.items() for sp in sps}
    for name in SPLIT_NAMES:
        print(f"{name}: {splits[name]}")

    rows = []
    for speaker in speakers:
        utts = find_utterances(speaker, args.video_glob, args.audio_glob)
        print(f"{speaker}: {len(utts)} matched utterances")
        for stem, video_path, audio_path in utts:
            rows.append(
                {
                    "speaker": speaker,
                    "utt_id": stem,
                    "split": speaker_to_split[speaker],
                    "video_path": str(video_path),
                    "audio_path": str(audio_path),
                }
            )

    if not rows:
        raise SystemExit("No matched video/audio pairs found across the given speakers.")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker", "utt_id", "split", "video_path", "audio_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} utterances from {len(speakers)} speakers to {MANIFEST_PATH}")
    for name in SPLIT_NAMES:
        n = sum(1 for r in rows if r["split"] == name)
        print(f"  {name}: {n} utterances ({len(splits[name])} speakers)")


if __name__ == "__main__":
    main()
