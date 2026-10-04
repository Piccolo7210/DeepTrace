"""
Bridges the Kaggle-downloaded, pre-split GRID corpus (download_and_split_grid.py's
train/ + test/ output) into the flat per-speaker layout manifest.py expects, without
copying any of the (large) video/audio files.

download_and_split_grid.py's train/test split is a *within-speaker*, per-utterance
80/20 split -- unrelated to (and not used by) manifest.py's own *speaker-disjoint*
3-way split (sync_train/ar_train/heldout_real, see manifest.py's docstring). This
script just pools each speaker's train+test utterances back together via symlinks, so
manifest.py sees every utterance for the speakers you choose, and (optionally) caps
how many utterances per speaker get linked to control total preprocessed size -- each
linked utterance becomes a ~2.7MB face-crop frames.npy after preprocess.py runs, so
linking all ~1000 utterances for many speakers adds up fast (see
Speech/kaggle/KAGGLE_INSTRUCTIONS.md's sizing note).

Expected source layout, matching download_and_split_grid.py's OUT_DIR:
    <source-dir>/
      train/s1/audio/*.wav   train/s1/video/*.mpg
      train/s2/...
      test/s1/audio/*.wav    test/s1/video/*.mpg
      test/s2/...

Produces (symlinks only; GRID_RAW_DIR is created if needed):
    data/grid_raw/
      s1/video/*.mpg
      s1/audio/*.wav
      s2/...

That flat shape means manifest.py needs no code changes to read it -- just call it
afterwards with --video-glob "video/*.mpg" --audio-glob "audio/*.wav".

Only ever meant to run inside the Kaggle notebook's Linux container (against the
Kaggle-downloaded data, which never touches this machine) -- unlike manifest.py it's
not a general-purpose tool, just glue for that one known source layout.

Usage (inside the Kaggle preprocessing notebook):
    python reorganize_grid.py --source-dir /kaggle/input/<grid-data-slug>/grid_data \
        --speakers s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19,s20 \
        --max-per-speaker 140
"""

import argparse
import random
from pathlib import Path

from config import GRID_RAW_DIR

SPLITS = ("train", "test")


def find_speaker_pairs(source_dir, speaker):
    """Pools video/audio files for one speaker across train/ and test/, matched by
    filename stem. Aborts if a stem appears in both -- shouldn't happen given the
    source's per-utterance split, so surfacing it beats silently picking one."""
    pairs = {}
    for split in SPLITS:
        video_dir = source_dir / split / speaker / "video"
        audio_dir = source_dir / split / speaker / "audio"
        videos = {p.stem: p for p in video_dir.glob("*.mpg")} if video_dir.is_dir() else {}
        audios = {p.stem: p for p in audio_dir.glob("*.wav")} if audio_dir.is_dir() else {}

        stems = sorted(set(videos) & set(audios))
        missing_audio = sorted(set(videos) - set(audios))
        missing_video = sorted(set(audios) - set(videos))
        if missing_audio:
            print(f"  {speaker}/{split}: {len(missing_audio)} videos have no matching audio (skipped)")
        if missing_video:
            print(f"  {speaker}/{split}: {len(missing_video)} audio files have no matching video (skipped)")

        for stem in stems:
            if stem in pairs:
                raise SystemExit(
                    f"{speaker}: stem {stem!r} found in both train/ and test/ -- that "
                    f"shouldn't happen given a per-utterance split; investigate {source_dir} "
                    f"before continuing."
                )
            pairs[stem] = (videos[stem], audios[stem])

    return pairs


def link_speaker(source_dir, speaker, max_per_speaker, seed):
    pairs = find_speaker_pairs(source_dir, speaker)
    stems = sorted(pairs)

    if max_per_speaker is not None and len(stems) > max_per_speaker:
        # Shuffle, not sort-then-truncate: GRID stems encode the sentence grammar
        # positionally (first letter = command word), so an alphabetic prefix would
        # systematically drop whole command-word categories instead of sampling
        # evenly across them.
        random.Random(seed).shuffle(stems)
        stems = stems[:max_per_speaker]

    video_out = GRID_RAW_DIR / speaker / "video"
    audio_out = GRID_RAW_DIR / speaker / "audio"
    video_out.mkdir(parents=True, exist_ok=True)
    audio_out.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        video_src, audio_src = pairs[stem]
        video_dst = video_out / video_src.name
        audio_dst = audio_out / audio_src.name
        video_dst.unlink(missing_ok=True)
        audio_dst.unlink(missing_ok=True)
        video_dst.symlink_to(video_src.resolve())
        audio_dst.symlink_to(audio_src.resolve())

    return len(pairs), len(stems)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge Kaggle GRID train/+test/ into manifest.py's expected layout, via symlinks."
    )
    parser.add_argument(
        "--source-dir", type=Path, required=True,
        help="Path to the grid_data folder from the download notebook's dataset, e.g. "
             "/kaggle/input/<slug>/grid_data (run `!find /kaggle/input -maxdepth 4` first "
             "if unsure of the exact mount path).",
    )
    parser.add_argument(
        "--speakers", type=str, required=True,
        help="Comma-separated speaker ids, e.g. s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,...,s20",
    )
    parser.add_argument(
        "--max-per-speaker", type=int, default=None,
        help="Cap utterances linked per speaker (train+test pooled, then seeded-shuffled "
             "before truncating). Omit for no cap.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for --max-per-speaker.")
    return parser.parse_args()


def main():
    args = parse_args()
    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]

    missing = [s for s in speakers if not any((args.source_dir / split / s).is_dir() for split in SPLITS)]
    if missing:
        raise SystemExit(f"No train/ or test/ directory found for speakers {missing} under {args.source_dir}.")

    total_matched, total_kept = 0, 0
    for speaker in speakers:
        matched, kept = link_speaker(args.source_dir, speaker, args.max_per_speaker, args.seed)
        print(f"{speaker}: {matched} matched utterances (train+test pooled), linked {kept}")
        total_matched += matched
        total_kept += kept

    print(f"\nLinked {total_kept}/{total_matched} utterances across {len(speakers)} speakers into {GRID_RAW_DIR}")
    print('Next: python manifest.py --speakers ... --video-glob "video/*.mpg" --audio-glob "audio/*.wav"')


if __name__ == "__main__":
    main()
