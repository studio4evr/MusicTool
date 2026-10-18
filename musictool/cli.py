"""Command line interface: ``musictool song.wav -o song.mid``."""

import argparse
import re
import sys
from pathlib import Path

from .pipeline import TranscriptionOptions, transcribe
from .separation import NEURAL_MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="musictool",
        description="Transcribe electronic music audio into a multi-track MIDI file using AI.",
        epilog="Run `musictool serve` to start the web app.",
    )
    serve = argparse.ArgumentParser(prog="musictool serve", description="Start the web app.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--open", action="store_true", help="open the app in the default browser")

    parser.add_argument("input", nargs="?", help="audio file (wav, mp3, flac, ogg, aiff, ...)")
    parser.add_argument("-o", "--output", help="output .mid path (default: next to input)")
    parser.add_argument("--separate-tracks", action="store_true", help="also write one .mid per track")
    parser.add_argument("--save-stems", action="store_true", help="also write the separated audio stems as .wav")
    parser.add_argument("--separation", choices=["auto", "hpss", *NEURAL_MODELS], default="auto",
                        help="stem separation: auto (best installed), fast DSP (hpss) or a neural model")
    parser.add_argument("--drum-engine", choices=["auto", "ensemble", "drumsep", "adtof", "dsp"], default="auto",
                        help="drums: DrumSep kit separation + ADTOF (ensemble), either alone, or the DSP classifier")
    parser.add_argument("--bpm", type=float, help="override detected tempo")
    parser.add_argument("--quantize", type=int, default=16, help="grid division, e.g. 16 for 16ths; 0 disables")
    parser.add_argument("--no-drums", action="store_true")
    parser.add_argument("--no-bass", action="store_true")
    parser.add_argument("--no-melodic", action="store_true")
    parser.add_argument("--onset-threshold", type=float, help="0-1, higher = fewer notes (default: tuned per stem)")
    parser.add_argument("--frame-threshold", type=float, help="0-1, higher = shorter notes (default: tuned per stem)")
    parser.add_argument("--min-note-ms", type=float, help="default: tuned per stem (50-80)")
    parser.add_argument("--synth-engine", choices=["auto", "vote", "hybrid", "layers", "yourmt3", "basic-pitch"], default="auto",
                        help="synths: Basic Pitch + 4 YourMT3+ passes voting (vote), Basic Pitch split by YourMT3+ labels (hybrid), "
                             "lead + pad layers, YourMT3+ alone, or Basic Pitch as one track")
    parser.add_argument("--mt3-variant", choices=["moe-nops", "moe-ps", "ptf-ps"], default="moe-nops",
                        help="YourMT3+ checkpoint for synth instrument labels")
    parser.add_argument("--mt3-input", choices=["full", "no-drums"], default="no-drums",
                        help="feed that model the full mix or the mix minus the drum stem")
    parser.add_argument("--bass-mt3-variant", choices=["moe-nops", "moe-ps", "ptf-ps"], default="ptf-ps",
                        help="YourMT3+ checkpoint for the bass (mono engine yourmt3)")
    parser.add_argument("--mono-engine", choices=["auto", "yourmt3", "crepe", "pyin", "basic-pitch"], default="auto",
                        help="engine for monophonic bass/vocal lines")
    parser.add_argument("--drum-sensitivity", type=float, default=0.5, help="0-1, higher = more drum hits")

    # argparse can't mix an optional positional with subcommands, so route `serve` manually.
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "serve":
        args = serve.parse_args(argv[1:])
        import uvicorn

        if args.open:
            import threading
            import webbrowser

            url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
            threading.Timer(1.5, webbrowser.open, [url]).start()
            print(f"MusicTool running at {url} (press Ctrl+C to stop)")
        uvicorn.run("musictool.web.app:app", host=args.host, port=args.port)
        return 0

    args = parser.parse_args(argv)
    if not args.input:
        parser.print_help()
        return 1

    src = Path(args.input)
    if not src.exists():
        parser.error(f"file not found: {src}")
    out = Path(args.output) if args.output else src.with_suffix(".mid")

    options = TranscriptionOptions(
        separation=args.separation,
        bpm=args.bpm,
        quantize=args.quantize or None,
        drums=not args.no_drums,
        bass=not args.no_bass,
        melodic=not args.no_melodic,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        min_note_ms=args.min_note_ms,
        drum_sensitivity=args.drum_sensitivity,
        drum_engine=args.drum_engine,
        mono_engine=args.mono_engine,
        synth_engine=args.synth_engine,
        mt3_variant=args.mt3_variant,
        mt3_input=args.mt3_input,
        bass_mt3_variant=args.bass_mt3_variant,
        stems_dir=out.parent if args.save_stems else None,
    )

    def progress(stage: str, frac: float) -> None:
        print(f"[{frac * 100:5.1f}%] {stage}", file=sys.stderr)

    result = transcribe(src, options, progress)
    result.write(out)
    print(f"Tempo: {result.bpm:.1f} BPM")
    print("Engines: " + ", ".join(f"{k}={v}" for k, v in result.engines.items()))
    for track in result.tracks:
        print(f"  {track.name:<8} {len(track.notes):5d} notes")
    print(f"Wrote {out}")

    if args.separate_tracks:
        for track in result.tracks:
            slug = re.sub(r"[^\w]+", "_", track.name.lower()).strip("_")  # "Bells / Mallets" -> bells_mallets
            path = out.with_name(f"{out.stem}_{slug}.mid")
            result.write(path, [track.name])
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
