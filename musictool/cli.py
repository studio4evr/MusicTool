"""Command line interface: ``musictool song.wav -o song.mid``."""

import argparse
import sys
from pathlib import Path

from .pipeline import TranscriptionOptions, transcribe


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
    parser.add_argument("--separation", choices=["hpss", "demucs"], default="hpss",
                        help="stem separation: fast DSP (hpss) or neural Demucs (needs `pip install demucs`)")
    parser.add_argument("--bpm", type=float, help="override detected tempo")
    parser.add_argument("--quantize", type=int, default=16, help="grid division, e.g. 16 for 16ths; 0 disables")
    parser.add_argument("--no-drums", action="store_true")
    parser.add_argument("--no-bass", action="store_true")
    parser.add_argument("--no-melodic", action="store_true")
    parser.add_argument("--onset-threshold", type=float, default=0.5, help="0-1, higher = fewer notes")
    parser.add_argument("--frame-threshold", type=float, default=0.3, help="0-1, higher = shorter notes")
    parser.add_argument("--min-note-ms", type=float, default=80.0)
    parser.add_argument("--mono-engine", choices=["pyin", "basic-pitch"], default="pyin",
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
        mono_engine=args.mono_engine,
    )

    def progress(stage: str, frac: float) -> None:
        print(f"[{frac * 100:5.1f}%] {stage}", file=sys.stderr)

    result = transcribe(src, options, progress)
    result.write(out)
    print(f"Tempo: {result.bpm:.1f} BPM")
    for track in result.tracks:
        print(f"  {track.name:<8} {len(track.notes):5d} notes")
    print(f"Wrote {out}")

    if args.separate_tracks:
        for track in result.tracks:
            path = out.with_name(f"{out.stem}_{track.name.lower()}.mid")
            result.write(path, [track.name])
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
