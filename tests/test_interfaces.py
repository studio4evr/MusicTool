import time

import pretty_midi
import pytest

from musictool.cli import main


def test_cli(track, tmp_path, capsys):
    out = tmp_path / "song.mid"
    dsp = ["--separation", "hpss", "--drum-engine", "dsp", "--mono-engine", "pyin"]
    assert main([str(track[0]), "-o", str(out), "--separate-tracks", "--no-melodic", *dsp]) == 0
    assert out.exists()
    assert (tmp_path / "song_drums.mid").exists()
    assert (tmp_path / "song_bass.mid").exists()
    assert "BPM" in capsys.readouterr().out


def test_cli_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope.wav")])


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from musictool.web.app import app

    return TestClient(app)


def test_web_roundtrip(client, track):
    assert client.get("/").status_code == 200
    with open(track[0], "rb") as fh:
        res = client.post(
            "/api/transcribe",
            files={"file": ("my song.wav", fh, "audio/wav")},
            data={"melodic": "false", "quantize": "16", "separation": "hpss", "drum_engine": "dsp", "mono_engine": "pyin"},
        )
    assert res.status_code == 200, res.text
    job_id = res.json()["id"]

    deadline = time.time() + 120
    while True:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error") or time.time() > deadline:
            break
        time.sleep(0.2)
    assert job["status"] == "done", job
    assert [t["name"] for t in job["result"]["tracks"]] == ["Drums", "Bass"]

    midi = client.get(f"/api/jobs/{job_id}/midi")
    assert midi.status_code == 200
    assert 'filename="my song.mid"' in midi.headers["content-disposition"]
    assert midi.content[:4] == b"MThd"

    one = client.get(f"/api/jobs/{job_id}/midi", params={"track": "Bass"})
    assert one.status_code == 200
    assert client.get(f"/api/jobs/{job_id}/midi", params={"track": "Nope"}).status_code == 404


def test_web_rejects_bad_upload(client):
    res = client.post("/api/transcribe", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400
    assert client.get("/api/jobs/doesnotexist").status_code == 404
