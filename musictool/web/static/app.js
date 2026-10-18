"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = { Drums: "--drums", Bass: "--bass", Synths: "--synths", Vocals: "--vocals" };
const DRUM_ROWS = [
  { pitch: 49, label: "Cymbal" },
  { pitch: 47, label: "Toms" },
  { pitch: 46, label: "Open hat" },
  { pitch: 42, label: "Hi-hat" },
  { pitch: 38, label: "Snare" },
  { pitch: 36, label: "Kick" },
];
const PX_PER_SEC = 90;
const ROW_H = 6;
const DRUM_ROW_H = 14;
const LABEL_W = 70;

let selectedFile = null;
let result = null;
let jobId = null;

// ---------- upload ----------
const drop = $("drop");
$("file").addEventListener("change", (e) => pickFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pickFile(e.dataTransfer.files[0]));

function pickFile(file) {
  if (!file) return;
  selectedFile = file;
  $("filename").textContent = file.name;
  $("go").disabled = false;
}

// Untouched = "auto": the server uses thresholds tuned per stem.
let noteSensitivityTouched = false;
$("onset_out").textContent = "auto";
$("onset_threshold").addEventListener("input", (e) => {
  noteSensitivityTouched = true;
  $("onset_out").textContent = Number(e.target.value).toFixed(2);
});
$("drum_sensitivity").addEventListener("input", (e) => ($("drum_out").textContent = Number(e.target.value).toFixed(2)));

const ENGINE_LABELS = {
  "demucs-ft": "Demucs v4 fine-tuned", demucs: "Demucs v4", hpss: "DSP separation",
  ensemble: "DrumSep + ADTOF drums", drumsep: "DrumSep drums", adtof: "ADTOF drums", dsp: "DSP drums", crepe: "CREPE bass", pyin: "pYIN bass",
  "basic-pitch": "Basic Pitch", vote: "Synths: Basic Pitch + YourMT3+ voting", hybrid: "Instruments: Basic Pitch + YourMT3+", yourmt3: "YourMT3+", layers: "Lead + pad layers",
};
// Colours for the extra instrument tracks YourMT3+ produces.
const EXTRA_COLORS = ["#b197fc", "#ffa94d", "#63e6be", "#f783ac", "#74c0fc", "#d8f5a2", "#ffe066", "#e599f7", "#ff8787"];

fetch("/api/info").then((r) => r.json()).then((info) => {
  // Grey out engines this machine doesn't have, and say what "Auto" resolves to.
  for (const opt of document.querySelectorAll("option[data-needs]")) {
    if (!info[opt.dataset.needs]) {
      opt.disabled = true;
      opt.textContent += " – not installed";
    }
  }
  const d = info.defaults;
  const auto = { separation: d.separation, drum_engine: d.drum_engine, mono_engine: d.mono_engine, synth_engine: d.synth_engine };
  for (const [id, value] of Object.entries(auto)) {
    const opt = document.querySelector(`#${id} option[value="auto"]`);
    opt.textContent = `Auto: ${ENGINE_LABELS[value] || value}${info.gpu ? "" : " (CPU)"}`;
  }
});

$("go").addEventListener("click", async () => {
  if (!selectedFile) return;
  stopPlayback();
  show("progress"); hide("error"); hide("result");
  setProgress("Uploading…", 0);
  $("go").disabled = true;

  const form = new FormData();
  form.append("file", selectedFile);
  for (const id of ["separation", "drum_engine", "mono_engine", "synth_engine", "quantize", "drum_sensitivity"]) {
    form.append(id, $(id).value);
  }
  if ($("min_note_ms").value) form.append("min_note_ms", $("min_note_ms").value);
  // The slider is "sensitivity"; Basic Pitch wants a confidence threshold.
  if (noteSensitivityTouched) form.append("onset_threshold", (1 - Number($("onset_threshold").value)).toFixed(2));
  if ($("bpm").value) form.append("bpm", $("bpm").value);
  for (const id of ["drums", "bass", "melodic"]) form.append(id, $(id).checked);

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    jobId = body.id;
    poll();
  } catch (err) {
    fail(err.message);
  }
});

async function poll() {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    const job = await res.json();
    if (!res.ok) throw new Error(job.detail || res.statusText);
    if (job.status === "error") return fail(job.error);
    setProgress(job.stage, job.progress);
    if (job.status === "done") return showResult(job);
    setTimeout(poll, 700);
  } catch (err) {
    fail(err.message);
  }
}

function setProgress(stage, frac) {
  $("stage").textContent = stage;
  $("barfill").style.width = `${Math.round(frac * 100)}%`;
}

function fail(message) {
  hide("progress");
  $("error").textContent = `Transcription failed: ${message}`;
  show("error");
  $("go").disabled = false;
}

const show = (id) => ($(id).hidden = false);
const hide = (id) => ($(id).hidden = true);
const color = (name) => {
  if (COLORS[name]) return getComputedStyle(document.documentElement).getPropertyValue(COLORS[name]).trim();
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return EXTRA_COLORS[h % EXTRA_COLORS.length];
};
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// ---------- result ----------
function showResult(job) {
  hide("progress");
  $("go").disabled = false;
  result = job.result;
  const total = result.tracks.reduce((n, t) => n + t.notes.length, 0);
  $("title").textContent = job.filename;
  $("meta").textContent = `${result.bpm.toFixed(1)} BPM · ${fmtTime(result.duration)} · ${total} notes`;
  $("dl-all").href = `/api/jobs/${jobId}/midi`;
  $("engines").textContent = Object.values(result.engines || {}).map((e) => ENGINE_LABELS[e] || e).join(" · ");

  const stemLinks = $("stem-links");
  stemLinks.innerHTML = "";
  for (const stem of job.stems || []) {
    const a = document.createElement("a");
    a.href = `/api/jobs/${jobId}/stems/${stem}`;
    a.download = "";
    a.textContent = stem;
    stemLinks.appendChild(a);
  }
  $("stems").hidden = !(job.stems || []).length;

  const list = $("tracks");
  list.innerHTML = "";
  for (const t of result.tracks) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="swatch"></span><span class="name"></span><span class="count"></span><a download>Download track</a>`;
    li.querySelector(".swatch").style.background = color(t.name);
    li.querySelector(".name").textContent = t.name;
    li.querySelector(".count").textContent = `${t.notes.length} notes`;
    li.querySelector("a").href = `/api/jobs/${jobId}/midi?track=${encodeURIComponent(t.name)}`;
    list.appendChild(li);
  }

  $("audio").src = URL.createObjectURL(selectedFile);
  position = 0;
  show("result");
  drawRoll();
}

// ---------- piano roll ----------
function rollLayout() {
  const pitched = result.tracks.filter((t) => !t.is_drum).flatMap((t) => t.notes.map((n) => n.pitch));
  const lo = pitched.length ? Math.min(...pitched) - 2 : 48;
  const hi = pitched.length ? Math.max(...pitched) + 2 : 72;
  const hasDrums = result.tracks.some((t) => t.is_drum && t.notes.length);
  const pitchH = (hi - lo + 1) * ROW_H;
  const drumH = hasDrums ? DRUM_ROWS.length * DRUM_ROW_H + 8 : 0;
  return { lo, hi, pitchH, drumH, height: pitchH + drumH, width: LABEL_W + Math.ceil(result.duration * PX_PER_SEC) + 20 };
}

function drawRoll() {
  const canvas = $("roll");
  const L = rollLayout();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = L.width * dpr;
  canvas.height = L.height * dpr;
  canvas.style.width = `${L.width}px`;
  canvas.style.height = `${L.height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, L.width, L.height);

  // pitch grid: shade black keys, label each C
  for (let p = L.lo; p <= L.hi; p++) {
    const y = (L.hi - p) * ROW_H;
    if ([1, 3, 6, 8, 10].includes(p % 12)) {
      ctx.fillStyle = "#11141a";
      ctx.fillRect(LABEL_W, y, L.width, ROW_H);
    }
    if (p % 12 === 0) {
      ctx.fillStyle = "#2a2f3a";
      ctx.fillRect(LABEL_W, y + ROW_H - 1, L.width, 1);
      ctx.fillStyle = "#8b93a5";
      ctx.font = "10px system-ui";
      ctx.fillText(`C${p / 12 - 1}`, 8, y + ROW_H);
    }
  }
  // beat lines
  const beat = 60 / result.bpm;
  for (let b = 0, t = 0; t < result.duration; b++, t = b * beat) {
    ctx.fillStyle = b % 4 === 0 ? "#2f3542" : "#1a1e26";
    ctx.fillRect(LABEL_W + t * PX_PER_SEC, 0, 1, L.height);
  }
  // drum lane
  if (L.drumH) {
    const top = L.pitchH + 8;
    ctx.fillStyle = "#2a2f3a";
    ctx.fillRect(0, L.pitchH + 3, L.width, 1);
    DRUM_ROWS.forEach((row, i) => {
      ctx.fillStyle = "#8b93a5";
      ctx.font = "10px system-ui";
      ctx.fillText(row.label, 8, top + i * DRUM_ROW_H + 10);
    });
  }
  for (const track of result.tracks) {
    ctx.fillStyle = color(track.name);
    for (const n of track.notes) {
      const x = LABEL_W + n.start * PX_PER_SEC;
      ctx.globalAlpha = 0.45 + 0.55 * (n.velocity / 127);
      if (track.is_drum) {
        const row = DRUM_ROWS.findIndex((r) => r.pitch === n.pitch);
        if (row < 0) continue;
        ctx.fillRect(x, L.pitchH + 8 + row * DRUM_ROW_H + 2, 4, DRUM_ROW_H - 4);
      } else {
        const w = Math.max(2, (n.end - n.start) * PX_PER_SEC - 1);
        ctx.fillRect(x, (L.hi - n.pitch) * ROW_H, w, ROW_H - 1);
      }
    }
  }
  ctx.globalAlpha = 1;
  // playhead
  ctx.fillStyle = "#fff";
  ctx.fillRect(LABEL_W + position * PX_PER_SEC, 0, 2, L.height);
}

$("roll").addEventListener("click", (e) => {
  const rect = e.target.getBoundingClientRect();
  const t = Math.max(0, (e.clientX - rect.left - LABEL_W) / PX_PER_SEC);
  const wasPlaying = playing;
  stopPlayback();
  position = Math.min(t, result.duration);
  drawRoll();
  if (wasPlaying) startPlayback();
});

// ---------- playback (WebAudio synth preview) ----------
let audioCtx = null;
let playing = false;
let position = 0;
let startedAt = 0;
let schedTimer = null;
let queue = [];
let qi = 0;
let noiseBuffer = null;
let master = null;
const LOOKAHEAD = 0.25;

$("play").addEventListener("click", () => (playing ? stopPlayback() : startPlayback()));

function startPlayback() {
  if (!result) return;
  audioCtx = audioCtx || new AudioContext();
  audioCtx.resume();
  if (!noiseBuffer) {
    noiseBuffer = audioCtx.createBuffer(1, audioCtx.sampleRate, audioCtx.sampleRate);
    const d = noiseBuffer.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  master = audioCtx.createGain();
  master.gain.value = 0.5;
  master.connect(audioCtx.destination);

  queue = result.tracks
    .flatMap((t) => t.notes.map((n) => ({ ...n, track: t.name, drum: t.is_drum })))
    .filter((n) => n.start >= position)
    .sort((a, b) => a.start - b.start);
  qi = 0;
  startedAt = audioCtx.currentTime + 0.05 - position;
  playing = true;
  $("play").textContent = "❚❚ Pause";

  const audio = $("audio");
  audio.muted = !$("hear-audio").checked;
  audio.currentTime = position;
  audio.play().catch(() => {});

  schedTimer = setInterval(schedule, 50);
  schedule();
  requestAnimationFrame(tick);
}

function stopPlayback() {
  if (!playing) return;
  playing = false;
  clearInterval(schedTimer);
  position = Math.min(audioCtx.currentTime - startedAt, result.duration);
  master.gain.setTargetAtTime(0, audioCtx.currentTime, 0.02);
  $("audio").pause();
  $("play").textContent = "▶ Play";
}

$("hear-audio").addEventListener("change", (e) => ($("audio").muted = !e.target.checked));
$("hear-midi").addEventListener("change", (e) => {
  if (master) master.gain.setTargetAtTime(e.target.checked ? 0.5 : 0, audioCtx.currentTime, 0.02);
});

function schedule() {
  const horizon = audioCtx.currentTime + LOOKAHEAD;
  while (qi < queue.length && startedAt + queue[qi].start < horizon) {
    const n = queue[qi++];
    if ($("hear-midi").checked) voice(n, Math.max(audioCtx.currentTime, startedAt + n.start));
  }
}

function tick() {
  if (!playing) return;
  position = audioCtx.currentTime - startedAt;
  if (position >= result.duration) {
    stopPlayback();
    position = 0;
  }
  $("time").textContent = fmtTime(Math.max(0, position));
  drawRoll();
  const wrap = document.querySelector(".roll-wrap");
  const x = LABEL_W + position * PX_PER_SEC;
  if (x > wrap.scrollLeft + wrap.clientWidth - 80 || x < wrap.scrollLeft) wrap.scrollLeft = x - 120;
  requestAnimationFrame(tick);
}

const hz = (p) => 440 * Math.pow(2, (p - 69) / 12);

function env(gainNode, t, peak, attack, release, dur) {
  const g = gainNode.gain;
  g.setValueAtTime(0, t);
  g.linearRampToValueAtTime(peak, t + attack);
  g.setTargetAtTime(0, t + Math.max(attack, dur), release);
}

function noise(t, dur, filterType, freq, peak) {
  const src = audioCtx.createBufferSource();
  src.buffer = noiseBuffer;
  const f = audioCtx.createBiquadFilter();
  f.type = filterType;
  f.frequency.value = freq;
  const g = audioCtx.createGain();
  env(g, t, peak, 0.001, dur / 3, 0);
  src.connect(f).connect(g).connect(master);
  src.start(t);
  src.stop(t + dur * 2);
}

function voice(n, t) {
  const vel = n.velocity / 127;
  if (n.drum) {
    if (n.pitch === 36) {
      const o = audioCtx.createOscillator();
      const g = audioCtx.createGain();
      o.frequency.setValueAtTime(150, t);
      o.frequency.exponentialRampToValueAtTime(45, t + 0.12);
      env(g, t, vel, 0.002, 0.08, 0);
      o.connect(g).connect(master);
      o.start(t);
      o.stop(t + 0.5);
    } else if (n.pitch === 38) {
      noise(t, 0.18, "bandpass", 1800, vel * 0.7);
    } else if (n.pitch === 47) {
      noise(t, 0.25, "lowpass", 400, vel * 0.8);
    } else if (n.pitch === 49) {
      noise(t, 1.2, "highpass", 5000, vel * 0.3);
    } else {
      noise(t, n.pitch === 46 ? 0.3 : 0.05, "highpass", 7000, vel * 0.35);
    }
    return;
  }
  const dur = n.end - n.start;
  const o = audioCtx.createOscillator();
  const f = audioCtx.createBiquadFilter();
  const g = audioCtx.createGain();
  o.frequency.value = hz(n.pitch);
  if (n.track === "Bass") {
    o.type = "sawtooth";
    f.frequency.value = 700;
    env(g, t, vel * 0.5, 0.005, 0.03, dur);
  } else {
    o.type = n.track === "Vocals" ? "triangle" : "sawtooth";
    f.frequency.value = 2800;
    env(g, t, vel * 0.12, 0.01, 0.08, dur);
  }
  o.connect(f).connect(g).connect(master);
  o.start(t);
  o.stop(t + dur + 0.6);
}
