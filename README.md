# vlm-stream-runner

A native macOS full-duplex camera monitor for MiniCPM-o 4.5 and
`llama.cpp-omni`. By default, camera frames, microphone chunks, and synthesized
speech stay in memory. The runner disables llama.cpp-omni runtime artifacts, so
ordinary execution creates no WAV, JPEG, token, embedding, timing,
completion-flag, log, or temporary files. Files are written only when explicitly
requested with `--perfetto-trace` or `--dump`.

## Build

Requirements: macOS, Xcode command-line tools, CMake, Ninja, and FFmpeg with
AVFoundation support.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vlm-stream-runner -j 8
```

The build links the runner directly to `omni` and `llama-common`; it does not
start an HTTP server or Python runtime.

## Run

```sh
./build/bin/vlm-stream-runner
```

Options:

```text
vlm-stream-runner
  [--model-dir /Users/markma/Documents/models/MiniCPM-o-4_5-gguf]
  [--camera-index 0]
  [--ref-audio PATH]
  [--vision-backend coreml|metal]
  [--perfetto-trace PATH]
  [--dump K]
```

Grant camera and microphone access to the terminal application that launches
the runner. If camera index 0 is not the built-in camera, list AVFoundation
devices with:

```sh
ffmpeg -f avfoundation -list_devices true -i ""
```

Press Ctrl-C for an orderly shutdown: capture stops, submitted observations
finish, TTS drains, queued playback finishes, and model/audio resources are
released.

### Artifact dump

Pass `--dump K` to save artifacts for scheduled iteration IDs 1 through K in a
`dump/` directory. If `dump/` already exists, the runner removes its prior
`iteration-*` directories before capture so the new dump cannot be confused
with stale output. Unrelated files in `dump/` are preserved. Each accepted
iteration uses this layout:

```text
dump/iteration-000001/observation.jpg
dump/iteration-000001/observation.wav
dump/iteration-000001/llm.txt
dump/iteration-000001/tts.wav
```

`llm.txt` is empty for a successful silent/LISTEN result, and `tts.wav` exists
only when that iteration produces synthesized audio. A scheduled tick rejected
for backpressure or missing input has no iteration directory. Because iteration
IDs advance once per second, `--dump 120` covers approximately the first two
minutes while leaving all later iterations artifact-free.

### Perfetto trace

Pass `--perfetto-trace monitor.json` to write a Chrome Trace JSON file with six
logical pipeline tracks: observation capture, joint vision/audio encoding, LLM
prefill+decode, TTS, Token2Wav, and audio playback. The same iteration keeps the
same color across tracks. Rejected one-second observations and discarded
playback audio are labeled `dropped` in gray.

Parallel work for one observation is deliberately collapsed. Camera and
microphone capture form one observation interval, and concurrent vision/audio
encoding forms one interval from launch through their join (wall time is
approximately `max(vision, audio)`, not their sum). Component-level tracks are
not emitted.

Trace records are buffered in bounded memory and serialized during orderly
shutdown. The output is capped at 64 MiB; events beyond the cap are counted in
the trace metadata instead of growing the file. Open the JSON file in the
Perfetto UI.

## Runtime behavior

- The camera is one persistent FFmpeg process capturing its supported native
  1280×720 mode at 30 FPS and producing a full-frame, aspect-correct,
  high-quality JPEG at 1 FPS. Vision uses the model's high-image setting: one
  overview plus up to two detail slices. Only the newest complete frame is
  retained.
- The default microphone stays active during TTS playback. Each observation
  carries a one-second, 16 kHz mono WAV buffer.
- One isolated asynchronous duplex stream is used. No more than two
  observations can be outstanding; capture ticks are dropped when inference
  falls behind.
- TTS PCM goes directly to a 24 kHz playback queue capped at 30 seconds. Oldest
  audio is dropped with a warning if that cap is reached.
- The context is 8192 tokens with turn-aware sliding thresholds of 7000/5000.
- The duplex LLM sampler is greedy so its SPEAK/LISTEN and reminder-text
  decisions are deterministic. The browser-oriented three-observation forced
  LISTEN startup gate is disabled, while TTS retains its intended sampling
  behavior.
- The prompt requires exactly `Please get off your phone` on every observation
  where the person is visibly holding a phone, and silence otherwise. Scrolling,
  tapping, or looking at the screen is not required.

Supporting multiple inputs later means constructing one `StreamWorker` and one
independent `omni_context` per stream. Do not multiplex them through ordinary
text-generation slots; keep using the asynchronous duplex session API.

## Tests

```sh
cmake --build build --target vlm-stream-runner-tests llama-omni-test-duplex -j 8
ctest --test-dir build --output-on-failure
```

The local unit suite covers JPEG framing, latest-frame replacement,
microphone chunking, WAV construction, prompt wrapping, resampling, bounded
playback, and queue shutdown. The model-backed duplex test retains path input
by default and adds memory/artifact modes:

```sh
./build/bin/llama-omni-test-duplex \
  -m /Users/markma/Documents/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-F16.gguf \
  --omni --memory-io --no-artifacts -o /tmp/omni-artifact-check \
  --test llama.cpp-omni/tools/omni/assets/test_case/duplex_omni_test_case/duplex_omni_test_case_ 9
```

`--no-artifacts` refuses a non-empty output directory and verifies that the
directory remains absent or empty. Run the same command without `--memory-io`
to exercise backward-compatible path input.
