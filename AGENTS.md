# Project guidance for vlm-stream-runner

This is an ongoing native macOS C++ project for a long-running, full-duplex
MiniCPM-o 4.5 camera monitor. Preserve the behavior and constraints below when
making changes. The nested `llama.cpp-omni/AGENTS.md` also applies to changes in
that repository; in particular, do not commit or push on the user's behalf.

## Product intent

- Monitor the built-in MacBook camera and default microphone continuously.
- Once per second, submit the latest visual observation plus the latest one
  second of microphone audio to MiniCPM-o 4.5's streaming duplex pipeline.
- If the person is actively scrolling through a phone, say exactly
  `Please get off your phone` and nothing else. Repeat on every qualifying
  observation. Otherwise remain silent and ignore speech.
- The prompt language is English. Voice timbre and accent come from the
  reference WAV; do not try to fix a reference-voice accent by changing the
  prompt language. `--ref-audio PATH` is the intended override.
- This is meant to run for hours. Runtime memory and queues must remain bounded,
  and ordinary execution must not grow files on disk.
- The microphone intentionally remains live while synthesized speech plays.
  Echo cancellation/suppression is currently out of scope.

## Model and platform

- The already-downloaded model root is
  `/Users/markma/Documents/models/MiniCPM-o-4_5-gguf`.
- Never download into, modify, clean, or delete anything in that model directory.
- Use the F16 LLM, vision, audio, TTS, projector, and Token2Wav components.
- CoreML/ANE is the default vision backend; Metal is supported through
  `--vision-backend metal`. LLM and TTS use Metal layers.
- The LLM context is 8192 tokens. Turn-aware sliding-window thresholds are
  7000 high-water and 5000 low-water.
- Validate all required model components, FFmpeg, devices, and permissions
  before expensive model allocation whenever possible.
- This is macOS-specific: camera capture uses FFmpeg's AVFoundation input and
  microphone/playback use miniaudio. Terminal camera and microphone permissions
  are expected.

## Current architecture and invariants

- Keep the runner native C++ and linked directly to `omni` and `llama-common`.
  Do not introduce an HTTP server or Python runtime.
- Use llama.cpp-omni's asynchronous duplex session API, not ordinary
  llama.cpp text-generation slots.
- `num_streams` is currently 1. Future streams should each own an independent
  `StreamWorker` and `omni_context`; do not multiplex independent cameras into
  the current context.
- Allow at most two outstanding observations. If inference is behind, reject
  the scheduled tick and retain the latest capture rather than queueing stale
  work.
- A scheduled iteration ID is one-based and advances on every 1 Hz tick,
  including rejected ticks. Propagate it through encoder, LLM, TTS,
  Token2Wav, PCM callbacks, playback, and traces.
- Camera input runs at a supported AVFoundation rate (currently 30 FPS), then
  FFmpeg filters it to 1 FPS and scales to 640x360 MJPEG over stdout. Retain
  only the latest complete JPEG in memory.
- A duplex `OmniDuplexFrame` currently contains one selected JPEG plus one
  second of 16 kHz mono audio. The physical camera observes many frames, but
  the current runner/model integration selects one image per one-second model
  observation. Do not describe that as multiple model image frames unless the
  engine and model input representation are deliberately extended.
- Vision and audio encoding for the same observation run in parallel. Their
  enclosing wall time is approximately `max(vision, audio)`, not their sum.
- TTS and Token2Wav have independent workers and may process different
  iterations concurrently; keep them as distinct pipeline stages.
- TTS PCM is resampled/routed directly into an in-memory 24 kHz playback queue.
  The queue is capped at 30 seconds; drop oldest tagged PCM on overflow and
  emit a warning plus a trace drop event.

## Disk and API compatibility

- Call `omni_set_runtime_artifacts_enabled(ctx, false)` in the runner.
- Artifact-disabled execution must not create output directories, WAV/JPEG
  files, tokens, embeddings, debug text, merged audio, timing files,
  completion flags, logs, archives, or temporary media. Prevent artifacts at
  their source; do not add cleanup-afterward as the primary strategy.
- The only intentional runtime file is an explicitly requested Perfetto trace.
- Prefer in-memory JPEG and WAV buffers. In-memory data takes precedence over
  legacy paths in `OmniDuplexFrame`.
- Preserve llama.cpp-omni path-based inputs and default artifact-producing
  behavior for existing CLI/server consumers.
- Preserve the original PCM callback API. Iteration-aware callbacks and trace
  hooks must remain additive/backward-compatible.
- Existing workspace files and user traces such as `monitor.json` are user
  data. Do not overwrite, remove, or clean them unless explicitly requested.

## Perfetto trace contract

Tracing is opt-in through `--perfetto-trace PATH`. It emits dependency-free
Chrome Trace JSON that Perfetto can load directly, is buffered in bounded
memory, is capped at 64 MiB, uses no temporary file, and finalizes during
graceful shutdown (also best-effort during exception unwinding).

There are exactly six logical tracks:

1. `Observation capture`: one interval enclosing parallel camera/microphone
   capture for iteration N.
2. `Multimodal encoder`: one interval from before launching vision/audio work
   through both joining. Never emit vision/audio component tracks.
3. `LLM prefill + decode`: one interval combining sequential embedding prefill
   and autoregressive decode. Never emit prefill/decode/projector subtracks.
4. `TTS`: one interval per real TTS chunk, including same-chunk synchronous
   preparation and inference.
5. `Token2Wav`: one interval per token window, combining same-window internal
   processing. Never emit Token2Wav-internal component tracks.
6. `Audio playback`: actual miniaudio playback intervals, coalesced by
   iteration.

Use one of six colors based on iteration and keep the same iteration color on
every track. Label accepted work `iteration N`. Label rejected observation
windows and playback-overflow work `dropped` in gray, with a reason argument.
Multiple same-iteration intervals are allowed on TTS, Token2Wav, and playback.

Known trace caveat: current silent/listen results create tiny TTS spans of
roughly 0.03-0.08 ms even though no TTS model inference occurred. Future trace
work should start the TTS span only when real synthesis work begins. Empty
Token2Wav and playback tracks are correct when the model remains silent.

## Shutdown and diagnostics

- SIGINT/SIGTERM should stop capture, finish submitted inference, close the
  duplex session, drain TTS and queued playback, release audio/model resources,
  and finalize an enabled trace.
- FFmpeg can print `Immediate exit requested` while its pipe is intentionally
  terminated during Ctrl-C. This is expected if the runner subsequently exits
  0 and writes/finalizes the trace; do not misdiagnose that message alone as a
  failed shutdown.
- Report camera index, permission, microphone, FFmpeg, missing-model, and
  inference failures clearly.
- The MacBook camera used during development supports 640x480 and 1280x720 at
  15 or 30 FPS. Do not request the unsupported 10 FPS AVFoundation input mode.

## Development preferences

- Keep changes minimal and use llama.cpp-omni as much as practical. Modifying
  the nested repository is explicitly allowed for this project when necessary.
- Prefer small, dependency-free implementations. Installing a useful tool is
  allowed when it materially improves the result, but avoid adding runtime
  dependencies without need.
- Preserve backward compatibility and the artifact-free default runner path.
- Be precise about concurrency: distinguish work that truly overlaps across
  iterations from parallel or sequential components operating on the same
  iteration. Collapse same-iteration components into the enclosing logical
  stage when discussing or tracing the pipeline.
- Communicate concisely and lead with the conclusion. For pipeline explanations,
  prefer a clean table or numbered stage list over a fragile diagram. Support
  claims about overlap, synchronization, shutdown, or performance with code or
  trace evidence.
- The user understands the architecture and has supplied detailed designs;
  preserve their explicit stage definitions and semantics rather than silently
  substituting a different abstraction.
- Do not create commits, push branches, submit PRs, or modify the model assets
  without explicit authorization.

## Build and verification

Configure and build from the repository root:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vlm-stream-runner llama-omni-test-duplex vlm-stream-runner-tests -j 8
ctest --test-dir build --output-on-failure
```

For model-backed in-memory/artifact verification, use the bundled duplex test
assets and the local model. A known passing shape is nine frames with
`--memory-io --no-artifacts`; verify the designated output directory remains
absent or empty. Also retain coverage for legacy path input.

Before handing off relevant changes:

- Run `git diff --check` in both the root and `llama.cpp-omni` repositories.
- Build all affected targets and run the unit suite.
- For engine changes, run the model-backed duplex test when feasible.
- For capture/shutdown/trace changes, perform a short live camera/microphone run
  and Ctrl-C when permissions permit, then parse the resulting JSON and inspect
  all six track definitions.
- Check accepted/dropped labels, color consistency, iteration propagation,
  stage ordering, overlap, queue bounds, and `omitted_events`.
- Do not claim the two-hour soak test has passed unless it was actually run.

Current healthy short-run baseline (not a hard performance requirement): a
25-iteration CoreML trace had no dropped or omitted events, encoder mean about
345 ms (max 365 ms), LLM mean about 183 ms (max 194 ms), and at least 423 ms
of headroom before the next 1 Hz deadline. Some LLM latency growth as context
accumulates is expected; verify long-run sliding-window behavior with a soak
test rather than extrapolating from this short sample.
