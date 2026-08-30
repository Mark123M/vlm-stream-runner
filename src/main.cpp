#include "runner_support.hpp"

#include "common/common.h"
#include "tools/omni/omni.h"

#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define ma_atomic_global_lock vlm_runner_ma_atomic_global_lock
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace {

constexpr int kNumStreams = 1;
constexpr int kMaxOutstandingObservations = 2;
volatile std::sig_atomic_t g_stop_requested = 0;

void signal_handler(int) {
    g_stop_requested = 1;
}

struct Options {
    fs::path model_dir = "/Users/markma/Documents/models/MiniCPM-o-4_5-gguf";
    int camera_index = 0;
    fs::path ref_audio = fs::path(VLM_STREAM_RUNNER_SOURCE_DIR) /
        "llama.cpp-omni/tools/omni/assets/default_ref_audio/default_ref_audio.wav";
    std::string vision_backend = "coreml";
    std::optional<fs::path> perfetto_trace;
    std::int64_t dump_iterations = 0;
};

struct ModelPaths {
    fs::path llm;
    fs::path vision;
    fs::path audio;
    fs::path tts;
    fs::path projector;
    fs::path coreml;
    fs::path token2wav;
};

void usage(const char * program) {
    std::printf(
        "Usage: %s [options]\n\n"
        "  --model-dir PATH              model directory (default: %s)\n"
        "  --camera-index N              AVFoundation camera index (default: 0)\n"
        "  --ref-audio PATH              reference voice WAV\n"
        "  --vision-backend coreml|metal vision encoder backend (default: coreml)\n"
        "  --perfetto-trace PATH          write a bounded six-track Perfetto JSON trace\n"
        "  --dump K                       dump artifacts for scheduled iterations 1 through K\n"
        "  -h, --help                    show this help\n",
        program, Options{}.model_dir.c_str());
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&]() -> const char * {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model-dir") {
            options.model_dir = require_value();
        } else if (arg == "--camera-index") {
            const std::string value = require_value();
            std::size_t consumed = 0;
            options.camera_index = std::stoi(value, &consumed);
            if (consumed != value.size() || options.camera_index < 0) {
                throw std::runtime_error("--camera-index must be a non-negative integer");
            }
        } else if (arg == "--ref-audio") {
            options.ref_audio = require_value();
        } else if (arg == "--vision-backend") {
            options.vision_backend = require_value();
            if (options.vision_backend != "coreml" && options.vision_backend != "metal") {
                throw std::runtime_error("--vision-backend must be coreml or metal");
            }
        } else if (arg == "--perfetto-trace") {
            options.perfetto_trace = fs::path(require_value());
            if (options.perfetto_trace->empty()) {
                throw std::runtime_error("--perfetto-trace path must not be empty");
            }
        } else if (arg == "--dump") {
            const std::string value = require_value();
            std::size_t consumed = 0;
            try {
                options.dump_iterations = std::stoll(value, &consumed);
            } catch (const std::exception &) {
                throw std::runtime_error("--dump must be a positive integer");
            }
            if (consumed != value.size() || options.dump_iterations <= 0) {
                throw std::runtime_error("--dump must be a positive integer");
            }
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    return options;
}

ModelPaths model_paths(const fs::path & root) {
    return {
        root / "MiniCPM-o-4_5-F16.gguf",
        root / "vision/MiniCPM-o-4_5-vision-F16.gguf",
        root / "audio/MiniCPM-o-4_5-audio-F16.gguf",
        root / "tts/MiniCPM-o-4_5-tts-F16.gguf",
        root / "tts/MiniCPM-o-4_5-projector-F16.gguf",
        root / "vision/coreml_minicpmo45_vit_all_f16.mlmodelc",
        root / "token2wav-gguf",
    };
}

void require_regular_file(const fs::path & path, const char * description) {
    std::error_code error;
    if (!fs::is_regular_file(path, error)) {
        throw std::runtime_error(std::string("missing ") + description + ": " + path.string());
    }
}

void validate_models(const Options & options, const ModelPaths & paths) {
    require_regular_file(paths.llm, "F16 LLM");
    require_regular_file(paths.vision, "F16 vision encoder");
    require_regular_file(paths.audio, "F16 audio encoder");
    require_regular_file(paths.tts, "F16 TTS model");
    require_regular_file(paths.projector, "F16 TTS projector");
    require_regular_file(options.ref_audio, "reference audio");
    for (const char * component : {"hifigan2.gguf", "flow_extra.gguf", "prompt_cache.gguf",
                                   "encoder.gguf", "flow_matching.gguf"}) {
        require_regular_file(paths.token2wav / component, "Token2Wav component");
    }
    if (options.vision_backend == "coreml") {
        std::error_code error;
        if (!fs::is_directory(paths.coreml, error)) {
            throw std::runtime_error("missing compiled CoreML vision model: " + paths.coreml.string());
        }
    }
}

std::optional<std::string> find_executable(const std::string & name) {
    if (name.find('/') != std::string::npos) {
        return access(name.c_str(), X_OK) == 0 ? std::optional<std::string>(name) : std::nullopt;
    }
    const char * path_env = std::getenv("PATH");
    if (path_env == nullptr) return std::nullopt;
    std::string path(path_env);
    std::size_t begin = 0;
    while (begin <= path.size()) {
        const std::size_t end = path.find(':', begin);
        fs::path candidate = path.substr(begin, end == std::string::npos ? end : end - begin);
        candidate /= name;
        if (access(candidate.c_str(), X_OK) == 0) return candidate.string();
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return std::nullopt;
}

class CameraCapture {
public:
    CameraCapture(std::string ffmpeg, int camera_index, vlm::LatestFrame & latest)
        : ffmpeg_(std::move(ffmpeg)), camera_index_(camera_index), latest_(latest) {}

    ~CameraCapture() { stop(); }

    void start() {
        int pipe_fds[2];
        if (pipe(pipe_fds) != 0) throw std::runtime_error("unable to create FFmpeg pipe");

        child_pid_ = fork();
        if (child_pid_ < 0) {
            close(pipe_fds[0]);
            close(pipe_fds[1]);
            throw std::runtime_error("unable to fork FFmpeg camera process");
        }
        if (child_pid_ == 0) {
            dup2(pipe_fds[1], STDOUT_FILENO);
            close(pipe_fds[0]);
            close(pipe_fds[1]);
            const std::string input = std::to_string(camera_index_) + ":none";
            execl(ffmpeg_.c_str(), "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                  "-f", "avfoundation", "-framerate", "30", "-pixel_format", "nv12",
                  "-i", input.c_str(),
                  "-vf", "fps=1,scale=640:360", "-q:v", "4", "-f", "image2pipe",
                  "-vcodec", "mjpeg", "pipe:1", static_cast<char *>(nullptr));
            _exit(127);
        }

        close(pipe_fds[1]);
        read_fd_ = pipe_fds[0];
        running_.store(true);
        reader_ = std::thread([this] { read_loop(); });
    }

    bool wait_for_first_frame(std::chrono::seconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline && running_.load() && !g_stop_requested) {
            if (latest_.generation() != 0) return true;
            std::this_thread::sleep_for(50ms);
        }
        return latest_.generation() != 0;
    }

    void stop() {
        running_.store(false);
        if (child_pid_ > 0) kill(child_pid_, SIGTERM);
        if (reader_.joinable()) reader_.join();
        if (read_fd_ >= 0) {
            close(read_fd_);
            read_fd_ = -1;
        }
        if (child_pid_ > 0) {
            int status = 0;
            waitpid(child_pid_, &status, 0);
            child_pid_ = -1;
        }
    }

    std::string error() const {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return error_;
    }

private:
    void read_loop() {
        vlm::JpegFramer framer;
        std::vector<std::uint8_t> chunk(64 * 1024);
        while (running_.load()) {
            const ssize_t bytes = read(read_fd_, chunk.data(), chunk.size());
            if (bytes > 0) {
                for (auto & frame : framer.feed(chunk.data(), static_cast<std::size_t>(bytes))) {
                    latest_.set(std::move(frame));
                }
            } else if (bytes == 0) {
                set_error("FFmpeg camera process closed its stream; check camera index and permission");
                break;
            } else if (errno != EINTR) {
                set_error(std::string("camera pipe read failed: ") + std::strerror(errno));
                break;
            }
        }
        running_.store(false);
    }

    void set_error(std::string message) {
        std::lock_guard<std::mutex> lock(error_mutex_);
        error_ = std::move(message);
    }

    std::string ffmpeg_;
    int camera_index_;
    vlm::LatestFrame & latest_;
    pid_t child_pid_ = -1;
    int read_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread reader_;
    mutable std::mutex error_mutex_;
    std::string error_;
};

class AudioDevices {
public:
    AudioDevices(vlm::CapturePcmBuffer & capture, vlm::BoundedPcmQueue & playback,
                 vlm::PerfettoTrace * trace)
        : capture_(capture), playback_(playback), trace_(trace) {}

    ~AudioDevices() {
        stop_capture();
        stop_playback();
    }

    void start() {
        ma_device_config capture_config = ma_device_config_init(ma_device_type_capture);
        capture_config.capture.format = ma_format_f32;
        capture_config.capture.channels = 1;
        capture_config.sampleRate = vlm::kMicSampleRate;
        capture_config.dataCallback = capture_callback;
        capture_config.pUserData = this;
        ma_result result = ma_device_init(nullptr, &capture_config, &capture_device_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error(std::string("microphone initialization failed: ") +
                                     ma_result_description(result));
        }
        capture_initialized_ = true;

        ma_device_config playback_config = ma_device_config_init(ma_device_type_playback);
        playback_config.playback.format = ma_format_f32;
        playback_config.playback.channels = 1;
        playback_config.sampleRate = vlm::kPlaybackSampleRate;
        playback_config.dataCallback = playback_callback;
        playback_config.pUserData = this;
        result = ma_device_init(nullptr, &playback_config, &playback_device_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error(std::string("speaker initialization failed: ") +
                                     ma_result_description(result));
        }
        playback_initialized_ = true;

        result = ma_device_start(&capture_device_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error(std::string("microphone start failed (check permission): ") +
                                     ma_result_description(result));
        }
        capture_started_ = true;
        result = ma_device_start(&playback_device_);
        if (result != MA_SUCCESS) {
            throw std::runtime_error(std::string("speaker start failed: ") + ma_result_description(result));
        }
        playback_started_ = true;
    }

    void stop_capture() {
        if (capture_started_) {
            ma_device_stop(&capture_device_);
            capture_started_ = false;
        }
        if (capture_initialized_) {
            ma_device_uninit(&capture_device_);
            capture_initialized_ = false;
        }
    }

    void stop_playback() {
        if (playback_started_) {
            ma_device_stop(&playback_device_);
            playback_started_ = false;
        }
        flush_playback_trace();
        if (playback_initialized_) {
            ma_device_uninit(&playback_device_);
            playback_initialized_ = false;
        }
    }

private:
    static void capture_callback(ma_device * device, void *, const void * input, ma_uint32 frames) {
        auto * self = static_cast<AudioDevices *>(device->pUserData);
        if (input != nullptr) self->capture_.push(static_cast<const float *>(input), frames);
    }

    static void playback_callback(ma_device * device, void * output, const void *, ma_uint32 frames) {
        auto * self = static_cast<AudioDevices *>(device->pUserData);
        auto * samples = static_cast<float *>(output);
        std::fill(samples, samples + frames, 0.0f);
        const auto callback_start = vlm::PerfettoTrace::Clock::now();
        std::size_t offset = 0;
        while (offset < frames) {
            std::int64_t iteration = -1;
            const std::size_t copied = self->playback_.pull_segment(
                samples + offset, frames - offset, &iteration);
            if (copied == 0) break;
            const auto begin = callback_start + std::chrono::duration_cast<vlm::PerfettoTrace::Clock::duration>(
                std::chrono::duration<double>(static_cast<double>(offset) / vlm::kPlaybackSampleRate));
            const auto end = callback_start + std::chrono::duration_cast<vlm::PerfettoTrace::Clock::duration>(
                std::chrono::duration<double>(static_cast<double>(offset + copied) / vlm::kPlaybackSampleRate));
            self->record_playback(iteration, begin, end);
            offset += copied;
        }
        if (offset < frames) self->flush_playback_trace();
    }

    void record_playback(std::int64_t iteration, vlm::PerfettoTrace::Clock::time_point begin,
                         vlm::PerfettoTrace::Clock::time_point end) {
        if (trace_ == nullptr || iteration <= 0) return;
        if (playback_trace_active_ && playback_trace_iteration_ == iteration &&
            begin <= playback_trace_end_ + 2ms) {
            playback_trace_end_ = std::max(playback_trace_end_, end);
            return;
        }
        flush_playback_trace();
        playback_trace_active_ = true;
        playback_trace_iteration_ = iteration;
        playback_trace_begin_ = begin;
        playback_trace_end_ = end;
    }

    void flush_playback_trace() {
        if (!playback_trace_active_ || trace_ == nullptr) return;
        trace_->record(vlm::PipelineStage::Playback, playback_trace_iteration_,
                       playback_trace_begin_, playback_trace_end_);
        playback_trace_active_ = false;
    }

    vlm::CapturePcmBuffer & capture_;
    vlm::BoundedPcmQueue & playback_;
    vlm::PerfettoTrace * trace_;
    ma_device capture_device_{};
    ma_device playback_device_{};
    bool capture_initialized_ = false;
    bool playback_initialized_ = false;
    bool capture_started_ = false;
    bool playback_started_ = false;
    bool playback_trace_active_ = false;
    std::int64_t playback_trace_iteration_ = -1;
    vlm::PerfettoTrace::Clock::time_point playback_trace_begin_{};
    vlm::PerfettoTrace::Clock::time_point playback_trace_end_{};
};

class StreamWorker {
public:
    enum class SubmitResult {
        Submitted,
        Backpressure,
        NoCameraFrame,
        PushFailed,
    };

    StreamWorker(omni_context * context, vlm::LatestFrame & camera,
                 vlm::CapturePcmBuffer & microphone, vlm::ArtifactDumper * dump)
        : context_(context), camera_(camera), microphone_(microphone), dump_(dump) {}

    SubmitResult submit_latest(std::int64_t iteration) {
        if (outstanding_.load() >= kMaxOutstandingObservations) return SubmitResult::Backpressure;
        OmniDuplexFrame frame;
        if (!camera_.copy(frame.img_bytes)) return SubmitResult::NoCameraFrame;
        frame.aud_bytes = vlm::make_wav_pcm16(microphone_.take_latest(vlm::kOneSecondMicSamples));
        frame.user_seq = iteration;
        outstanding_.fetch_add(1);
        if (omni_duplex_push_frame(context_, frame) < 0) {
            outstanding_.fetch_sub(1);
            return SubmitResult::PushFailed;
        }
        if (dump_ != nullptr &&
            !dump_->record_observation(iteration, frame.img_bytes, frame.aud_bytes)) {
            warn_dump_error();
        }
        return SubmitResult::Submitted;
    }

    void start_consumer() {
        consumer_ = std::thread([this] {
            while (!producer_done_.load() || outstanding_.load() > 0) {
                OmniDuplexFrameResult result;
                if (!omni_duplex_wait_next_frame(context_, &result, 250)) continue;
                outstanding_.fetch_sub(1);
                if (!result.ok) {
                    std::fprintf(stderr, "inference failed for observation %lld\n",
                                 static_cast<long long>(result.user_seq));
                } else {
                    if (dump_ != nullptr && !dump_->record_llm(result.user_seq, result.text)) {
                        warn_dump_error();
                    }
                    if (result.is_speak) {
                        std::printf("[observation %lld] %s\n",
                                    static_cast<long long>(result.user_seq), result.text.c_str());
                        std::fflush(stdout);
                    }
                }
            }
        });
    }

    int outstanding() const { return outstanding_.load(); }

    void finish() {
        producer_done_.store(true);
        if (consumer_.joinable()) consumer_.join();
    }

private:
    void warn_dump_error() {
        std::string error;
        if (dump_ != nullptr && dump_->take_error(error)) {
            std::fprintf(stderr, "warning: artifact dump disabled after an error: %s\n",
                         error.c_str());
        }
    }

    omni_context * context_;
    vlm::LatestFrame & camera_;
    vlm::CapturePcmBuffer & microphone_;
    vlm::ArtifactDumper * dump_;
    std::atomic<int> outstanding_{0};
    std::atomic<bool> producer_done_{false};
    std::thread consumer_;
};

using OmniContextPtr = std::unique_ptr<omni_context, decltype(&omni_free)>;

struct ModelRuntime {
    std::unique_ptr<common_params> params;
    OmniContextPtr context;
};

ModelRuntime initialize_model(const Options & options, const ModelPaths & paths,
                              vlm::BoundedPcmQueue & playback,
                              vlm::PerfettoTrace * trace,
                              vlm::ArtifactDumper * dump) {
    auto params = std::make_unique<common_params>();
    params->model.path = paths.llm.string();
    params->vpm_model = paths.vision.string();
    params->apm_model = paths.audio.string();
    params->tts_model = paths.tts.string();
    params->projector_model = paths.projector.string();
    params->n_ctx = 8192;
    params->n_parallel = kNumStreams;
    params->n_gpu_layers = 99;
    if (options.vision_backend == "coreml") {
        params->vision_coreml_model_path = paths.coreml.string();
    }

    common_init();
    OmniContextPtr context(
        omni_init(params.get(), 2, true, paths.tts.parent_path().string(),
                  99, "gpu", true, nullptr, nullptr, ""),
        &omni_free);
    if (!context) throw std::runtime_error("model initialization failed");

    context->async = true;
    context->ref_audio_path = options.ref_audio.string();
    context->sliding_window_config.mode = "turn";
    context->sliding_window_config.high_water_tokens = 7000;
    context->sliding_window_config.low_water_tokens = 5000;
    const auto wrapped = vlm::wrap_duplex_system_prompt(vlm::monitor_prompt());
    context->omni_voice_clone_prompt = wrapped.first;
    context->omni_assistant_prompt = wrapped.second;
    context->audio_voice_clone_prompt = wrapped.first;
    context->audio_assistant_prompt = wrapped.second;
    if (trace != nullptr) {
        context->duplex_trace_cb = [trace](const OmniDuplexTraceEvent & event) {
            vlm::PipelineStage stage;
            switch (event.stage) {
                case OmniDuplexTraceStage::Encoder:   stage = vlm::PipelineStage::Encoder; break;
                case OmniDuplexTraceStage::Llm:       stage = vlm::PipelineStage::Llm; break;
                case OmniDuplexTraceStage::Tts:       stage = vlm::PipelineStage::Tts; break;
                case OmniDuplexTraceStage::Token2Wav: stage = vlm::PipelineStage::Token2Wav; break;
                default: return;
            }
            trace->record(stage, event.user_seq, event.begin, event.end);
        };
    }
    context->audio_output_cb_ex = [&playback, trace, dump](const float * samples, int count,
                                                           int sample_rate, bool,
                                                           std::int64_t iteration) {
        if (dump != nullptr &&
            !dump->append_tts(iteration, samples, static_cast<std::size_t>(std::max(0, count)),
                              sample_rate)) {
            std::string error;
            if (dump->take_error(error)) {
                std::fprintf(stderr, "warning: artifact dump disabled after an error: %s\n",
                             error.c_str());
            }
        }
        auto converted = vlm::resample_linear(samples, static_cast<std::size_t>(std::max(0, count)),
                                              sample_rate, vlm::kPlaybackSampleRate);
        const std::size_t dropped = playback.enqueue(converted.data(), converted.size(), iteration);
        if (dropped != 0) {
            std::fprintf(stderr, "warning: TTS playback queue overflow; dropped %.2f seconds of oldest audio\n",
                         static_cast<double>(dropped) / vlm::kPlaybackSampleRate);
            if (trace != nullptr) {
                const auto begin = vlm::PerfettoTrace::Clock::now();
                const auto end = begin + std::chrono::duration_cast<vlm::PerfettoTrace::Clock::duration>(
                    std::chrono::duration<double>(static_cast<double>(dropped) /
                                                  vlm::kPlaybackSampleRate));
                trace->record_dropped(vlm::PipelineStage::Playback, iteration, begin, end,
                                      "playback_queue_overflow");
            }
        }
    };
    if (!omni_set_runtime_artifacts_enabled(context.get(), false)) {
        throw std::runtime_error("the selected Token2Wav backend cannot run artifact-free");
    }
    return {std::move(params), std::move(context)};
}

} // namespace

int main(int argc, char ** argv) {
    static_assert(kNumStreams == 1, "this runner currently provisions one isolated stream context");
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    try {
        const Options options = parse_options(argc, argv);
        const ModelPaths paths = model_paths(options.model_dir);
        validate_models(options, paths);
        const auto ffmpeg = find_executable("ffmpeg");
        if (!ffmpeg) throw std::runtime_error("FFmpeg was not found in PATH");

        std::unique_ptr<vlm::PerfettoTrace> trace;
        if (options.perfetto_trace) {
            trace = std::make_unique<vlm::PerfettoTrace>(options.perfetto_trace->string());
        }
        std::unique_ptr<vlm::ArtifactDumper> dump;
        if (options.dump_iterations > 0) {
            dump = std::make_unique<vlm::ArtifactDumper>("dump", options.dump_iterations);
            std::printf("Dumping artifacts for scheduled iterations 1 through %lld to %s\n",
                        static_cast<long long>(options.dump_iterations), dump->path().c_str());
        }

        vlm::LatestFrame latest_camera_frame;
        vlm::CapturePcmBuffer microphone;
        vlm::BoundedPcmQueue playback(30 * vlm::kPlaybackSampleRate);
        AudioDevices audio(microphone, playback, trace.get());
        CameraCapture camera(*ffmpeg, options.camera_index, latest_camera_frame);

        // Permission and device failures happen before allocating the 25+ GB F16 model.
        audio.start();
        camera.start();
        if (!camera.wait_for_first_frame(10s)) {
            if (g_stop_requested) {
                std::printf("Interrupted before model initialization; releasing camera and audio.\n");
                return 130;
            }
            const std::string detail = camera.error();
            throw std::runtime_error(detail.empty()
                ? "no camera frame received in 10 seconds; check camera permission and index"
                : detail);
        }

        std::printf("Camera and microphone ready. Loading F16 duplex model...\n");
        std::fflush(stdout);
        ModelRuntime model = initialize_model(options, paths, playback, trace.get(), dump.get());
        if (g_stop_requested) {
            std::printf("Interrupted during model initialization; releasing model, camera, and audio.\n");
            return 130;
        }
        if (!omni_duplex_session_begin(model.context.get(), options.ref_audio.string(), "")) {
            throw std::runtime_error("duplex session initialization failed");
        }
        if (g_stop_requested) {
            std::printf("Interrupted during duplex session initialization; releasing resources.\n");
            camera.stop();
            audio.stop_capture();
            omni_duplex_session_end(model.context.get());
            model.context.reset();
            playback.close();
            audio.stop_playback();
            return 130;
        }

        StreamWorker worker(model.context.get(), latest_camera_frame, microphone, dump.get());
        worker.start_consumer();
        std::printf("Monitoring at one observation per second. Press Ctrl-C to stop.\n");

        std::int64_t iteration = 1;
        auto observation_end = std::chrono::steady_clock::now();
        auto next_tick = observation_end + 1s;
        while (!g_stop_requested) {
            const auto submit_result = worker.submit_latest(iteration);
            if (trace != nullptr) {
                const auto observation_begin = observation_end - 1s;
                if (submit_result == StreamWorker::SubmitResult::Submitted) {
                    trace->record(vlm::PipelineStage::Observation, iteration,
                                  observation_begin, observation_end);
                } else {
                    const char * reason = "push_failed";
                    if (submit_result == StreamWorker::SubmitResult::Backpressure) {
                        reason = "inference_backpressure";
                    } else if (submit_result == StreamWorker::SubmitResult::NoCameraFrame) {
                        reason = "no_camera_frame";
                    }
                    trace->record_dropped(vlm::PipelineStage::Observation, iteration,
                                          observation_begin, observation_end, reason);
                }
            }
            if (submit_result == StreamWorker::SubmitResult::Backpressure) {
                std::fprintf(stderr, "warning: inference is behind; dropping capture tick and retaining latest frame\n");
            } else if (submit_result == StreamWorker::SubmitResult::NoCameraFrame) {
                std::fprintf(stderr, "warning: no camera frame available; dropping capture tick\n");
            } else if (submit_result == StreamWorker::SubmitResult::PushFailed) {
                std::fprintf(stderr, "warning: duplex frame submission failed; dropping capture tick\n");
            }
            while (!g_stop_requested && std::chrono::steady_clock::now() < next_tick) {
                std::this_thread::sleep_for(25ms);
            }
            observation_end = next_tick;
            next_tick += 1s;
            ++iteration;
        }

        std::printf("Stopping capture and draining inference/TTS...\n");
        camera.stop();
        audio.stop_capture();
        worker.finish();
        omni_duplex_session_end(model.context.get());
        if (!omni_duplex_drain_tts_audio(model.context.get(), 120000, 1000)) {
            std::fprintf(stderr, "warning: timed out while draining TTS generation\n");
        }
        model.context.reset();

        const auto playback_deadline = std::chrono::steady_clock::now() + 35s;
        while (!playback.empty() && std::chrono::steady_clock::now() < playback_deadline) {
            std::this_thread::sleep_for(50ms);
        }
        playback.close();
        audio.stop_playback();
        if (trace != nullptr) {
            std::string trace_error;
            if (!trace->finalize(&trace_error)) throw std::runtime_error(trace_error);
            std::printf("Perfetto trace written to %s\n", trace->path().c_str());
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "vlm-stream-runner: %s\n", error.what());
        return 1;
    }
}
