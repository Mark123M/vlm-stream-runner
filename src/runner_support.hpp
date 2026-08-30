#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vlm {

constexpr int kMicSampleRate = 16000;
constexpr int kPlaybackSampleRate = 24000;
constexpr std::size_t kOneSecondMicSamples = kMicSampleRate;

enum class PipelineStage : std::uint8_t {
    Observation = 1,
    Encoder = 2,
    Llm = 3,
    Tts = 4,
    Token2Wav = 5,
    Playback = 6,
};

class PerfettoTrace {
public:
    using Clock = std::chrono::steady_clock;
    static constexpr std::size_t kDefaultMaxBytes = 64 * 1024 * 1024;

    explicit PerfettoTrace(std::string path, std::size_t max_bytes = kDefaultMaxBytes);
    ~PerfettoTrace();

    PerfettoTrace(const PerfettoTrace &) = delete;
    PerfettoTrace & operator=(const PerfettoTrace &) = delete;

    void record(PipelineStage stage, std::int64_t iteration,
                Clock::time_point begin, Clock::time_point end);
    void record_dropped(PipelineStage stage, std::int64_t iteration,
                        Clock::time_point begin, Clock::time_point end,
                        std::string reason);
    bool finalize(std::string * error = nullptr);
    const std::string & path() const { return path_; }

private:
    struct Event {
        PipelineStage stage;
        std::int64_t iteration;
        std::int64_t begin_us;
        std::int64_t end_us;
        bool dropped;
        std::string reason;
    };

    void add(Event event);

    std::string path_;
    std::size_t max_bytes_;
    std::size_t max_events_;
    mutable std::mutex mutex_;
    std::vector<Event> events_;
    std::uint64_t omitted_events_ = 0;
    bool finalized_ = false;
};

class ArtifactDumper {
public:
    ArtifactDumper(std::string path, std::int64_t max_iteration);

    ArtifactDumper(const ArtifactDumper &) = delete;
    ArtifactDumper & operator=(const ArtifactDumper &) = delete;

    bool record_observation(std::int64_t iteration,
                            const std::vector<std::uint8_t> & image,
                            const std::vector<std::uint8_t> & audio);
    bool record_llm(std::int64_t iteration, const std::string & text);
    bool append_tts(std::int64_t iteration, const float * samples,
                    std::size_t count, int sample_rate);
    bool take_error(std::string & error);
    const std::string & path() const { return path_; }

private:
    struct TtsFile {
        int sample_rate = 0;
        std::uint64_t samples = 0;
    };

    bool includes(std::int64_t iteration) const;
    std::string iteration_path(std::int64_t iteration) const;
    bool ensure_iteration_directory(std::int64_t iteration);
    bool write_file(const std::string & path, const std::uint8_t * data, std::size_t size);
    bool fail(std::string error);

    std::string path_;
    std::int64_t max_iteration_;
    std::mutex mutex_;
    std::unordered_map<std::int64_t, TtsFile> tts_files_;
    std::string error_;
    bool error_reported_ = false;
};

class JpegFramer {
public:
    explicit JpegFramer(std::size_t max_frame_bytes = 8 * 1024 * 1024);
    std::vector<std::vector<std::uint8_t>> feed(const std::uint8_t * data, std::size_t size);
    std::size_t discarded_frames() const { return discarded_frames_; }

private:
    std::vector<std::uint8_t> buffer_;
    std::size_t max_frame_bytes_;
    std::size_t discarded_frames_ = 0;
};

class LatestFrame {
public:
    void set(std::vector<std::uint8_t> frame);
    bool copy(std::vector<std::uint8_t> & frame, std::uint64_t * generation = nullptr) const;
    std::uint64_t generation() const;

private:
    mutable std::mutex mutex_;
    std::vector<std::uint8_t> frame_;
    std::uint64_t generation_ = 0;
};

class CapturePcmBuffer {
public:
    explicit CapturePcmBuffer(std::size_t capacity_samples = 2 * kMicSampleRate);
    void push(const float * samples, std::size_t count);
    // Returns the newest chunk, zero-padding its beginning if capture has not
    // produced a full second yet, and drops all older capture backlog.
    std::vector<float> take_latest(std::size_t count);
    std::size_t size() const;

private:
    mutable std::mutex mutex_;
    std::deque<float> samples_;
    std::size_t capacity_samples_;
};

class BoundedPcmQueue {
public:
    explicit BoundedPcmQueue(std::size_t capacity_samples = 30 * kPlaybackSampleRate);
    // Returns how many oldest samples were dropped.
    std::size_t enqueue(const float * samples, std::size_t count,
                        std::int64_t iteration = -1);
    // Pulls at most one contiguous iteration segment and does not zero-fill.
    std::size_t pull_segment(float * output, std::size_t count,
                             std::int64_t * iteration = nullptr);
    std::size_t pull(float * output, std::size_t count);
    void close();
    bool closed() const;
    bool empty() const;
    std::size_t size() const;

private:
    struct Segment {
        std::vector<float> samples;
        std::size_t offset = 0;
        std::int64_t iteration = -1;
    };

    void drop_front(std::size_t count);

    mutable std::mutex mutex_;
    std::deque<Segment> segments_;
    std::size_t size_samples_ = 0;
    std::size_t capacity_samples_;
    bool closed_ = false;
};

std::vector<float> resample_linear(const float * samples, std::size_t count,
                                   int source_rate, int destination_rate);
std::vector<std::uint8_t> make_wav_pcm16(const std::vector<float> & samples,
                                         int sample_rate = kMicSampleRate);

const std::string & monitor_prompt();
std::pair<std::string, std::string> wrap_duplex_system_prompt(const std::string & prompt);

} // namespace vlm
