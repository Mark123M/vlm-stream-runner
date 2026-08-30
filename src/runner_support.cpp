#include "runner_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace vlm {
namespace {

namespace fs = std::filesystem;

constexpr const char * kTrackNames[] = {
    "",
    "1 Observation capture",
    "2 Multimodal encoder",
    "3 LLM prefill + decode",
    "4 TTS",
    "5 Token2Wav",
    "6 Audio playback",
};

constexpr const char * kIterationColors[] = {
    "good", "bad", "yellow", "olive", "generic_work", "rail_response",
};

bool is_dump_iteration_name(const fs::path & path) {
    static constexpr const char * kPrefix = "iteration-";
    const std::string name = path.filename().string();
    const std::size_t prefix_size = std::strlen(kPrefix);
    return name.size() > prefix_size && name.compare(0, prefix_size, kPrefix) == 0 &&
           std::all_of(name.begin() + static_cast<std::ptrdiff_t>(prefix_size), name.end(),
                       [](unsigned char c) { return c >= '0' && c <= '9'; });
}

std::string json_escape(const std::string & value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char c : value) {
        switch (c) {
            case '\"': escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (c < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += hex[c >> 4];
                    escaped += hex[c & 0x0f];
                } else {
                    escaped += static_cast<char>(c);
                }
        }
    }
    return escaped;
}

std::int64_t trace_us(PerfettoTrace::Clock::time_point value) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        value.time_since_epoch()).count();
}

const char * track_name(PipelineStage stage) {
    const auto index = static_cast<std::size_t>(stage);
    return index < std::size(kTrackNames) ? kTrackNames[index] : "Unknown";
}

std::string serialize_event(PipelineStage stage, std::int64_t iteration,
                            std::int64_t begin_us, std::int64_t end_us,
                            bool dropped, const std::string & reason) {
    const int tid = static_cast<int>(stage);
    const std::int64_t duration = std::max<std::int64_t>(0, end_us - begin_us);
    std::ostringstream out;
    out << "{\"ph\":\"X\",\"pid\":1,\"tid\":" << tid
        << ",\"ts\":" << begin_us << ",\"dur\":" << duration
        << ",\"cat\":\"pipeline\",\"name\":\"";
    if (dropped) {
        out << "dropped\",\"cname\":\"thread_state_unknown\",\"args\":{\"iteration\":"
            << iteration << ",\"reason\":\"" << json_escape(reason) << "\"}}";
    } else {
        const std::size_t color = iteration > 0
            ? static_cast<std::size_t>((iteration - 1) % std::size(kIterationColors)) : 0;
        out << "iteration " << iteration << "\",\"cname\":\""
            << kIterationColors[color] << "\",\"args\":{\"iteration\":"
            << iteration << "}}";
    }
    return out.str();
}

void append_u16(std::vector<std::uint8_t> & out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void append_u32(std::vector<std::uint8_t> & out, std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
    }
}

std::uint8_t * store_u32(std::uint8_t (&out)[4], std::uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out[shift / 8] = static_cast<std::uint8_t>((value >> shift) & 0xff);
    }
    return out;
}

} // namespace

PerfettoTrace::PerfettoTrace(std::string path, std::size_t max_bytes)
    : path_(std::move(path)),
      max_bytes_(std::max<std::size_t>(4096, max_bytes)),
      max_events_(std::max<std::size_t>(1, max_bytes_ / 256)) {
    if (path_.empty()) throw std::runtime_error("Perfetto trace path is empty");
    std::ofstream probe(path_, std::ios::binary | std::ios::trunc);
    if (!probe) throw std::runtime_error("unable to create Perfetto trace: " + path_);
    events_.reserve(max_events_);
}

PerfettoTrace::~PerfettoTrace() {
    finalize(nullptr);
}

void PerfettoTrace::record(PipelineStage stage, std::int64_t iteration,
                           Clock::time_point begin, Clock::time_point end) {
    add({stage, iteration, trace_us(begin), trace_us(end), false, {}});
}

void PerfettoTrace::record_dropped(PipelineStage stage, std::int64_t iteration,
                                   Clock::time_point begin, Clock::time_point end,
                                   std::string reason) {
    add({stage, iteration, trace_us(begin), trace_us(end), true, std::move(reason)});
}

void PerfettoTrace::add(Event event) {
    if (event.end_us < event.begin_us) event.end_us = event.begin_us;
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_ || events_.size() >= max_events_) {
        ++omitted_events_;
        return;
    }
    events_.push_back(std::move(event));
}

bool PerfettoTrace::finalize(std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) return true;

    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    if (!output) {
        if (error) *error = "unable to write Perfetto trace: " + path_;
        return false;
    }

    const std::string prefix = "{\"traceEvents\":[\n";
    output << prefix;
    std::size_t bytes_written = prefix.size();
    bool first = true;

    auto write_json_event = [&](const std::string & json) {
        const std::size_t delimiter = first ? 0 : 2;
        constexpr std::size_t kSuffixReserve = 256;
        if (bytes_written + delimiter + json.size() + kSuffixReserve > max_bytes_) {
            ++omitted_events_;
            return false;
        }
        if (!first) output << ",\n";
        output << json;
        bytes_written += delimiter + json.size();
        first = false;
        return true;
    };

    for (int tid = 1; tid <= 6; ++tid) {
        std::ostringstream metadata;
        metadata << "{\"ph\":\"M\",\"pid\":1,\"tid\":" << tid
                 << ",\"name\":\"thread_name\",\"args\":{\"name\":\""
                 << track_name(static_cast<PipelineStage>(tid)) << "\"}}";
        write_json_event(metadata.str());
    }
    for (const Event & event : events_) {
        write_json_event(serialize_event(event.stage, event.iteration, event.begin_us,
                                         event.end_us, event.dropped, event.reason));
    }

    std::ostringstream suffix;
    suffix << "\n],\"displayTimeUnit\":\"ms\",\"metadata\":{\"omitted_events\":"
           << omitted_events_ << "}}\n";
    output << suffix.str();
    output.close();
    if (!output) {
        if (error) *error = "failed while writing Perfetto trace: " + path_;
        return false;
    }
    finalized_ = true;
    return true;
}

ArtifactDumper::ArtifactDumper(std::string path, std::int64_t max_iteration)
    : path_(std::move(path)), max_iteration_(max_iteration) {
    if (path_.empty()) throw std::runtime_error("artifact dump path is empty");
    if (max_iteration_ <= 0) throw std::runtime_error("artifact dump count must be positive");

    std::error_code error;
    const bool exists = fs::exists(path_, error);
    if (error) {
        throw std::runtime_error("unable to inspect artifact dump path " + path_ +
                                 ": " + error.message());
    }
    if (exists) {
        if (!fs::is_directory(path_, error) || error) {
            throw std::runtime_error("artifact dump path is not a directory: " + path_);
        }
        for (fs::directory_iterator entry(path_, error), end; !error && entry != end;
             entry.increment(error)) {
            if (!is_dump_iteration_name(entry->path())) continue;
            fs::remove_all(entry->path(), error);
        }
        if (error) {
            throw std::runtime_error("unable to replace existing artifact dump in " + path_ +
                                     ": " + error.message());
        }
        return;
    }
    if (!fs::create_directories(path_, error) || error) {
        throw std::runtime_error("unable to create artifact dump directory: " + path_ +
                                 (error ? ": " + error.message() : ""));
    }
}

bool ArtifactDumper::includes(std::int64_t iteration) const {
    return iteration > 0 && iteration <= max_iteration_;
}

std::string ArtifactDumper::iteration_path(std::int64_t iteration) const {
    std::ostringstream name;
    name << "iteration-" << std::setfill('0') << std::setw(6) << iteration;
    return (fs::path(path_) / name.str()).string();
}

bool ArtifactDumper::ensure_iteration_directory(std::int64_t iteration) {
    std::error_code error;
    const std::string directory = iteration_path(iteration);
    fs::create_directories(directory, error);
    if (!error) return true;
    return fail("unable to create artifact directory " + directory +
                (error ? ": " + error.message() : ""));
}

bool ArtifactDumper::write_file(const std::string & path, const std::uint8_t * data,
                                std::size_t size) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) return fail("unable to create artifact: " + path);
    if (size != 0) output.write(reinterpret_cast<const char *>(data), size);
    output.close();
    if (!output) return fail("failed while writing artifact: " + path);
    return true;
}

bool ArtifactDumper::fail(std::string error) {
    if (error_.empty()) error_ = std::move(error);
    return false;
}

bool ArtifactDumper::record_observation(std::int64_t iteration,
                                        const std::vector<std::uint8_t> & image,
                                        const std::vector<std::uint8_t> & audio) {
    if (!includes(iteration)) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_.empty() || !ensure_iteration_directory(iteration)) return false;
    const fs::path directory(iteration_path(iteration));
    return write_file((directory / "observation.jpg").string(), image.data(), image.size()) &&
           write_file((directory / "observation.wav").string(), audio.data(), audio.size());
}

bool ArtifactDumper::record_llm(std::int64_t iteration, const std::string & text) {
    if (!includes(iteration)) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_.empty() || !ensure_iteration_directory(iteration)) return false;
    const auto * data = reinterpret_cast<const std::uint8_t *>(text.data());
    return write_file((fs::path(iteration_path(iteration)) / "llm.txt").string(),
                      data, text.size());
}

bool ArtifactDumper::append_tts(std::int64_t iteration, const float * samples,
                                std::size_t count, int sample_rate) {
    if (!includes(iteration) || samples == nullptr || count == 0) return true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!error_.empty() || !ensure_iteration_directory(iteration)) return false;
    if (sample_rate <= 0) return fail("invalid TTS sample rate for iteration " +
                                      std::to_string(iteration));

    TtsFile & state = tts_files_[iteration];
    if (state.sample_rate == 0) state.sample_rate = sample_rate;
    std::vector<float> converted;
    if (sample_rate != state.sample_rate) {
        converted = resample_linear(samples, count, sample_rate, state.sample_rate);
        samples = converted.data();
        count = converted.size();
    }

    constexpr std::uint64_t kMaxPcmSamples =
        (std::numeric_limits<std::uint32_t>::max() - 36ULL) / sizeof(std::int16_t);
    if (count > kMaxPcmSamples - state.samples) {
        return fail("TTS artifact exceeds the WAV size limit for iteration " +
                    std::to_string(iteration));
    }

    const std::vector<float> chunk(samples, samples + count);
    const std::vector<std::uint8_t> wav = make_wav_pcm16(chunk, state.sample_rate);
    const std::string path = (fs::path(iteration_path(iteration)) / "tts.wav").string();
    if (state.samples == 0) {
        if (!write_file(path, wav.data(), wav.size())) return false;
    } else {
        std::fstream output(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!output) return fail("unable to append TTS artifact: " + path);
        output.seekp(0, std::ios::end);
        output.write(reinterpret_cast<const char *>(wav.data() + 44), wav.size() - 44);

        const std::uint64_t total_samples = state.samples + count;
        const std::uint32_t data_size = static_cast<std::uint32_t>(
            total_samples * sizeof(std::int16_t));
        std::uint8_t encoded[4];
        output.seekp(4);
        output.write(reinterpret_cast<const char *>(store_u32(encoded, 36 + data_size)), 4);
        output.seekp(40);
        output.write(reinterpret_cast<const char *>(store_u32(encoded, data_size)), 4);
        output.close();
        if (!output) return fail("failed while appending TTS artifact: " + path);
    }
    state.samples += count;
    return true;
}

bool ArtifactDumper::take_error(std::string & error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_.empty() || error_reported_) return false;
    error = error_;
    error_reported_ = true;
    return true;
}

JpegFramer::JpegFramer(std::size_t max_frame_bytes) : max_frame_bytes_(max_frame_bytes) {}

std::vector<std::vector<std::uint8_t>> JpegFramer::feed(const std::uint8_t * data, std::size_t size) {
    std::vector<std::vector<std::uint8_t>> frames;
    if (data != nullptr && size != 0) {
        buffer_.insert(buffer_.end(), data, data + size);
    }

    for (;;) {
        const std::uint8_t soi_bytes[] = {0xff, 0xd8};
        auto soi = std::search(buffer_.begin(), buffer_.end(),
                               std::begin(soi_bytes), std::end(soi_bytes));
        if (soi == buffer_.end()) {
            const bool keep_ff = !buffer_.empty() && buffer_.back() == 0xff;
            buffer_.clear();
            if (keep_ff) buffer_.push_back(0xff);
            break;
        }
        if (soi != buffer_.begin()) buffer_.erase(buffer_.begin(), soi);

        const std::uint8_t eoi_bytes[] = {0xff, 0xd9};
        auto eoi = std::search(buffer_.begin() + std::min<std::size_t>(2, buffer_.size()),
                               buffer_.end(), std::begin(eoi_bytes), std::end(eoi_bytes));
        if (eoi == buffer_.end()) {
            if (buffer_.size() > max_frame_bytes_) {
                buffer_.clear();
                ++discarded_frames_;
            }
            break;
        }
        eoi += 2;
        frames.emplace_back(buffer_.begin(), eoi);
        buffer_.erase(buffer_.begin(), eoi);
    }
    return frames;
}

void LatestFrame::set(std::vector<std::uint8_t> frame) {
    std::lock_guard<std::mutex> lock(mutex_);
    frame_ = std::move(frame);
    ++generation_;
}

bool LatestFrame::copy(std::vector<std::uint8_t> & frame, std::uint64_t * generation) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (frame_.empty()) return false;
    frame = frame_;
    if (generation != nullptr) *generation = generation_;
    return true;
}

std::uint64_t LatestFrame::generation() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return generation_;
}

CapturePcmBuffer::CapturePcmBuffer(std::size_t capacity_samples)
    : capacity_samples_(std::max<std::size_t>(1, capacity_samples)) {}

void CapturePcmBuffer::push(const float * samples, std::size_t count) {
    if (samples == nullptr || count == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);
    for (std::size_t i = 0; i < count; ++i) samples_.push_back(samples[i]);
    while (samples_.size() > capacity_samples_) samples_.pop_front();
}

std::vector<float> CapturePcmBuffer::take_latest(std::size_t count) {
    std::vector<float> result(count, 0.0f);
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t available = std::min(count, samples_.size());
    auto begin = samples_.end() - static_cast<std::ptrdiff_t>(available);
    std::copy(begin, samples_.end(), result.end() - static_cast<std::ptrdiff_t>(available));
    samples_.clear();
    return result;
}

std::size_t CapturePcmBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_.size();
}

BoundedPcmQueue::BoundedPcmQueue(std::size_t capacity_samples)
    : capacity_samples_(std::max<std::size_t>(1, capacity_samples)) {}

void BoundedPcmQueue::drop_front(std::size_t count) {
    count = std::min(count, size_samples_);
    size_samples_ -= count;
    while (count != 0 && !segments_.empty()) {
        Segment & segment = segments_.front();
        const std::size_t available = segment.samples.size() - segment.offset;
        const std::size_t removed = std::min(count, available);
        segment.offset += removed;
        count -= removed;
        if (segment.offset == segment.samples.size()) segments_.pop_front();
    }
}

std::size_t BoundedPcmQueue::enqueue(const float * samples, std::size_t count,
                                     std::int64_t iteration) {
    if (samples == nullptr || count == 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return 0;
    std::size_t dropped = 0;
    if (count >= capacity_samples_) {
        dropped = size_samples_ + count - capacity_samples_;
        segments_.clear();
        size_samples_ = 0;
        samples += count - capacity_samples_;
        count = capacity_samples_;
    } else {
        const std::size_t overflow = size_samples_ + count > capacity_samples_
            ? size_samples_ + count - capacity_samples_ : 0;
        dropped = overflow;
        drop_front(overflow);
    }
    Segment segment;
    segment.samples.assign(samples, samples + count);
    segment.iteration = iteration;
    segments_.push_back(std::move(segment));
    size_samples_ += count;
    return dropped;
}

std::size_t BoundedPcmQueue::pull_segment(float * output, std::size_t count,
                                          std::int64_t * iteration) {
    if (output == nullptr || count == 0) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    if (segments_.empty()) return 0;
    Segment & segment = segments_.front();
    const std::size_t copied = std::min(count, segment.samples.size() - segment.offset);
    std::copy_n(segment.samples.data() + segment.offset, copied, output);
    if (iteration) *iteration = segment.iteration;
    segment.offset += copied;
    size_samples_ -= copied;
    if (segment.offset == segment.samples.size()) segments_.pop_front();
    return copied;
}

std::size_t BoundedPcmQueue::pull(float * output, std::size_t count) {
    if (output == nullptr || count == 0) return 0;
    std::size_t copied = 0;
    while (copied < count) {
        const std::size_t next = pull_segment(output + copied, count - copied);
        if (next == 0) break;
        copied += next;
    }
    std::fill(output + copied, output + count, 0.0f);
    return copied;
}

void BoundedPcmQueue::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
}

bool BoundedPcmQueue::closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
}

bool BoundedPcmQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_samples_ == 0;
}

std::size_t BoundedPcmQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return size_samples_;
}

std::vector<float> resample_linear(const float * samples, std::size_t count,
                                   int source_rate, int destination_rate) {
    if (samples == nullptr || count == 0 || source_rate <= 0 || destination_rate <= 0) return {};
    if (source_rate == destination_rate) return {samples, samples + count};
    const std::size_t output_count = static_cast<std::size_t>(
        std::llround(static_cast<double>(count) * destination_rate / source_rate));
    std::vector<float> output(output_count);
    const double step = static_cast<double>(source_rate) / destination_rate;
    for (std::size_t i = 0; i < output_count; ++i) {
        const double position = i * step;
        const std::size_t left = std::min(static_cast<std::size_t>(position), count - 1);
        const std::size_t right = std::min(left + 1, count - 1);
        const float fraction = static_cast<float>(position - left);
        output[i] = samples[left] + (samples[right] - samples[left]) * fraction;
    }
    return output;
}

std::vector<std::uint8_t> make_wav_pcm16(const std::vector<float> & samples, int sample_rate) {
    const std::uint32_t data_size = static_cast<std::uint32_t>(samples.size() * sizeof(std::int16_t));
    std::vector<std::uint8_t> wav;
    wav.reserve(44 + data_size);
    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    append_u32(wav, 36 + data_size);
    wav.insert(wav.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '});
    append_u32(wav, 16);
    append_u16(wav, 1);
    append_u16(wav, 1);
    append_u32(wav, static_cast<std::uint32_t>(sample_rate));
    append_u32(wav, static_cast<std::uint32_t>(sample_rate * sizeof(std::int16_t)));
    append_u16(wav, sizeof(std::int16_t));
    append_u16(wav, 16);
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    append_u32(wav, data_size);
    for (float sample : samples) {
        if (!std::isfinite(sample)) sample = 0.0f;
        sample = std::max(-1.0f, std::min(1.0f, sample));
        const auto pcm = static_cast<std::int16_t>(std::lrint(sample * 32767.0f));
        append_u16(wav, static_cast<std::uint16_t>(pcm));
    }
    return wav;
}

const std::string & monitor_prompt() {
    static const std::string prompt =
        "Streaming Duplex Conversation! You are a visual phone-use monitor. "
        "For every one-second observation, make exactly one binary decision using the current image. "
        "If the person is visibly holding a mobile phone in either hand, your entire response must be "
        "exactly \"Please get off your phone\" without quotation marks. End immediately after the word "
        "phone; do not add punctuation, a label, or any other text. Otherwise remain completely silent and "
        "produce no text or audio. Treat a partially visible phone as held "
        "when it is visibly in a hand. Do not require scrolling, tapping, or looking at the screen. "
        "Repeat the exact reminder on every qualifying observation, even if it was just said. "
        "Ignore all speech and audio content when making this decision.";
    return prompt;
}

std::pair<std::string, std::string> wrap_duplex_system_prompt(const std::string & prompt) {
    return {"<|im_start|>system\n" + prompt + "\n<|audio_start|>",
            "<|audio_end|><|im_end|>\n"};
}

} // namespace vlm
