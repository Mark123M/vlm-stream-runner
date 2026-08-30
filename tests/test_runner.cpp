#include "runner_support.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(expression) do { \
    if (!(expression)) { \
        std::cerr << __FILE__ << ':' << __LINE__ << ": check failed: " #expression "\n"; \
        ++failures; \
    } \
} while (false)

std::uint32_t le32(const std::vector<std::uint8_t> & data, std::size_t offset) {
    return static_cast<std::uint32_t>(data[offset]) |
           (static_cast<std::uint32_t>(data[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(data[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(data[offset + 3]) << 24);
}

void test_jpeg_framing() {
    vlm::JpegFramer framer(32);
    const std::uint8_t first[] = {0x00, 0xff, 0xd8, 0x01};
    const std::uint8_t second[] = {0x02, 0xff, 0xd9, 0xff, 0xd8, 0x03, 0xff, 0xd9};
    CHECK(framer.feed(first, sizeof(first)).empty());
    const auto frames = framer.feed(second, sizeof(second));
    CHECK(frames.size() == 2);
    CHECK(frames[0] == std::vector<std::uint8_t>({0xff, 0xd8, 0x01, 0x02, 0xff, 0xd9}));
    CHECK(frames[1] == std::vector<std::uint8_t>({0xff, 0xd8, 0x03, 0xff, 0xd9}));
}

void test_latest_frame_dropping() {
    vlm::LatestFrame latest;
    latest.set({1, 2, 3});
    latest.set({4, 5});
    std::vector<std::uint8_t> frame;
    std::uint64_t generation = 0;
    CHECK(latest.copy(frame, &generation));
    CHECK(frame == std::vector<std::uint8_t>({4, 5}));
    CHECK(generation == 2);
}

void test_microphone_chunking() {
    vlm::CapturePcmBuffer buffer(6);
    const float old_samples[] = {1, 2, 3, 4};
    const float new_samples[] = {5, 6, 7, 8};
    buffer.push(old_samples, 4);
    buffer.push(new_samples, 4);
    const auto chunk = buffer.take_latest(4);
    CHECK(chunk == std::vector<float>({5, 6, 7, 8}));
    CHECK(buffer.size() == 0);

    buffer.push(new_samples, 2);
    const auto padded = buffer.take_latest(4);
    CHECK(padded == std::vector<float>({0, 0, 5, 6}));
}

void test_wav_construction() {
    const auto wav = vlm::make_wav_pcm16({-1.0f, 0.0f, 1.0f});
    CHECK(wav.size() == 50);
    CHECK(std::string(wav.begin(), wav.begin() + 4) == "RIFF");
    CHECK(std::string(wav.begin() + 8, wav.begin() + 12) == "WAVE");
    CHECK(le32(wav, 24) == 16000);
    CHECK(le32(wav, 40) == 6);
}

void test_prompt_wrapping() {
    const std::string & prompt = vlm::monitor_prompt();
    CHECK(prompt.find("visibly holding a mobile phone") != std::string::npos);
    CHECK(prompt.find("Do not require scrolling, tapping, or looking at the screen") != std::string::npos);
    CHECK(prompt.find("exactly \"Please get off your phone\" without quotation marks") != std::string::npos);
    CHECK(prompt.find("do not add punctuation, a label, or any other text") != std::string::npos);
    CHECK(prompt.find("SPEAK") == std::string::npos);
    CHECK(prompt.find("LISTEN") == std::string::npos);
    CHECK(prompt.find("actively scrolling through a phone") == std::string::npos);
    const auto wrapped = vlm::wrap_duplex_system_prompt(prompt);
    CHECK(wrapped.first.rfind("<|im_start|>system\n", 0) == 0);
    CHECK(wrapped.first.find(prompt) != std::string::npos);
    const std::string audio_start = "<|audio_start|>";
    CHECK(wrapped.first.size() >= audio_start.size() &&
          wrapped.first.substr(wrapped.first.size() - audio_start.size()) == audio_start);
    CHECK(wrapped.second == "<|audio_end|><|im_end|>\n");
}

void test_bounded_playback_queue() {
    vlm::BoundedPcmQueue queue(5);
    const float first[] = {1, 2, 3, 4};
    const float second[] = {5, 6, 7};
    CHECK(queue.enqueue(first, 4, 11) == 0);
    CHECK(queue.enqueue(second, 3, 12) == 2);
    float output[5] = {};
    std::int64_t iteration = -1;
    CHECK(queue.pull_segment(output, 5, &iteration) == 2);
    CHECK(iteration == 11);
    CHECK(std::vector<float>(output, output + 2) == std::vector<float>({3, 4}));
    CHECK(queue.pull_segment(output + 2, 3, &iteration) == 3);
    CHECK(iteration == 12);
    CHECK(std::vector<float>(output, output + 5) == std::vector<float>({3, 4, 5, 6, 7}));
    queue.close();
    CHECK(queue.closed());
    CHECK(queue.empty());
    CHECK(queue.enqueue(first, 4) == 0);
    CHECK(queue.empty());
}

std::size_t occurrences(const std::string & text, const std::string & needle) {
    std::size_t count = 0;
    for (std::size_t position = 0; (position = text.find(needle, position)) != std::string::npos;
         position += needle.size()) {
        ++count;
    }
    return count;
}

void test_perfetto_trace() {
    const std::string path = "/private/tmp/vlm-stream-runner-unit-trace.json";
    const auto begin = vlm::PerfettoTrace::Clock::now();
    {
        vlm::PerfettoTrace trace(path, 4096);
        trace.record(vlm::PipelineStage::Observation, 1, begin, begin + std::chrono::seconds(1));
        trace.record(vlm::PipelineStage::Encoder, 2, begin + std::chrono::seconds(1),
                     begin + std::chrono::milliseconds(1250));
        trace.record_dropped(vlm::PipelineStage::Observation, 3,
                             begin + std::chrono::seconds(2), begin + std::chrono::seconds(3),
                             "inference_backpressure");
        for (int i = 0; i < 100; ++i) {
            trace.record(vlm::PipelineStage::Llm, 4 + i, begin, begin + std::chrono::milliseconds(1));
        }
        std::string error;
        CHECK(trace.finalize(&error));
        CHECK(error.empty());
        CHECK(trace.finalize(&error));
    }

    std::ifstream input(path, std::ios::binary);
    const std::string json((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    CHECK(!json.empty());
    CHECK(json.front() == '{');
    CHECK(json.find("\"traceEvents\"") != std::string::npos);
    CHECK(occurrences(json, "\"thread_name\"") == 6);
    CHECK(json.find("1 Observation capture") != std::string::npos);
    CHECK(json.find("2 Multimodal encoder") != std::string::npos);
    CHECK(json.find("6 Audio playback") != std::string::npos);
    CHECK(json.find("iteration 1") != std::string::npos);
    CHECK(json.find("\"name\":\"dropped\"") != std::string::npos);
    CHECK(json.find("thread_state_unknown") != std::string::npos);
    CHECK(json.find("inference_backpressure") != std::string::npos);
    CHECK(json.find("omitted_events") != std::string::npos);
    CHECK(json.size() <= 4096);
    CHECK(json.find("Vision encode") == std::string::npos);
    CHECK(json.find("Audio encode") == std::string::npos);
    std::remove(path.c_str());
}

void test_resampling() {
    const float input[] = {0, 1, 0, -1};
    const auto doubled = vlm::resample_linear(input, 4, 2, 4);
    CHECK(doubled.size() == 8);
    CHECK(std::fabs(doubled[0]) < 1e-6f);
    CHECK(std::fabs(doubled[2] - 1.0f) < 1e-6f);
}

void test_artifact_dump() {
    namespace fs = std::filesystem;
    const fs::path path = fs::temp_directory_path() /
        ("vlm-stream-runner-unit-dump-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    const std::vector<std::uint8_t> image = {0xff, 0xd8, 0xff, 0xd9};
    const std::vector<std::uint8_t> audio = vlm::make_wav_pcm16({0.0f, 0.5f});
    {
        vlm::ArtifactDumper dump(path.string(), 2);
        CHECK(dump.record_observation(1, image, audio));
        CHECK(dump.record_llm(1, "Please get off your phone"));
        const float first[] = {-1.0f, 0.0f};
        const float second[] = {1.0f};
        CHECK(dump.append_tts(1, first, 2, 24000));
        CHECK(dump.append_tts(1, second, 1, 24000));

        CHECK(dump.record_observation(2, image, audio));
        CHECK(dump.record_llm(2, ""));

        CHECK(dump.record_observation(3, image, audio));
        CHECK(!fs::exists(path / "iteration-000003"));
    }

    const fs::path iteration = path / "iteration-000001";
    std::ifstream image_input(iteration / "observation.jpg", std::ios::binary);
    const std::vector<std::uint8_t> image_bytes(
        (std::istreambuf_iterator<char>(image_input)), std::istreambuf_iterator<char>());
    CHECK(image_bytes == std::vector<std::uint8_t>({0xff, 0xd8, 0xff, 0xd9}));

    std::ifstream text_input(iteration / "llm.txt", std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(text_input)),
                           std::istreambuf_iterator<char>());
    CHECK(text == "Please get off your phone");
    CHECK(fs::is_regular_file(path / "iteration-000002" / "llm.txt"));
    CHECK(fs::file_size(path / "iteration-000002" / "llm.txt") == 0);
    CHECK(!fs::exists(path / "iteration-000002" / "tts.wav"));

    std::ifstream wav_input(iteration / "tts.wav", std::ios::binary);
    const std::vector<std::uint8_t> wav(
        (std::istreambuf_iterator<char>(wav_input)), std::istreambuf_iterator<char>());
    CHECK(wav.size() == 50);
    if (wav.size() >= 44) {
        CHECK(le32(wav, 24) == 24000);
        CHECK(le32(wav, 40) == 6);
    }

    {
        std::ofstream keep(path / "keep.txt");
        keep << "unrelated";
    }
    {
        vlm::ArtifactDumper replacement(path.string(), 1);
        CHECK(!fs::exists(path / "iteration-000001"));
        CHECK(!fs::exists(path / "iteration-000002"));
        CHECK(fs::is_regular_file(path / "keep.txt"));
        CHECK(replacement.record_observation(1, image, audio));
        CHECK(replacement.record_llm(1, ""));
    }
    CHECK(fs::is_regular_file(path / "iteration-000001" / "observation.jpg"));
    CHECK(fs::is_regular_file(path / "iteration-000001" / "llm.txt"));
    CHECK(!fs::exists(path / "iteration-000001" / "tts.wav"));
    CHECK(!fs::exists(path / "iteration-000002"));
    CHECK(fs::is_regular_file(path / "keep.txt"));
    fs::remove_all(path);
}

} // namespace

int main() {
    test_jpeg_framing();
    test_latest_frame_dropping();
    test_microphone_chunking();
    test_wav_construction();
    test_prompt_wrapping();
    test_bounded_playback_queue();
    test_perfetto_trace();
    test_resampling();
    test_artifact_dump();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "all runner unit tests passed\n";
    return EXIT_SUCCESS;
}
