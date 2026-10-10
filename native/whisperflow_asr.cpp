// Thin C API in front of whisper.cpp.
//
// Python must not depend on whisper.cpp struct layouts. Those structs change
// between releases. This file is the only place that touches them.
//
// Data flow for one call:
//   float32 PCM (16 kHz, mono, -1..1)
//     -> whisper_full()          model stays in memory from wf_load()
//     -> segment text
//     -> NUL-terminated buffer the caller owns
//
// The shared library is compiled with the same SYCL toolchain as whisper.cpp.
// wf_load(use_gpu=1) asks whisper to run on the Intel backend loaded from
// backend_dir (the directory that contains libggml-sycl.so).

#include "whisper.h"
#include "ggml-backend.h"

#include <cstring>
#include <string>

static whisper_context * g_ctx = nullptr;
static std::string g_text;

// whisper.cpp writes progress to stderr by default. That would interleave
// with the partial lines the CLI prints, so keep warnings and errors only.
static void wf_log(enum ggml_log_level level, const char * text, void * /*user*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fprintf(stderr, "%s", text);
    }
}

static void copy_out(const std::string & text, char * out, int out_cap) {
    if (out == nullptr || out_cap <= 0) {
        return;
    }
    const int n = static_cast<int>(text.size());
    const int keep = n < out_cap - 1 ? n : out_cap - 1;
    if (keep > 0) {
        memcpy(out, text.data(), static_cast<size_t>(keep));
    }
    out[keep] = '\0';
}

static std::string trim_copy(const std::string & in) {
    size_t begin = 0;
    while (begin < in.size() && (in[begin] == ' ' || in[begin] == '\n' || in[begin] == '\t')) {
        ++begin;
    }
    size_t end = in.size();
    while (end > begin && (in[end - 1] == ' ' || in[end - 1] == '\n' || in[end - 1] == '\t')) {
        --end;
    }
    std::string out;
    out.reserve(end - begin);
    bool space = false;
    for (size_t i = begin; i < end; ++i) {
        const char c = in[i];
        if (c == ' ' || c == '\n' || c == '\t') {
            space = !out.empty();
            continue;
        }
        if (space) {
            out.push_back(' ');
            space = false;
        }
        out.push_back(c);
    }
    return out;
}

extern "C" {

// backend_dir may be null. When set, ggml loads libggml-sycl.so from that
// folder instead of the process working directory.
int wf_load(const char * model_path, int use_gpu, int gpu_device, const char * backend_dir) {
    whisper_log_set(wf_log, nullptr);
    if (backend_dir != nullptr && backend_dir[0] != '\0') {
        ggml_backend_load_all_from_path(backend_dir);
    } else {
        ggml_backend_load_all();
    }

    if (g_ctx != nullptr) {
        whisper_free(g_ctx);
        g_ctx = nullptr;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = use_gpu != 0;
    cparams.gpu_device = gpu_device;
    // Flash attention on SYCL is still experimental in whisper.cpp. Leave it
    // off so the first GPU path matches the regular encoder.
    cparams.flash_attn = false;

    g_ctx = whisper_init_from_file_with_params(model_path, cparams);
    return g_ctx == nullptr ? 1 : 0;
}

const char * wf_system_info(void) {
    return whisper_print_system_info();
}

// single_segment=1 is for short streaming windows (one rolling hypothesis).
// The final pass over a finished phrase uses single_segment=0 so whisper can
// split sentences the way it does for a normal file.
int wf_transcribe(
    const float * samples,
    int n_samples,
    int n_threads,
    int single_segment,
    const char * language,
    const char * initial_prompt,
    int beam_size,
    float no_speech_thold,
    char * out,
    int out_cap) {
    if (g_ctx == nullptr || samples == nullptr || n_samples <= 0) {
        copy_out("", out, out_cap);
        return 1;
    }

    enum whisper_sampling_strategy strategy = (beam_size > 1 && single_segment == 0)
        ? WHISPER_SAMPLING_BEAM_SEARCH
        : WHISPER_SAMPLING_GREEDY;

    whisper_full_params params = whisper_full_default_params(strategy);
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = false;
    params.print_special = false;
    params.no_timestamps = true;
    params.single_segment = single_segment != 0;
    // Each window is transcribed on its own. The Python accumulator is what
    // stitches windows together, so past decoder state would double-count.
    params.no_context = true;

    if (language != nullptr && language[0] != '\0') {
        params.language = language;
    } else {
        params.language = "en";
    }

    if (initial_prompt != nullptr && initial_prompt[0] != '\0') {
        params.initial_prompt = initial_prompt;
    }

    params.n_threads = n_threads > 0 ? n_threads : 4;
    params.suppress_blank = true;
    params.suppress_nst = true;
    params.no_speech_thold = no_speech_thold > 0.0f ? no_speech_thold : 0.6f;
    params.temperature_inc = 0.0f;

    if (strategy == WHISPER_SAMPLING_BEAM_SEARCH) {
        params.beam_search.beam_size = beam_size;
    }

    if (whisper_full(g_ctx, params, samples, n_samples) != 0) {
        copy_out("", out, out_cap);
        return 2;
    }

    g_text.clear();
    const int n_segments = whisper_full_n_segments(g_ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char * text = whisper_full_get_segment_text(g_ctx, i);
        if (text != nullptr) {
            if (!g_text.empty()) {
                g_text.push_back(' ');
            }
            g_text += text;
        }
    }
    g_text = trim_copy(g_text);
    copy_out(g_text, out, out_cap);
    return 0;
}

// Releases the model. The CLI does not call this between utterances.
void wf_free(void) {
    if (g_ctx != nullptr) {
        whisper_free(g_ctx);
        g_ctx = nullptr;
    }
    g_text.clear();
}

}  // extern "C"
