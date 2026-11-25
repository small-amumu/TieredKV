// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Zijie Hu
// T3 near-data CPU correction kernels (rp_cpu.py): simple C++ / OpenMP, compiler-vectorised, not hand-tuned SIMD.
// Records: words = little-endian bit stream of 128 codes of `bits` bits (16 * bits bytes per row);
// meta = (min, step) as fp16 bits; value = code * step + min.
// round1: logits[h, g, m] = scale * sum_c q[h, g, c] * value(row ids[h, m], c)            (key candidates)
// round2: out[h, g, c]    = sum_j p[h, g, j] * value(row ids[h, j], c)                    (value rows)
#include <torch/extension.h>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstring>

static inline float h2f(uint16_t h) { return _cvtsh_ss(h); }

// 8 consecutive codes occupy exactly `bits` bytes, so each group of 8 is one unaligned load of `bits` bytes.
static inline void decode_row(const uint8_t* row, int bits, float mn, float st, float* out) {
    const uint64_t mask = (1ull << bits) - 1ull;
    for (int grp = 0; grp < 16; ++grp) {
        uint64_t x = 0;
        std::memcpy(&x, row + grp * bits, bits);
        for (int k = 0; k < 8; ++k) out[grp * 8 + k] = static_cast<float>((x >> (k * bits)) & mask) * st + mn;
    }
}

void set_threads(int64_t n) { omp_set_num_threads(static_cast<int>(n)); }

int64_t max_threads() { return omp_get_max_threads(); }

static void check_store(const torch::Tensor& words, const torch::Tensor& meta, int64_t bits) {
    TORCH_CHECK(words.is_cpu() && meta.is_cpu() && words.is_contiguous() && meta.is_contiguous());
    TORCH_CHECK(words.scalar_type() == torch::kInt32 && meta.scalar_type() == torch::kInt16);
    TORCH_CHECK(bits >= 1 && bits <= 8 && words.numel() * 4 >= 16 * bits);
    TORCH_CHECK(meta.numel() >= 2 * (words.numel() * 4 / (16 * bits)), "meta smaller than 2 values per row");
}

torch::Tensor round1(torch::Tensor words, torch::Tensor meta, torch::Tensor q, torch::Tensor ids, int64_t bits,
                     double scale) {
    check_store(words, meta, bits);
    TORCH_CHECK(q.is_cpu() && ids.is_cpu() && q.is_contiguous() && ids.is_contiguous());
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 && ids.scalar_type() == torch::kInt64);
    TORCH_CHECK(q.dim() == 3 && q.size(2) == 128 && ids.dim() == 2 && ids.size(0) == q.size(0));
    const int64_t BH = q.size(0), Gq = q.size(1), M = ids.size(1);
    const int64_t rb = 16 * bits;
    const int64_t nrows = words.numel() * 4 / rb;
    auto out = torch::empty({BH, Gq, M}, torch::kFloat32);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(words.data_ptr());
    const uint16_t* MT = reinterpret_cast<const uint16_t*>(meta.data_ptr());
    const float* Q = q.data_ptr<float>();
    const int64_t* I = ids.data_ptr<int64_t>();
    float* O = out.data_ptr<float>();
    const float sc = static_cast<float>(scale);
    bool bad = false;
#pragma omp parallel for collapse(2) schedule(static) reduction(|| : bad)
    for (int64_t h = 0; h < BH; ++h) {
        for (int64_t m = 0; m < M; ++m) {
            const int64_t r = I[h * M + m];
            if (r < 0 || r >= nrows) { bad = true; continue; }
            float v[128];
            decode_row(W + r * rb, static_cast<int>(bits), h2f(MT[2 * r]), h2f(MT[2 * r + 1]), v);
            for (int64_t g = 0; g < Gq; ++g) {
                const float* qq = Q + (h * Gq + g) * 128;
                float a = 0.f;
#pragma omp simd reduction(+ : a)
                for (int c = 0; c < 128; ++c) a += qq[c] * v[c];
                O[(h * Gq + g) * M + m] = a * sc;
            }
        }
    }
    TORCH_CHECK(!bad, "round1: row id out of range");
    return out;
}

// round2 runs over (head, chunk of CH value rows) tasks so that small batches still spread over many threads:
// each task decodes its rows once and writes a partial sum; a second pass adds the chunks per head.
torch::Tensor round2(torch::Tensor words, torch::Tensor meta, torch::Tensor p, torch::Tensor ids, int64_t bits) {
    check_store(words, meta, bits);
    TORCH_CHECK(p.is_cpu() && ids.is_cpu() && p.is_contiguous() && ids.is_contiguous());
    TORCH_CHECK(p.scalar_type() == torch::kFloat32 && ids.scalar_type() == torch::kInt64);
    TORCH_CHECK(p.dim() == 3 && p.size(1) <= 8 && ids.dim() == 2 && ids.size(0) == p.size(0)
                && ids.size(1) == p.size(2));
    const int64_t BH = p.size(0), Gq = p.size(1), K = ids.size(1);
    const int64_t CH = 8;
    const int64_t NCH = (K + CH - 1) / CH;
    const int64_t rb = 16 * bits;
    const int64_t nrows = words.numel() * 4 / rb;
    auto part = torch::zeros({BH, NCH, Gq, 128}, torch::kFloat32);
    auto out = torch::empty({BH, Gq, 128}, torch::kFloat32);
    const uint8_t* W = reinterpret_cast<const uint8_t*>(words.data_ptr());
    const uint16_t* MT = reinterpret_cast<const uint16_t*>(meta.data_ptr());
    const float* P = p.data_ptr<float>();
    const int64_t* I = ids.data_ptr<int64_t>();
    float* PT = part.data_ptr<float>();
    float* O = out.data_ptr<float>();
    bool bad = false;
#pragma omp parallel for collapse(2) schedule(static) reduction(|| : bad)
    for (int64_t h = 0; h < BH; ++h) {
        for (int64_t ch = 0; ch < NCH; ++ch) {
            float v[128];
            float* acc = PT + (h * NCH + ch) * Gq * 128;
            const int64_t j1 = std::min(K, (ch + 1) * CH);
            for (int64_t j = ch * CH; j < j1; ++j) {
                const int64_t r = I[h * K + j];
                if (r < 0 || r >= nrows) { bad = true; continue; }
                decode_row(W + r * rb, static_cast<int>(bits), h2f(MT[2 * r]), h2f(MT[2 * r + 1]), v);
                for (int64_t g = 0; g < Gq; ++g) {
                    const float pj = P[(h * Gq + g) * K + j];
                    float* a = acc + g * 128;
#pragma omp simd
                    for (int c = 0; c < 128; ++c) a[c] += pj * v[c];
                }
            }
        }
    }
    TORCH_CHECK(!bad, "round2: row id out of range");
#pragma omp parallel for collapse(2) schedule(static)
    for (int64_t h = 0; h < BH; ++h) {
        for (int64_t g = 0; g < Gq; ++g) {
            float* o = O + (h * Gq + g) * 128;
            for (int c = 0; c < 128; ++c) o[c] = 0.f;
            for (int64_t ch = 0; ch < NCH; ++ch) {
                const float* a = PT + ((h * NCH + ch) * Gq + g) * 128;
#pragma omp simd
                for (int c = 0; c < 128; ++c) o[c] += a[c];
            }
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("round1", &round1);
    m.def("round2", &round2);
    m.def("set_threads", &set_threads);
    m.def("max_threads", &max_threads);
}
