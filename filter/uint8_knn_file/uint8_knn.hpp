#pragma once

#include <cstdint>
#include <cstdio>
#include <immintrin.h>
#include <queue>
#include <vector>

inline int32_t dist_u7_l2(const uint8_t *x, const uint8_t *y, int32_t dim) {
  auto sum = _mm512_setzero_si512();
  for (int32_t i = 0; i < dim; i += 64) {
    auto xx = _mm512_loadu_epi8(x + i);
    auto yy = _mm512_loadu_epi8(y + i);
    auto t = _mm512_sub_epi8(xx, yy);
    t = _mm512_abs_epi8(t);
    sum = _mm512_dpbusd_epi32(sum, t, t);
  }
  return _mm512_reduce_add_epi32(sum);
}

inline int32_t dist_u8_s8_dot(const uint8_t *x, const int8_t *y, int32_t dim) {
  auto sum = _mm512_setzero_si512();
  for (int32_t i = 0; i < dim; i += 64) {
    auto xx = _mm512_loadu_epi8(x + i);
    auto yy = _mm512_loadu_epi8(y + i);
    sum = _mm512_dpbusd_epi32(sum, xx, yy);
  }
  return _mm512_reduce_add_epi32(sum);
}

inline int32_t get_norm(const uint8_t *x, int32_t dim) {
  int32_t norm = 0;
  for (int i = 0; i < dim; ++i) {
    norm += ((int32_t)x[i] - 256) * x[i];
  }
  return norm;
}

// dim must be aligned to 64
inline void uint8_knn(int32_t dim, const uint8_t *Q, int32_t nq,
                      const uint8_t *X, int32_t n, int32_t topk, int32_t *ids) {
  std::vector<std::priority_queue<std::pair<int32_t, int32_t>>> queues(nq);
  std::vector<int32_t> xnorms(n);

  // #pragma omp parallel for schedule(static)
  for (int i = 0; i < n; ++i) {
    xnorms[i] = get_norm(X + i * dim, dim);
  }

  std::vector<int8_t> y_tmp(dim * nq);
  for (int i = 0; i < dim * nq; ++i) {
    y_tmp[i] = (int)Q[i] - 128;
  }

  auto push_to_queue = [&](auto &queue, int32_t i, int32_t d) {
    if (queue.size() < topk) {
      queue.emplace(d, i);
    } else if (queue.top().first > d) {
      queue.pop();
      queue.emplace(d, i);
    }
  };

  // #pragma omp parallel for schedule(dynamic)
  for (int32_t j = 0; j < nq; ++j) {
    auto cur_q = y_tmp.data() + j * dim;
    for (int32_t i = 0; i < n; ++i) {
      auto cur_x = X + i * dim;
      auto dist = xnorms[i] - 2 * dist_u8_s8_dot(cur_x, cur_q, dim);
      push_to_queue(queues[j], i, dist);
    }
  }

  // #pragma omp parallel for schedule(dynamic)
  for (int32_t i = 0; i < nq; ++i) {
    auto &queue = queues[i];
    int sz = queue.size();
    for (int j = 0; j < std::min(topk, sz); ++j) {
      ids[(i + 1) * topk - j - 1] = queue.top().second;
      queue.pop();
    }
  }
}