#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float
L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

#if defined(USE_AVX512)

// Favor using AVX512 if available.
static float
L2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        // sum = _mm512_fmadd_ps(diff, diff, sum);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif

#if defined(USE_AVX)

// Favor using AVX if available.
static float
L2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
}

#endif

#if defined(USE_SSE)

static float
L2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float *pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}
#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC<float> L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;

static float
L2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}
#endif


#if defined(USE_SSE)
static float
L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);


    size_t qty4 = qty >> 2;

    const float *pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

static float
L2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}
#endif

class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    #if defined(USE_AVX512)
        if (AVX512Capable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        else if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #elif defined(USE_AVX)
        if (AVXCapable())
            L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
    #endif

        if (dim % 16 == 0)
            fstdistfunc_ = L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = L2SqrSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2Space() {}
};

/*
My error code, silly danny!
#if defined(USE_AVX512)
static int32_t
L2SqrISIMD16ExtAVX512(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    unsigned char *pVect1v = (unsigned char *) pVect1;
    unsigned char *pVect2v = (unsigned char *) pVect2;
    size_t qty = *((size_t *) qty_ptr);
    int32_t PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const unsigned char *pEnd1v = pVect1v + (qty16 << 4);

    __m512i diff, v1, v2;
    __m512i sum = _mm512_set1_epi32(0);

    while (pVect1v < pEnd1v) {
        v1 = _mm512_loadu_epi16(pVect1v);
        pVect1v += 4;
        v2 = _mm512_loadu_epi16(pVect2v);
        pVect2v += 4;
        diff = _mm512_sub_epi32(v1, v2);
        sum = _mm512_add_epi32(sum, _mm512_mul_epi32(diff, diff));

        v1 = _mm512_loadu_epi16(pVect1v);
        pVect1v += 4;
        v2 = _mm512_loadu_epi16(pVect2v);
        pVect2v += 4;
        diff = _mm512_sub_epi32(v1, v2);
        sum = _mm512_add_epi32(sum, _mm512_mul_epi32(diff, diff));

        v1 = _mm512_loadu_epi16(pVect1v);
        pVect1v += 4;
        v2 = _mm512_loadu_epi16(pVect2v);
        pVect2v += 4;
        diff = _mm512_sub_epi32(v1, v2);
        sum = _mm512_add_epi32(sum, _mm512_mul_epi32(diff, diff));

        v1 = _mm512_loadu_epi16(pVect1v);
        pVect1v += 4;
        v2 = _mm512_loadu_epi16(pVect2v);
        pVect2v += 4;
        diff = _mm512_sub_epi32(v1, v2);
        sum = _mm512_add_epi32(sum, _mm512_mul_epi32(diff, diff));
    }

    _mm512_store_epi32(TmpRes, sum);
    int32_t res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
            TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
            TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}
#endif
*/


//#if defined(USE_AVX512)
//
//static int32_t
//L2SqrISIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//    unsigned char *pVect1 = (unsigned char *) pVect1v;
//    unsigned char *pVect2 = (unsigned char *) pVect2v;
//    size_t qty = *((size_t *) qty_ptr);
//    size_t qty16 = qty >> 4;
//    const unsigned char *pEnd1 = pVect1 + (qty16 << 4);
//    __m256i v1, v2;
//    __m512i diff, v11, v22;
//    __m512i sum = _mm512_set1_epi32(0);
//    while (pVect1 < pEnd1) {
//        v1 = _mm256_loadu_si256((const __m256i*) pVect1);
//        pVect1 += 32;
//        v2 = _mm256_loadu_si256((const __m256i*) pVect2);
//        pVect2 += 32;
//        v11 = _mm512_cvtepu8_epi16(v1);
//        v22 = _mm512_cvtepu8_epi16(v2);
//        diff = _mm512_sub_epi16(v11, v22);
//        sum = _mm512_dpwssd_epi32(sum, diff, diff);
//    }
//    auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum, 0), _mm512_extracti32x8_epi32(sum, 1));
//    auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
//    auto tmp1 = _mm_hadd_epi32(sumhh, sumhh);
//    return _mm_extract_epi32(tmp1, 0) + _mm_extract_epi32(tmp1, 1);
//}
//
//
//#endif
//
//#if defined(USE_AVX512)
//
//static int32_t
//L2SqrISIMD16ExtAVX512V2(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
//    int8_t *pVect1 = (int8_t *) pVect1v;
//    int8_t *pVect2 = (int8_t *) pVect2v;
//    size_t qty = *((size_t *) qty_ptr);
//    size_t qty16 = qty >> 4;
//    const int8_t *pEnd1 = pVect1 + (qty16 << 4);
//    __m512i v1, v2;
//    __m512i diff, v11, v22;
//    __m512i sum = _mm512_set1_epi32(0);
//    while (pVect1 < pEnd1) {
//        v1 = _mm512_loadu_si512((const __m512i*) pVect1);
//        pVect1 += 64;
//        v2 = _mm512_loadu_si512((const __m512i*) pVect2);
//        pVect2 += 64;
//        diff = _mm512_sub_epi8(v11, v22);
//        diff = _mm512_abs_epi8(diff);
//        sum = _mm512_dpbusd_epi32(sum, diff, diff);
//    }
//    auto sumh = _mm256_add_epi32(_mm512_extracti32x8_epi32(sum, 0), _mm512_extracti32x8_epi32(sum, 1));
//    auto sumhh = _mm_add_epi32(_mm256_castsi256_si128(sumh), _mm256_extracti128_si256(sumh, 1));
//    auto tmp1 = _mm_hadd_epi32(sumhh, sumhh);
//    return _mm_extract_epi32(tmp1, 0) + _mm_extract_epi32(tmp1, 1);
//}
//
//
//#endif




static int
L2SqrI4x(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    int res = 0;
    unsigned char *a = (unsigned char *) pVect1;
    unsigned char *b = (unsigned char *) pVect2;

    qty = qty >> 2;
    for (size_t i = 0; i < qty; i++) {
        res += (((int32_t)(*a)) - (*b)) * (((int32_t)(*a)) - (*b));
        a++;
        b++;
        res += (((int32_t)(*a)) - (*b)) * (((int32_t)(*a)) - (*b));
        a++;
        b++;
        res += (((int32_t)(*a)) - (*b)) * (((int32_t)(*a)) - (*b));
        a++;
        b++;
        res += (((int32_t)(*a)) - (*b)) * (((int32_t)(*a)) - (*b));
        a++;
        b++;
    }
    return (res);
}

static int L2SqrI(const void* __restrict pVect1, const void* __restrict pVect2, const void* __restrict qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    int res = 0;
    unsigned char* a = (unsigned char*)pVect1;
    unsigned char* b = (unsigned char*)pVect2;

    for (size_t i = 0; i < qty; i++) {
        res += ((*a) - (*b)) * ((*a) - (*b));
        a++;
        b++;
    }
    return (res);
}

class L2SpaceI : public SpaceInterface<int> {
    DISTFUNC<int> fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2SpaceI(size_t dim) {
        if (dim % 4 == 0) {
            // fstdistfunc_ = L2SqrISIMD16ExtAVX512;
            fstdistfunc_ = L2SqrI4x;
        } else {
            fstdistfunc_ = L2SqrI;
        }
        dim_ = dim;
        data_size_ = dim * sizeof(unsigned char);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC<int> get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

    ~L2SpaceI() {}
};
}  // namespace hnswlib
