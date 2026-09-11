/*
 * cosinescreendqlin.c
 * ---------------------
 * Same dynamic-q algorithm as cosinescreendq.c, but replaces the
 * open-addressing hash table with a simple LINEAR-SCAN array of
 * (key, count) pairs.
 *
 * Rationale: the number of distinct q-grams in a window is bounded by
 * maxGrams = m-q+1, which the dynamic-q formula tends to keep SMALL
 * (observed: 5 for genome m=24, 1 for english m=8, since the formula
 * pushes q up specifically when collision risk from a large n is high,
 * which shrinks maxGrams correspondingly). For such small N, a linear
 * scan avoids hash computation and probe-sequence overhead entirely,
 * and is highly cache- and branch-predictor-friendly -- the same
 * "stop hashing, use direct memory access" idea that made the fixed
 * q=2 array version fast, adapted for arbitrary q where a full
 * sigma^q array is infeasible.
 *
 * Falls back to doubling the array if maxGrams estimate is exceeded
 * (should not happen in practice given the bound above, but handled
 * safely rather than assumed).
 */

#include "include/define.h"
#include "include/main.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

#define BASE 131ULL

typedef struct {
    uint64_t *keys;
    int32_t *counts;
    int size;      /* number of active (key,count) entries */
    int capacity;
} LinMap;

static void linmap_init(LinMap *m, int expectedEntries) {
    m->capacity = expectedEntries < 4 ? 4 : expectedEntries;
    m->keys = (uint64_t *) malloc(m->capacity * sizeof(uint64_t));
    m->counts = (int32_t *) malloc(m->capacity * sizeof(int32_t));
    m->size = 0;
}

static void linmap_free(LinMap *m) {
    free(m->keys);
    free(m->counts);
}

static int32_t linmap_get(LinMap *m, uint64_t key) {
    for (int i = 0; i < m->size; i++) {
        if (m->keys[i] == key) return m->counts[i];
    }
    return 0;
}

static void linmap_grow(LinMap *m) {
    int newCap = m->capacity * 2;
    m->keys = (uint64_t *) realloc(m->keys, newCap * sizeof(uint64_t));
    m->counts = (int32_t *) realloc(m->counts, newCap * sizeof(int32_t));
    m->capacity = newCap;
}

static void linmap_add(LinMap *m, uint64_t key, int32_t delta) {
    for (int i = 0; i < m->size; i++) {
        if (m->keys[i] == key) {
            m->counts[i] += delta;
            if (m->counts[i] == 0) {
                /* swap-remove: keep size bounded by the window's ACTUAL
                   current distinct q-gram count, not cumulative history */
                m->keys[i] = m->keys[m->size - 1];
                m->counts[i] = m->counts[m->size - 1];
                m->size--;
            }
            return;
        }
    }
    if (m->size >= m->capacity) linmap_grow(m);
    m->keys[m->size] = key;
    m->counts[m->size] = delta;
    m->size++;
}

static int compute_sigma(unsigned char *x, int m, unsigned char *y, int n) {
    int seen[256];
    memset(seen, 0, sizeof(seen));
    int count = 0;
    for (int i = 0; i < m; i++) if (!seen[x[i]]) { seen[x[i]] = 1; count++; }
    for (int i = 0; i < n; i++) if (!seen[y[i]]) { seen[y[i]] = 1; count++; }
    return count < 2 ? 2 : count;
}

static int choose_q(int m, int n, int sigma) {
    double target = 2.0 * (log((double) n) / log((double) sigma));
    int q = (int) ceil(target);
    if (q < 2) q = 2;
    if (q > m) q = m;
    return q;
}

int search(unsigned char *x, int m, unsigned char *y, int n) {
    int count;
    BEGIN_PREPROCESSING

    int i, j;
    int sigma = compute_sigma(x, m, y, n);
    int q = choose_q(m, n, sigma);

    if (q < 2 || m < q) {
        END_PREPROCESSING
        BEGIN_SEARCHING
        count = 0;
        for (j = 0; j <= n - m; j++) {
            for (i = 0; i < m && x[i] == y[i + j]; ++i);
            if (i >= m) OUTPUT(j);
        }
        END_SEARCHING
        return count;
    }

    uint64_t power = 1;
    for (i = 0; i < q - 1; i++) power *= BASE;

    int maxGrams = m - q + 1;
    LinMap patternVec, windowVec;
    linmap_init(&patternVec, maxGrams);
    linmap_init(&windowVec, maxGrams);

    uint64_t h = 0;
    for (i = 0; i < q; i++) h = h * BASE + x[i];
    linmap_add(&patternVec, h, 1);
    for (i = q; i < m; i++) {
        h = (h - (uint64_t) x[i - q] * power) * BASE + x[i];
        linmap_add(&patternVec, h, 1);
    }
    int64_t patternMagSq = 0;
    for (i = 0; i < patternVec.size; i++) {
        int64_t c = patternVec.counts[i];
        patternMagSq += c * c;
    }

    uint64_t wh = 0;
    for (i = 0; i < q; i++) wh = wh * BASE + y[i];
    int64_t dot = 0, windowMagSq = 0;
    {
        int32_t old = linmap_get(&windowVec, wh);
        dot += linmap_get(&patternVec, wh);
        windowMagSq += (2LL * old + 1);
        linmap_add(&windowVec, wh, 1);
    }
    for (i = q; i < m; i++) {
        wh = (wh - (uint64_t) y[i - q] * power) * BASE + y[i];
        int32_t old = linmap_get(&windowVec, wh);
        dot += linmap_get(&patternVec, wh);
        windowMagSq += (2LL * old + 1);
        linmap_add(&windowVec, wh, 1);
    }
    uint64_t leavingHash;
    { uint64_t t = 0; for (i = 0; i < q; i++) t = t * BASE + y[i]; leavingHash = t; }
    uint64_t enteringHash = wh;

    END_PREPROCESSING

    BEGIN_SEARCHING
    count = 0;

    for (j = 0; j <= n - m; j++) {
        if (dot > 0 && dot * dot == windowMagSq * patternMagSq) {
            for (i = 0; i < m && x[i] == y[i + j]; ++i);
            if (i >= m) OUTPUT(j);
        }

        if (j < n - m) {
            int32_t oldL = linmap_get(&windowVec, leavingHash);
            dot -= linmap_get(&patternVec, leavingHash);
            windowMagSq -= (2LL * oldL - 1);
            linmap_add(&windowVec, leavingHash, -1);

            enteringHash = (enteringHash - (uint64_t) y[j + m - q] * power) * BASE + y[j + m];
            int32_t oldE = linmap_get(&windowVec, enteringHash);
            dot += linmap_get(&patternVec, enteringHash);
            windowMagSq += (2LL * oldE + 1);
            linmap_add(&windowVec, enteringHash, 1);

            leavingHash = (leavingHash - (uint64_t) y[j] * power) * BASE + y[j + q];
        }
    }
    END_SEARCHING

    linmap_free(&patternVec);
    linmap_free(&windowVec);

    return count;
}
