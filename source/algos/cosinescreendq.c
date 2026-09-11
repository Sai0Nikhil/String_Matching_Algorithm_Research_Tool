/*
 * cosinescreendq.c  (TIMING-INSTRUMENTED VERSION)
 * -----------------
 * Dynamic-q variant of cosinescreen.c: q is chosen per search call via
 * the birthday-bound formula q(m,n,sigma) = clamp(2, m, ceil(2*log_sigma(n))),
 * with sigma computed from the actual distinct bytes observed in text+pattern.
 *
 * Since q now varies (and can exceed 2), a fixed 65536-entry array is no
 * longer feasible (sigma^q can be astronomically large). Instead this uses:
 *   - a rolling 64-bit polynomial hash per q-gram (O(1) per shift, no
 *     substring allocation)
 *   - a hand-rolled open-addressing hash table (uint64_t keys, int32_t
 *     counts) instead of a language-level hash map, to avoid boxing
 *   - resize-and-purge when the table's load factor gets too high, which
 *     is REQUIRED: without it, every distinct q-gram seen across the
 *     entire sliding pass permanently occupies a slot, and the table
 *     silently fills and hangs on long texts.
 *   - the integer Cauchy-Schwarz screening test (no sqrt in the hot loop)
 *
 * --- INSTRUMENTATION NOTE ---
 * Added clock_gettime() timers around three phases:
 *   1. sigma+q selection  (compute_sigma + choose_q)
 *   2. preprocessing      (pattern vector + initial window vector build)
 *   3. searching          (the main sliding-window loop)
 * Timings are printed to stderr per search() call so SMART's stdout output
 * (used for the -occ summary) is untouched. Redirect stderr to a file to
 * collect per-pattern timing data across a full -pset run.
 * ---------------------------
 */

#include "include/define.h"
#include "include/main.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

#define EMPTY_KEY UINT64_MAX
#define BASE 131ULL

typedef struct {
    uint64_t *keys;
    int32_t *counts;
    uint32_t mask;      /* capacity - 1, capacity is a power of two */
    uint32_t used;      /* number of non-EMPTY slots (incl. stale zero-count) */
} PrimMap;

static void map_init(PrimMap *m, uint32_t expectedEntries) {
    uint32_t cap = 8;
    while (cap < expectedEntries * 4) cap <<= 1;
    m->keys = (uint64_t *) malloc(cap * sizeof(uint64_t));
    m->counts = (int32_t *) calloc(cap, sizeof(int32_t));
    for (uint32_t i = 0; i < cap; i++) m->keys[i] = EMPTY_KEY;
    m->mask = cap - 1;
    m->used = 0;
}

static void map_free(PrimMap *m) {
    free(m->keys);
    free(m->counts);
}

static uint32_t map_index(uint64_t *keys, uint32_t mask, uint64_t key) {
    uint32_t idx = (uint32_t) ((key ^ (key >> 32)) & mask);
    while (keys[idx] != EMPTY_KEY && keys[idx] != key) idx = (idx + 1) & mask;
    return idx;
}

static void map_resize_purge(PrimMap *m) {
    uint32_t newCap = (m->mask + 1) * 2;
    uint64_t *newKeys = (uint64_t *) malloc(newCap * sizeof(uint64_t));
    int32_t *newCounts = (int32_t *) calloc(newCap, sizeof(int32_t));
    for (uint32_t i = 0; i < newCap; i++) newKeys[i] = EMPTY_KEY;
    uint32_t newMask = newCap - 1;
    uint32_t newUsed = 0;

    for (uint32_t i = 0; i <= m->mask; i++) {
        if (m->keys[i] != EMPTY_KEY && m->counts[i] != 0) {
            uint32_t idx = map_index(newKeys, newMask, m->keys[i]);
            newKeys[idx] = m->keys[i];
            newCounts[idx] = m->counts[i];
            newUsed++;
        }
    }
    free(m->keys);
    free(m->counts);
    m->keys = newKeys;
    m->counts = newCounts;
    m->mask = newMask;
    m->used = newUsed;
}

static int32_t map_get(PrimMap *m, uint64_t key) {
    uint32_t idx = map_index(m->keys, m->mask, key);
    return m->keys[idx] == EMPTY_KEY ? 0 : m->counts[idx];
}

static void map_add(PrimMap *m, uint64_t key, int32_t delta) {
    uint32_t idx = map_index(m->keys, m->mask, key);
    if (m->keys[idx] == EMPTY_KEY) {
        m->keys[idx] = key;
        m->used++;
        if (m->used > (m->mask + 1) / 2) {
            map_resize_purge(m);
            idx = map_index(m->keys, m->mask, key);
            if (m->keys[idx] == EMPTY_KEY) { m->keys[idx] = key; m->used++; }
        }
    }
    m->counts[idx] += delta;
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

/* helper: milliseconds between two timespecs */
static double ms_between(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_nsec - a.tv_nsec) / 1e6;
}

int search(unsigned char *x, int m, unsigned char *y, int n) {
    int count;
    struct timespec t0, t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    BEGIN_PREPROCESSING

    int i, j;
    int sigma = compute_sigma(x, m, y, n);
    int q = choose_q(m, n, sigma);

    clock_gettime(CLOCK_MONOTONIC, &t1);  /* end of sigma+q selection */

    if (q < 2 || m < q) {
        /* degenerate: m too short for any q-gram, fall back to direct check */
        END_PREPROCESSING
        BEGIN_SEARCHING
        count = 0;
        for (j = 0; j <= n - m; j++) {
            for (i = 0; i < m && x[i] == y[i + j]; ++i);
            if (i >= m) OUTPUT(j);
        }
        END_SEARCHING
        fprintf(stderr, "[cosinescreendq] q=%d(degenerate) sigma_ms=%.4f preproc_ms=%.4f search_ms=%.4f\n",
                q, ms_between(t0, t1), 0.0, 0.0);
        return count;
    }

    uint64_t power = 1;
    for (i = 0; i < q - 1; i++) power *= BASE;

    int maxGrams = m - q + 1;
    PrimMap patternVec, windowVec;
    map_init(&patternVec, (uint32_t) maxGrams);
    map_init(&windowVec, (uint32_t) maxGrams);

    /* pattern vector, built with true rolling hash (O(m) total) */
    uint64_t h = 0;
    for (i = 0; i < q; i++) h = h * BASE + x[i];
    map_add(&patternVec, h, 1);
    for (i = q; i < m; i++) {
        h = (h - (uint64_t) x[i - q] * power) * BASE + x[i];
        map_add(&patternVec, h, 1);
    }
    int64_t patternMagSq = 0;
    for (uint32_t k = 0; k <= patternVec.mask; k++) {
        if (patternVec.keys[k] != EMPTY_KEY) {
            int64_t c = patternVec.counts[k];
            patternMagSq += c * c;
        }
    }

    /* initial window vector */
    uint64_t wh = 0;
    for (i = 0; i < q; i++) wh = wh * BASE + y[i];
    int64_t dot = 0, windowMagSq = 0;
    {
        int32_t old = map_get(&windowVec, wh);
        dot += map_get(&patternVec, wh);
        windowMagSq += (2LL * old + 1);
        map_add(&windowVec, wh, 1);
    }
    for (i = q; i < m; i++) {
        wh = (wh - (uint64_t) y[i - q] * power) * BASE + y[i];
        int32_t old = map_get(&windowVec, wh);
        dot += map_get(&patternVec, wh);
        windowMagSq += (2LL * old + 1);
        map_add(&windowVec, wh, 1);
    }
    uint64_t leavingHash = 0;
    { /* recompute hash of first q-gram for leaving pointer */
        uint64_t t = 0;
        for (i = 0; i < q; i++) t = t * BASE + y[i];
        leavingHash = t;
    }
    uint64_t enteringHash = wh;

    clock_gettime(CLOCK_MONOTONIC, &t2);  /* end of preprocessing */

    END_PREPROCESSING

    BEGIN_SEARCHING
    count = 0;

    for (j = 0; j <= n - m; j++) {
        if (dot > 0 && dot * dot == windowMagSq * patternMagSq) {
            for (i = 0; i < m && x[i] == y[i + j]; ++i);
            if (i >= m) OUTPUT(j);
        }

        if (j < n - m) {
            int32_t oldL = map_get(&windowVec, leavingHash);
            dot -= map_get(&patternVec, leavingHash);
            windowMagSq -= (2LL * oldL - 1);
            map_add(&windowVec, leavingHash, -1);

            enteringHash = (enteringHash - (uint64_t) y[j + m - q] * power) * BASE + y[j + m];
            int32_t oldE = map_get(&windowVec, enteringHash);
            dot += map_get(&patternVec, enteringHash);
            windowMagSq += (2LL * oldE + 1);
            map_add(&windowVec, enteringHash, 1);

            leavingHash = (leavingHash - (uint64_t) y[j] * power) * BASE + y[j + q];
        }
    }
    END_SEARCHING

    clock_gettime(CLOCK_MONOTONIC, &t3);  /* end of searching */

FILE *logf = fopen("/home/naren/smart/dq_debug.log", "a");
if (logf) {
    fprintf(logf, "q=%d maxGrams=%d sigma_ms=%.4f preproc_ms=%.4f search_ms=%.4f\n",
            q, maxGrams, ms_between(t0, t1), ms_between(t1, t2), ms_between(t2, t3));
    fclose(logf);
}

    map_free(&patternVec);
    map_free(&windowVec);

    return count;
}
