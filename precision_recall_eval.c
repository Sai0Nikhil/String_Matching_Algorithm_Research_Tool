/*
 * precision_recall_eval.c
 * ------------------------
 * Extends the original edit-distance-vs-cosine correlation harness into a
 * full precision/recall/F1 evaluation, generalized to arbitrary q (not
 * hardcoded to bigrams), matching what cosinescreendqlin.c actually does.
 *
 * Methodology:
 *   1. Generate many trials: a base pattern, mutated by a random number of
 *      substitution edits (0 to max_ed), verified against a "true" edit
 *      distance via DP.
 *   2. Define "true positive class" as: true_ed <= E (a chosen relevance
 *      cutoff -- e.g. E=3 means "within 3 edits counts as a real
 *      approximate match we want to find").
 *   3. Sweep cosine similarity thresholds t. At each t, classify each
 *      trial as predicted-positive if cosine_sim >= t.
 *   4. Compute confusion matrix (TP/FP/TN/FN) and derive precision,
 *      recall, F1, false-positive rate at each threshold.
 *
 * KNOWN SCOPE LIMITATION (stated explicitly, not hidden):
 * Only substitution edits are tested here. Insertions/deletions shift
 * every subsequent q-gram and are known to be harder for q-gram-based
 * filters to handle gracefully -- this harness does not evaluate that
 * case. Worth stating as a limitation in the paper, not silently
 * expanding the claim beyond what's tested.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define BASE 131ULL

/* ---------- Standard Levenshtein edit distance (unchanged) ---------- */
int edit_distance(unsigned char *a, int la, unsigned char *b, int lb) {
    int **dp = malloc((la + 1) * sizeof(int *));
    for (int i = 0; i <= la; i++) dp[i] = malloc((lb + 1) * sizeof(int));
    for (int i = 0; i <= la; i++) dp[i][0] = i;
    for (int j = 0; j <= lb; j++) dp[0][j] = j;
    for (int i = 1; i <= la; i++)
        for (int j = 1; j <= lb; j++) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int del = dp[i - 1][j] + 1, ins = dp[i][j - 1] + 1, sub = dp[i - 1][j - 1] + cost;
            int m1 = del < ins ? del : ins;
            dp[i][j] = m1 < sub ? m1 : sub;
        }
    int result = dp[la][lb];
    for (int i = 0; i <= la; i++) free(dp[i]);
    free(dp);
    return result;
}

/* ---------- Generalized arbitrary-q cosine similarity ----------
 * Uses a simple linear-scan (key,count) map, same structure as
 * cosinescreendqlin.c, so this genuinely mirrors what the real
 * algorithm does rather than a hardcoded-bigram shortcut. */
typedef struct {
    uint64_t keys[256];
    int32_t counts[256];
    int size;
} SmallMap;

static void map_reset(SmallMap *m) { m->size = 0; }

static int32_t map_get(SmallMap *m, uint64_t key) {
    for (int i = 0; i < m->size; i++) if (m->keys[i] == key) return m->counts[i];
    return 0;
}

static void map_add(SmallMap *m, uint64_t key, int delta) {
    for (int i = 0; i < m->size; i++) {
        if (m->keys[i] == key) {
            m->counts[i] += delta;
            return;
        }
    }
    if (m->size < 256) {
        m->keys[m->size] = key;
        m->counts[m->size] = delta;
        m->size++;
    }
}

static void build_qgram_vec(SmallMap *vec, unsigned char *s, int len, int q) {
    map_reset(vec);
    if (len < q) return;
    uint64_t h = 0;
    for (int i = 0; i < q; i++) h = h * BASE + s[i];
    map_add(vec, h, 1);
    uint64_t power = 1;
    for (int i = 0; i < q - 1; i++) power *= BASE;
    for (int i = q; i < len; i++) {
        h = (h - (uint64_t) s[i - q] * power) * BASE + s[i];
        map_add(vec, h, 1);
    }
}

double cosine_sim_q(unsigned char *a, int la, unsigned char *b, int lb, int q) {
    SmallMap vecA, vecB;
    build_qgram_vec(&vecA, a, la, q);
    build_qgram_vec(&vecB, b, lb, q);

    long dot = 0, magA = 0, magB = 0;
    for (int i = 0; i < vecA.size; i++) {
        int32_t c = vecA.counts[i];
        magA += (long) c * c;
        dot += (long) c * map_get(&vecB, vecA.keys[i]);
    }
    for (int i = 0; i < vecB.size; i++) {
        int32_t c = vecB.counts[i];
        magB += (long) c * c;
    }
    if (magA == 0 || magB == 0) return 0.0;
    return dot / (sqrt((double) magA) * sqrt((double) magB));
}

/* ---------- Trial generation: random substitution mutations ---------- */
void mutate(unsigned char *base, unsigned char *out, int m, int target_ed, const char *alpha, int alphaLen) {
    memcpy(out, base, m);
    int positions[256];
    for (int i = 0; i < m; i++) positions[i] = i;
    for (int i = 0; i < target_ed && i < m; i++) {
        int swap = i + rand() % (m - i);
        int tmp = positions[i]; positions[i] = positions[swap]; positions[swap] = tmp;
        int pos = positions[i];
        char newc;
        do { newc = alpha[rand() % alphaLen]; } while (newc == out[pos]);
        out[pos] = newc;
    }
}

/* ---------- Main: sweep q, then for each q sweep threshold ---------- */
int main(int argc, char **argv) {
    srand(2024);
    const char *alpha = "acgt";
    int alphaLen = 4;
    int m = 20;
    int trials_per_ed = 300;
    int max_ed = 8;
    int relevance_cutoff = 3;      /* "true positive class" = true_ed <= 3 */
    int q_values[] = {2, 3, 5, 8}; /* sweep across a few representative q values */
    int num_q = 4;

    unsigned char base[256];
    for (int i = 0; i < m; i++) base[i] = alpha[rand() % alphaLen];

    /* Pre-generate all trials once (mutated string + true ED), reused across q and thresholds
       so results across q values are directly comparable on the same trial set. */
    typedef struct { unsigned char mutated[256]; int true_ed; } Trial;
    int total_trials = (max_ed + 1) * trials_per_ed;
    Trial *trials = malloc(total_trials * sizeof(Trial));
    int nt = 0;

    for (int target_ed = 0; target_ed <= max_ed; target_ed++) {
        for (int t = 0; t < trials_per_ed; t++) {
            unsigned char mutated[256];
            mutate(base, mutated, m, target_ed, alpha, alphaLen);
            int true_ed = edit_distance(base, m, mutated, m);
            trials[nt].true_ed = true_ed;
            memcpy(trials[nt].mutated, mutated, m);
            nt++;
        }
    }

    printf("=== Precision/Recall/F1 sweep ===\n");
    printf("Relevance cutoff: true_ed <= %d counts as a real match\n", relevance_cutoff);
    printf("Total trials: %d\n\n", nt);

    for (int qi = 0; qi < num_q; qi++) {
        int q = q_values[qi];
        printf("--- q = %d ---\n", q);
        printf("%-8s %-6s %-6s %-6s %-6s %-10s %-10s %-10s %-8s\n",
               "thresh", "TP", "FP", "TN", "FN", "precision", "recall", "F1", "FPR");

        for (double thresh = 0.50; thresh <= 1.001; thresh += 0.05) {
            int TP = 0, FP = 0, TN = 0, FN = 0;
            for (int i = 0; i < nt; i++) {
                int is_relevant = (trials[i].true_ed <= relevance_cutoff);
                double cos = cosine_sim_q(base, m, trials[i].mutated, m, q);
                int predicted_positive = (cos >= thresh);

                if (predicted_positive && is_relevant) TP++;
                else if (predicted_positive && !is_relevant) FP++;
                else if (!predicted_positive && !is_relevant) TN++;
                else FN++;
            }
            double precision = (TP + FP) > 0 ? (double) TP / (TP + FP) : 0.0;
            double recall = (TP + FN) > 0 ? (double) TP / (TP + FN) : 0.0;
            double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
            double fpr = (FP + TN) > 0 ? (double) FP / (FP + TN) : 0.0;

            printf("%-8.2f %-6d %-6d %-6d %-6d %-10.4f %-10.4f %-10.4f %-8.4f\n",
                   thresh, TP, FP, TN, FN, precision, recall, f1, fpr);
        }
        printf("\n");
    }

    free(trials);
    return 0;
}
