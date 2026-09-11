/*
 * cosinescreen.c
 * ---------------
 * Fixed q=2 baseline. Bigrams encoded as (c1<<8)|c2 into a 65536-entry
 * flat array. O(m) setup/teardown, integer Cauchy-Schwarz screening
 * test (no sqrt in the hot loop).
 */

#include "include/define.h"
#include "include/main.h"
#include <math.h>

#define TABLE_SIZE 65536

int search(unsigned char *x, int m, unsigned char *y, int n) {
    int count;
    BEGIN_PREPROCESSING

    static int patternCounts[TABLE_SIZE];
    static int windowCounts[TABLE_SIZE];
    int i, j;
    long dot, windowMagSq, patternMagSq;

    if (m < 2 || m > n) {
        END_PREPROCESSING
        BEGIN_SEARCHING
        count = 0;
        if (m == 1) {
            for (j = 0; j <= n - m; j++) if (y[j] == x[0]) OUTPUT(j);
        }
        END_SEARCHING
        return count;
    }

    patternMagSq = 0;
    for (i = 0; i + 1 < m; i++) {
        int key = (x[i] << 8) | x[i + 1];
        int old = patternCounts[key];
        patternMagSq += (2L * old + 1);
        patternCounts[key] = old + 1;
    }

    dot = 0;
    windowMagSq = 0;
    for (i = 0; i + 1 < m; i++) {
        int key = (y[i] << 8) | y[i + 1];
        int oldWv = windowCounts[key];
        dot += patternCounts[key];
        windowMagSq += (2L * oldWv + 1);
        windowCounts[key] = oldWv + 1;
    }

    END_PREPROCESSING

    BEGIN_SEARCHING
    count = 0;

    for (j = 0; j <= n - m; j++) {
        if (dot > 0 && dot * dot == windowMagSq * patternMagSq) {
            for (i = 0; i < m && x[i] == y[i + j]; ++i);
            if (i >= m) OUTPUT(j);
        }

        if (j < n - m) {
            int lk = (y[j] << 8) | y[j + 1];
            int ek = (y[j + m - 1] << 8) | y[j + m];

            int oldL = windowCounts[lk];
            dot -= patternCounts[lk];
            windowMagSq -= (2L * oldL - 1);
            windowCounts[lk] = oldL - 1;

            int oldE = windowCounts[ek];
            dot += patternCounts[ek];
            windowMagSq += (2L * oldE + 1);
            windowCounts[ek] = oldE + 1;
        }
    }
    END_SEARCHING

    for (i = 0; i + 1 < m; i++) patternCounts[(x[i] << 8) | x[i + 1]] = 0;
    {
        int lastStart = n - m;
        for (i = 0; i + 1 < m; i++) {
            int key = (y[lastStart + i] << 8) | y[lastStart + i + 1];
            windowCounts[key] = 0;
        }
    }

    return count;
}
