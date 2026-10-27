// CPU baseline: Karpathy's llama2.c forward() for stories260K under the board's conditions
// (256-token context, greedy argmax, one thread). Built by scripts/cpu_baseline.sh against run.c.
//   cpu_bench <model.bin> <T> <reps>
#define TESTING
#include "run.c"

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.bin T reps\n", argv[0]); return 1; }
    Transformer tr;
    build_transformer(&tr, argv[1]);
    int T = atoi(argv[2]), reps = atoi(argv[3]), V = tr.config.vocab_size;
    double sum = 0, best = 1e30;
    for (int r = 0; r < reps; r++) {
        int tok = 1;  // BOS, as on the board
        double t0 = now();
        for (int pos = 0; pos < T; pos++) {
            float *lg = forward(&tr, tok, pos);
            int a = 0;
            for (int i = 1; i < V; i++) if (lg[i] > lg[a]) a = i;
            tok = a;
        }
        double dt = now() - t0;
        sum += dt;
        if (dt < best) best = dt;
    }
    printf("{\"T\": %d, \"reps\": %d, \"tok_s_mean\": %.1f, \"tok_s_best\": %.1f, \"us_per_token\": %.3f}\n",
           T, reps, (double)T * reps / sum, T / best, 1e6 * sum / ((double)T * reps));
    free_transformer(&tr);
    return 0;
}
