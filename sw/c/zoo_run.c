/*
 * zoo_run.c -- instruction-set simulator of the microcoded zoo engine in plain C99.
 *
 * Executes the same microcode program and memory images as rtl/zoo/zoo_core.sv
 * (mem/zoo/<model>/: prog.mem weights.mem params.mem tables.mem luts.mem exp.mem
 * zoo_pkg.sv) with the same integer semantics as sw/zoo/quant.py:IntRunner.
 * Uses: (1) a third, independent implementation of the integer specification
 * (checked against the golden vectors tv.txt), (2) the CPU baseline: the same
 * integer work on the PYNQ-Z2's ARM Cortex-A9 or on a desktop CPU.
 *
 *   cc -O3 -o zoo_run zoo_run.c
 *   ./zoo_run mem/zoo/gpt2                 check every logit of tv.txt, then time it
 *   ./zoo_run mem/zoo/stories260k --gen 200   greedy generation from BOS (token 1)
 * Assumes arithmetic right shift of negative integers (gcc/clang on ARM and x86).
 */
#define _POSIX_C_SOURCE 200809L   /* clock_gettime with -std=c99 on glibc (PYNQ Linux) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { OP_END, OP_EMB16, OP_EMB8, OP_EMBWIN, OP_TABLOGIT, OP_HRESET, OP_NORM, OP_GEMV, OP_ROPE,
       OP_KVSTORE, OP_ATTN, OP_MUL, OP_GRUUPD, OP_COPY, OP_REQ, OP_BOW };
enum { B_HS = 10, B_R = 14, B_L = 15 };
enum { M_O8, M_RADD, M_LUT, M_LOGIT };
#define NBUF 11

/* ------------------------------------------------------------------ model */
static int P, V, T, NL, C, VLEN, KVD, TAB_W, PROG_DEPTH, W_DEPTH, PRM_DEPTH, TAB_DEPTH, LUT_DEPTH;
static uint64_t (*prog)[4];      /* 256-bit instructions as four 64-bit limbs, limb 0 = bits 63:0 */
static int8_t *wmem;             /* W_DEPTH words of P bytes (lane p = output p of a block) */
static uint64_t *pmem;
static int32_t *tab;
static uint8_t *lut;
static uint16_t explut[256];

/* ------------------------------------------------------------------ state */
static int8_t (*buf)[1];         /* NBUF x VLEN, accessed through B() */
static int16_t *R, *L;
static int8_t *Kc, *Vc;          /* NL x T x KVD */
static int32_t *S;
static int *hist;
#define B(b, i) (((int8_t *)buf)[(size_t)(b) * VLEN + (i)])

static void die(const char *m, const char *a) { fprintf(stderr, "zoo_run: %s %s\n", m, a ? a : ""); exit(2); }

static FILE *open_in(const char *dir, const char *f) {
  char p[1024];
  snprintf(p, sizeof p, "%s/%s", dir, f);
  FILE *fp = fopen(p, "r");
  if (!fp) die("cannot open", p);
  return fp;
}

static int pkg_int(const char *dir, const char *key) {
  FILE *fp = open_in(dir, "zoo_pkg.sv");
  char line[512], name[128];
  int v;
  while (fgets(line, sizeof line, fp))
    if (sscanf(line, " localparam int %127s = %d;", name, &v) == 2 && !strcmp(name, key)) { fclose(fp); return v; }
  fclose(fp);
  die("missing parameter", key);
  return 0;
}

/* read n hex words of `bits` bits; returns limbs (bits <= 64) */
static void read_hex64(const char *dir, const char *f, uint64_t *dst, int n) {
  FILE *fp = open_in(dir, f);
  char s[128];
  for (int i = 0; i < n; i++) {
    if (fscanf(fp, "%127s", s) != 1) die("short file", f);
    dst[i] = strtoull(s, NULL, 16);
  }
  fclose(fp);
}

static void load(const char *dir) {
  P = pkg_int(dir, "P"); V = pkg_int(dir, "V"); T = pkg_int(dir, "T"); NL = pkg_int(dir, "NL");
  C = pkg_int(dir, "C"); VLEN = pkg_int(dir, "VLEN"); KVD = pkg_int(dir, "KVD"); TAB_W = pkg_int(dir, "TAB_W");
  PROG_DEPTH = pkg_int(dir, "PROG_DEPTH"); W_DEPTH = pkg_int(dir, "W_DEPTH"); PRM_DEPTH = pkg_int(dir, "PRM_DEPTH");
  TAB_DEPTH = pkg_int(dir, "TAB_DEPTH"); LUT_DEPTH = pkg_int(dir, "LUT_DEPTH");
  if (P < 8 || P > 64 || (P & (P - 1))) die("P must be a power of two in [8, 64]", NULL);

  prog = calloc(PROG_DEPTH, sizeof *prog);
  FILE *fp = open_in(dir, "prog.mem");
  char s[128];
  for (int i = 0; i < PROG_DEPTH; i++) {
    if (fscanf(fp, "%127s", s) != 1 || strlen(s) != 64) die("bad prog.mem", NULL);
    for (int l = 0; l < 4; l++) {               /* most significant limb first in the text */
      char part[17];
      memcpy(part, s + 16 * (3 - l), 16);
      part[16] = 0;
      prog[i][l] = strtoull(part, NULL, 16);
    }
  }
  fclose(fp);
  wmem = calloc((size_t)W_DEPTH * P, 1);
  fp = open_in(dir, "weights.mem");
  char ws[160];
  for (int i = 0; i < W_DEPTH; i++) {               /* 2P hex digits, lane 0 = least significant byte */
    if (fscanf(fp, "%159s", ws) != 1 || (int)strlen(ws) != 2 * P) die("bad weights.mem", NULL);
    for (int p = 0; p < P; p++) {
      char byte[3] = {ws[2 * (P - 1 - p)], ws[2 * (P - 1 - p) + 1], 0};
      wmem[(size_t)i * P + p] = (int8_t)strtol(byte, NULL, 16);
    }
  }
  fclose(fp);
  pmem = calloc(PRM_DEPTH, 8); read_hex64(dir, "params.mem", pmem, PRM_DEPTH);
  uint64_t *tmp = calloc(TAB_DEPTH > LUT_DEPTH ? TAB_DEPTH : LUT_DEPTH, 8);
  read_hex64(dir, "tables.mem", tmp, TAB_DEPTH);
  tab = calloc(TAB_DEPTH, 4);
  for (int i = 0; i < TAB_DEPTH; i++)           /* sign-extend from TAB_W bits */
    tab[i] = (int32_t)((int64_t)(tmp[i] << (64 - TAB_W)) >> (64 - TAB_W));
  read_hex64(dir, "luts.mem", tmp, LUT_DEPTH);
  lut = malloc(LUT_DEPTH);
  for (int i = 0; i < LUT_DEPTH; i++) lut[i] = (uint8_t)tmp[i];
  uint64_t e[256];
  read_hex64(dir, "exp.mem", e, 256);
  for (int i = 0; i < 256; i++) explut[i] = (uint16_t)e[i];
  free(tmp);

  buf = calloc((size_t)NBUF * VLEN, 1);
  R = calloc(C, 2); L = calloc(V, 2); S = calloc(C, 4); hist = calloc(T, sizeof(int));
  Kc = calloc((size_t)NL * T * KVD, 1); Vc = calloc((size_t)NL * T * KVD, 1);
}

/* ------------------------------------------------------------------ arithmetic */
static inline int64_t sat(int64_t v, int bits) {
  int64_t hi = (1LL << (bits - 1)) - 1, lo = -(1LL << (bits - 1));
  return v > hi ? hi : v < lo ? lo : v;
}
static inline int64_t rq(int64_t acc, int64_t m0, int sh, int bits) {
  return sat((acc * m0 + (1LL << (sh - 1))) >> sh, bits);
}
static uint64_t isqrt64(uint64_t n) {                 /* floor(sqrt(n)), exact */
  uint64_t r = 0, bit = 1ULL << 62;
  while (bit > n) bit >>= 2;
  while (bit) {
    if (n >= r + bit) { n -= r + bit; r = (r >> 1) + bit; } else r >>= 1;
    bit >>= 2;
  }
  return r;
}
static int bitlen(uint64_t v) { int n = 0; while (v) { n++; v >>= 1; } return n; }

static inline uint64_t field(const uint64_t *w, int off, int n) {
  uint64_t lo = w[off / 64] >> (off % 64);
  if (off % 64 + n > 64) lo |= w[off / 64 + 1] << (64 - off % 64);
  return n == 64 ? lo : lo & ((1ULL << n) - 1);
}

/* ------------------------------------------------------------------ one decode step */
static void step(int tok, int t) {
  hist[t] = tok;
  for (int pc = 0; pc < PROG_DEPTH; pc++) {
    const uint64_t *w = prog[pc];
    int op = (int)field(w, 0, 8), dst = (int)field(w, 8, 4), s0 = (int)field(w, 12, 4), s1 = (int)field(w, 16, 4);
    int mode = (int)field(w, 20, 4), aux = (int)field(w, 24, 8);
    int n0 = (int)field(w, 32, 16), n1 = (int)field(w, 48, 16), n2 = (int)field(w, 64, 16), n3 = (int)field(w, 80, 16);
    int64_t b0 = (int64_t)field(w, 96, 32), b1 = (int64_t)field(w, 128, 32);
    uint64_t k64 = field(w, 160, 64), e32 = field(w, 224, 32);
    switch (op) {
    case OP_END: return;
    case OP_EMB16:
      for (int c = 0; c < n0; c++)
        R[c] = (int16_t)sat((int64_t)tab[b0 + (int64_t)tok * n0 + c] + ((mode & 1) ? tab[b1 + (int64_t)t * n0 + c] : 0), 16);
      break;
    case OP_EMB8:
      for (int c = 0; c < n0; c++) B(dst, c) = (int8_t)tab[b0 + (int64_t)tok * n0 + c];
      break;
    case OP_EMBWIN:
      for (int j = 0; j < n1; j++) {
        int row = t >= j ? hist[t - j] : n2;
        for (int c = 0; c < n0; c++) B(dst, j * n0 + c) = (int8_t)tab[b0 + (int64_t)row * n0 + c];
      }
      break;
    case OP_TABLOGIT:
      for (int v = 0; v < n0; v++) L[v] = (int16_t)tab[b0 + (int64_t)tok * n0 + v];
      break;
    case OP_HRESET:
      if (t == 0) for (int i = 0; i < n0; i++) B(B_HS, i) = (int8_t)tab[b0 + i];
      break;
    case OP_NORM: {
      int ln = mode & 1, src16 = mode & 2, dst16 = mode & 4, n = n0, shift = aux;
      uint64_t K = k64, E = e32;
      for (int seg = 0; seg < n1; seg++) {
        int base = seg * n;
        int64_t s = 0, sq = 0;
        for (int j = 0; j < n; j++) {
          int64_t x = src16 ? R[base + j] : B(s0, base + j);
          s += x; sq += x * x;
        }
        uint64_t Q = ln ? (uint64_t)n * ((uint64_t)n * (uint64_t)sq - (uint64_t)(s * s)) + E : (uint64_t)sq + E;
        int bl = bitlen(Q), k = bl < 40 ? (40 - bl) / 2 : 0;
        uint64_t rt = isqrt64(Q << (2 * k)), inv = (K << k) / rt;
        for (int j = 0; j < n; j++) {
          int64_t x = src16 ? R[base + j] : B(s0, base + j);
          int64_t d = ln ? (int64_t)n * x - s : x;
          int64_t y = (d * (int64_t)inv + (1LL << (shift - 1))) >> shift;
          if (dst16) R[base + j] = (int16_t)sat(y, 16);
          else B(dst, base + j) = (int8_t)sat(y, 8);
        }
      }
      break;
    }
    case OP_GEMV: {
      int nin = n0 + n1, nblk = n2 / P;
      static int8_t x[65536];
      for (int i = 0; i < n0; i++) x[i] = B(s0, i);
      for (int i = 0; i < n1; i++) x[n0 + i] = B(s1, i);
      const int8_t *wp = wmem + (size_t)b0 * P;
      for (int ob = 0; ob < nblk; ob++) {
        int64_t acc[64] = {0};
        for (int i = 0; i < nin; i++, wp += P) {
          int64_t xi = x[i];
          for (int p = 0; p < P; p++) acc[p] += wp[p] * xi;
        }
        for (int p = 0; p < P; p++) {
          int o = ob * P + p;
          if (o >= n3) continue;
          uint64_t pr = pmem[b1 + o];
          int64_t a = acc[p] + (int32_t)(pr >> 32);
          int64_t m0 = (pr >> 16) & 0xFFFF;
          int sh = (int)(pr & 0x3F);
          switch (mode) {
          case M_O8:    B(dst, o) = (int8_t)rq(a, m0, sh, 8); break;
          case M_RADD:  R[o] = (int16_t)sat(R[o] + rq(a, m0, sh, 16), 16); break;
          case M_LUT:   B(dst, o) = (int8_t)lut[aux * 256 + ((uint8_t)rq(a, m0, sh, 8))]; break;
          default:      L[o] = (int16_t)rq(a, m0, sh, 16); break;
          }
        }
      }
      break;
    }
    case OP_ROPE: {
      int nh = n0, dh = n1, hd = dh / 2, half = mode & 1;
      for (int h = 0; h < nh; h++) {
        int8_t orig[64];
        memcpy(orig, &B(s0, h * dh), dh);
        for (int i = 0; i < hd; i++) {
          int64_t c = tab[b0 + (int64_t)t * hd + i], s = tab[b1 + (int64_t)t * hd + i];
          int i1 = half ? i : 2 * i, i2 = half ? i + hd : 2 * i + 1;
          int64_t a = orig[i1], b = orig[i2], y1, y2;
          if (half) { y1 = (a * c + b * s + 8192) >> 14; y2 = (-a * s + b * c + 8192) >> 14; }
          else      { y1 = (a * c - b * s + 8192) >> 14; y2 = (a * s + b * c + 8192) >> 14; }
          B(dst, h * dh + i1) = (int8_t)sat(y1, 8);
          B(dst, h * dh + i2) = (int8_t)sat(y2, 8);
        }
      }
      break;
    }
    case OP_KVSTORE:
      memcpy(Kc + ((size_t)aux * T + t) * KVD, &B(s0, 0), n0);
      memcpy(Vc + ((size_t)aux * T + t) * KVD, &B(s1, 0), n0);
      break;
    case OP_ATTN: {
      int nh = n0, dh = n2, rep = n3, F = (int)(k64 & 0x3F);
      int64_t cexp = b1 & 0xFFFF;
      static int64_t sc[4096];
      static uint32_t e[4096];
      for (int h = 0; h < nh; h++) {
        int kv = h / rep;
        const int8_t *q = &B(s0, h * dh);
        int64_t m = 0;
        for (int j = 0; j <= t; j++) {
          const int8_t *kr = Kc + ((size_t)aux * T + j) * KVD + kv * dh;
          int64_t s = 0;
          for (int d = 0; d < dh; d++) s += (int64_t)q[d] * kr[d];
          sc[j] = s;
          if (j == 0 || s > m) m = s;
        }
        uint64_t E = 0;
        int64_t A[64] = {0};
        for (int j = 0; j <= t; j++) {
          uint64_t u = (uint64_t)(m - sc[j]) * (uint64_t)cexp, nn = u >> F, kk = (u >> (F - 8)) & 0xFF;
          e[j] = nn > 15 ? 0 : (uint32_t)(explut[kk] >> nn);
          E += e[j];
          const int8_t *vr = Vc + ((size_t)aux * T + j) * KVD + kv * dh;
          for (int d = 0; d < dh; d++) A[d] += (int64_t)e[j] * vr[d];
        }
        int64_t Rc = (int64_t)((1ULL << 32) / E);
        for (int d = 0; d < dh; d++) B(dst, h * dh + d) = (int8_t)sat((A[d] * Rc + (1LL << 31)) >> 32, 8);
      }
      break;
    }
    case OP_MUL: {
      int64_t m0 = (k64 >> 8) & 0xFFFF;
      int sh = (int)(k64 & 0x3F);
      for (int i = 0; i < n0; i++) B(dst, i) = (int8_t)rq((int64_t)B(s0, i) * B(s1, i), m0, sh, 8);
      break;
    }
    case OP_GRUUPD:
      for (int i = 0; i < n0; i++) {
        int64_t z = B(s0, i), hb = B(s1, i), h = B(B_HS, i);
        B(B_HS, i) = (int8_t)sat(h + ((z * (hb - h) + 64) >> 7), 8);
      }
      break;
    case OP_COPY:
      for (int i = 0; i < n0; i++) B(dst, i) = B(s0, i);
      break;
    case OP_REQ: {
      int64_t m0 = (k64 >> 8) & 0xFFFF;
      int sh = (int)(k64 & 0x3F);
      for (int i = 0; i < n0; i++) B(dst, i) = (int8_t)rq(R[i], m0, sh, 8);
      break;
    }
    case OP_BOW: {
      int64_t rcp = ((int64_t)tab[b0 + 2 * t] << 16) | (tab[b0 + 2 * t + 1] & 0xFFFF);  /* two 16-bit words */
      for (int c = 0; c < n0; c++) {
        S[c] = (t == 0 ? 0 : S[c]) + R[c];
        int64_t avg = ((int64_t)S[c] * rcp + (1LL << 19)) >> 20;
        R[c] = (int16_t)sat(R[c] + avg, 16);
      }
      break;
    }
    default: die("unknown opcode", NULL);
    }
  }
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <memdir> [--gen N] [--reps N]\n", argv[0]); return 2; }
  const char *dir = argv[1];
  int gen = 0, reps = 3;
  for (int i = 2; i < argc - 1; i++) {
    if (!strcmp(argv[i], "--gen")) gen = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--reps")) reps = atoi(argv[i + 1]);
  }
  load(dir);

  if (gen) {                                          /* greedy generation from BOS */
    int tok = 1;
    printf("%d", tok);
    double t0 = now();
    for (int t = 0; t < gen && t < T; t++) {
      step(tok, t);
      int best = 0;
      for (int v = 1; v < V; v++) if (L[v] > L[best]) best = v;
      tok = best;
      printf(" %d", tok);
    }
    printf("\n");
    fprintf(stderr, "%.1f us/token\n", 1e6 * (now() - t0) / gen);
    return 0;
  }

  /* 1) bit-exact check against the golden vectors */
  FILE *fp = open_in(dir, "tv.txt");
  int nseq, nv;
  if (fscanf(fp, "%d %d", &nseq, &nv) != 2 || nv != V) die("bad tv.txt", NULL);
  int **seqs = calloc(nseq, sizeof *seqs), *lens = calloc(nseq, sizeof(int));
  long steps = 0, errors = 0;
  for (int s = 0; s < nseq; s++) {
    if (fscanf(fp, "%d", &lens[s]) != 1) die("bad tv.txt", NULL);
    seqs[s] = calloc(lens[s], sizeof(int));
    for (int t = 0; t < lens[s]; t++) {
      if (fscanf(fp, "%d", &seqs[s][t]) != 1) die("bad tv.txt", NULL);
      step(seqs[s][t], t);
      for (int v = 0; v < V; v++) {
        int g;
        if (fscanf(fp, "%d", &g) != 1) die("bad tv.txt", NULL);
        if (g != L[v]) { if (errors < 5) printf("MISMATCH seq %d t %d logit %d: c %d gold %d\n", s, t, v, L[v], g); errors++; }
      }
      steps++;
    }
  }
  fclose(fp);

  /* 2) timing: the same token sequences, `reps` times */
  double best = 1e30;
  for (int r = 0; r < reps; r++) {
    double t0 = now();
    for (int s = 0; s < nseq; s++)
      for (int t = 0; t < lens[s]; t++) step(seqs[s][t], t);
    double dt = now() - t0;
    if (dt < best) best = dt;
  }
  printf("RESULT %s steps %ld errors %ld us_per_token %.3f\n", dir, steps, errors, 1e6 * best / steps);
  printf(errors ? "FAIL\n" : "PASS: C simulator is bit-exact with the integer reference\n");
  return errors != 0;
}
