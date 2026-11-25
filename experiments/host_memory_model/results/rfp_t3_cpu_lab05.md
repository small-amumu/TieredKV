# RFP near-data CPU correction with T3 measured CPU times (lab05)

- T3 input: `t3b_cpu.jsonl`, host lab05 (AMD EPYC 7K62 48-Core Processor, 96 logical CPUs, 1 NUMA node), rp_cpu.py 08d4d0c6, rp_cpu_ext.cpp ecb4aeea.
- CPU time per layer = median over timed (layer, step) of round 1 + round 2, all 8 KV heads and B sequences, rows mostly from DRAM (pool >> LLC, ids never reused). It is the time of this simple C++/OpenMP implementation (compiler auto-vectorized, no hand-written SIMD decode), not a hardware limit: one thread does 1.6 GMAC/s (3.2 GFLOP/s-eq) at B1.
- The records emulate a 32K context with row ids never reused (worst case for the cache). The same CPU time is used in the 4K columns. h54 residual pool vs the 192 MiB LLC: 4K B1 152 MiB, 4K B8 1,216 MiB, 4K B32 4,864 MiB. At 4K B8 / B32 rows still come from DRAM; at 4K B1 the pool can sit in the LLC, and in real decoding many selected rows repeat between steps (T1 trajectories: about 43 of the 128 first-stage rows change per step), so the real time can be lower. As a magnitude only: the one cache-resident control (B8, 16 threads, L=512, 1 region) takes 336.9 vs 426.9 us per layer (21% less), but it also differs in L and footprint.
- lab05 was not exclusive (load average 5-11 during the run, from other jobs) and is a different CPU from STAR, the platform the model is calibrated on (STAR: 2 x Xeon Platinum 8336C, 64 cores / 128 threads, 2 NUMA nodes). STAR's own measurement is pending.
- Model: rfp_model.py bee587d8 unchanged; MAC per sequence-step 31.46 M. The step is composed serially, weights + attention + host, for both the deployed and the CPU paths, as in the model; overlap of the CPU work with GPU work and the CUDA-graph cost of a per-layer CPU hand-off are not modelled. The wrapper reproduces the report's eight numbers (deployed and assumed-CPU rows) within 1 tok/s before any measured value is used.
- RTT is an assumption (T3 GPU part not run yet): 2 round trips per layer x 32 layers. At 15 us the 64 round trips take 0.96 ms per step; the 32-thread CPU time is 3x that at B1, 10x that at B8, 26x that at B32. The RTT effect below is small for that reason and grows if the CPU part gets faster.
- The model's 640 GFLOP/s is 32 of STAR's 128 logical CPUs (64 physical cores) at about 20 GFLOP/s each. 32 threads here = 32 of lab05's 48 physical cores; 64 threads use SMT and exceed that budget.
- `x` = the batch does not fit in 24 GB at that length (model's capacity rule); the number is shown as in the RFP report.

## Measured CPU time per decode step (32 layers)

- GFLOP/s-eq = 2 x MAC per second, the model's convention (decode counted as D = 128 MAC per row).
- Each cell: ms per step = 32 x median layer time [32 x Q1, 32 x Q3 of the per-layer times]; GFLOP/s-eq; steps timed and layer samples (steps x 32). Timing stops at the 2 s floor or when the pool of never-reused row ids runs out, whichever comes first. The floor was reached at (B, threads) = [(1, 1), (8, 1), (32, 1)]; cells below that reached it: none (the others stopped at the id cap).

| threads | B1 | B8 | B32 |
|---|---|---|---|
| 32 | 3.06 [3.00, 3.13]; 21; 131 steps, 4192 samples | 9.33 [9.18, 11.30]; 54; 17 steps, 544 samples | 25.30 [25.15, 25.50]; 80; 8 steps, 256 samples |
| 64 | 3.36 [3.32, 3.40]; 19; 131 steps, 4192 samples | 8.39 [8.19, 8.76]; 60; 17 steps, 544 samples | 20.89 [20.55, 23.72]; 96; 8 steps, 256 samples |
| model assumption | 0.10 (640) | 0.79 (640) | 3.15 (640) |

## Throughput, tok/s per GPU (RTT assumed 15 us unless named)

| variant | 4K B1 | 4K B8 | 4K B32 | 32K B1 | 32K B8 | 32K B32 |
|---|---|---|---|---|---|---|
| deployed: serial zero-copy row gather | 46 | 259 | 517 | 42 | 167 | 247 x |
| CPU correction, model assumption (640 GFLOP/s) | 46 | 335 | 1034 | 43 | 237 | 454 x |
| CPU correction, T3 32K-footprint CPU time 32 threads, RTT 5 us | 41 | 252 | 610 | 39 | 192 | 348 x |
| CPU correction, T3 32K-footprint CPU time 32 threads, RTT 15 us | 40 | 247 | 603 | 38 | 189 | 346 x |
| CPU correction, T3 32K-footprint CPU time 32 threads, RTT 30 us | 39 | 240 | 592 | 37 | 185 | 342 x |
| CPU correction, T3 32K-footprint CPU time 64 threads, RTT 5 us | 41 | 259 | 666 | 39 | 196 | 366 x |
| CPU correction, T3 32K-footprint CPU time 64 threads, RTT 15 us | 40 | 254 | 657 | 38 | 193 | 363 x |
| CPU correction, T3 32K-footprint CPU time 64 threads, RTT 30 us | 38 | 247 | 645 | 37 | 189 | 359 x |
| INT4 weights + deployed host path | 123 | 466 | 665 | 98 | 235 | 277 x |
| INT4 weights + CPU correction, model assumption | 124 | 788 | 1859 | 107 | 400 | 564 x |
| INT4 weights + CPU correction, T3 32K-footprint CPU time 32 threads | 90 | 428 | 813 | 81 | 280 | 406 x |
| INT4 weights + CPU correction, T3 32K-footprint CPU time 64 threads | 88 | 451 | 915 | 79 | 290 | 430 x |

## Host time per step on the critical path (ms), RTT assumed 15 us

| variant | 4K B1 | 4K B8 | 4K B32 | 32K B1 | 32K B8 | 32K B32 |
|---|---|---|---|---|---|---|
| deployed: serial zero-copy row gather | 1.09 | 8.76 | 35.03 | 1.97 | 15.77 | 63.09 |
| CPU correction, model assumption | 1.06 | 1.75 | 4.11 | 1.06 | 1.75 | 4.11 |
| CPU correction, T3 32K-footprint CPU time 32 threads | 4.02 | 10.29 | 26.26 | 4.02 | 10.29 | 26.26 |
| CPU correction, T3 32K-footprint CPU time 64 threads | 4.32 | 9.35 | 21.85 | 4.32 | 9.35 | 21.85 |
