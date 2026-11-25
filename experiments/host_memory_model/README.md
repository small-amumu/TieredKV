# Host-memory roofline and Pareto model (RFP)

[`rfp_model.py`](rfp_model.py) is an analytic model of host memory against per-step decode time for the tiered
cache. It covers:
- bytes per token in each tier;
- fetch time per step, for per-row and page reads;
- overlap with the scoring of the GPU base;
- the effect of demotion and caching.

Comments in the code call it "RFP".

It is calibrated on measured data: the algorithm-level kernel profile on the RTX 4090 (Llama-3.1-8B shape, CUDA
graph, `h54` / `h33`), and the fetch-timing, cache-hit and quality runs. It runs on a CPU in seconds and needs no
model.

[`rfp_t3_cpu.py`](rfp_t3_cpu.py) replaces the model's assumed CPU throughput with the throughput measured in
[`read_path`](../read_path), and recomputes the CPU-side-correction rows
([`results/rfp_t3_cpu_lab05.md`](results/rfp_t3_cpu_lab05.md)).

Run: `python rfp_model.py`, then `python rfp_t3_cpu.py <t3 summary>`.
