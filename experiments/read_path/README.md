# Host read path and CPU-side correction

**Question.** What does it cost to read the selected residual rows from host memory, and which read path should
be used: per-row zero-copy gather, whole-page DMA, or a mix? Could the correction run on the CPU instead, with only
small results crossing PCIe?

**Protocol**
- **Read paths** ([`rp_probe.py`](rp_probe.py)):
  - the same pinned layout and rows for every path;
  - sweeps over page size (8-64 rows), selected rows per page, K and V, batch 1 / 8 and working set;
  - 4 KB vs 2 MB host pages;
  - read-back checked byte for byte.
- **CPU-side correction** ([`rp_cpu.py`](rp_cpu.py), [`rp_cpu_ext.cpp`](rp_cpu_ext.cpp)): throughput of the
  residual decode and products on the CPU, and GPU-to-CPU round-trip latency.

**Result** ([`results/rp_star.md`](results/rp_star.md), [`results/rp_lab05.md`](results/rp_lab05.md),
[`results/t3_cpu_star.md`](results/t3_cpu_star.md))
- **Huge pages, RTX 4090 host:** 2 MB host pages make per-row zero-copy reads 1.9-5x faster.
- **Huge pages, RTX 5090 D host:** no difference up to 16K tokens, some gain from 32K up.
- **Whole-page reads:** whole-page zero-copy reads only match per-row reads when every row of the page is selected
  (K: 0.93-1.03x). With the measured clustering of about 2.2 rows per touched 16-row page, per-row reads win.
- **CPU-side correction:**
  - 32 sequences take 29.2 ms per step on 32 physical cores, about the length of a whole GPU decode step;
  - a GPU-to-CPU round trip costs 13 us (1 sequence) / 61 us (8 sequences).

Original invocation: `bash rp_run.sh <run_dir> <gpu>`; CPU part: `bash rp_t3_run.sh <run_dir> cpu|gpu <gpu>`.
`star_night_chain.sh` ran both on the RTX 4090 host. `rp_analyze.py` and `t3_summary.py` build the tables.
