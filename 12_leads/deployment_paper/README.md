# Deferred: embedded deployment material

Cut from the main manuscript on 2026-10-09. The main paper is now about
cross-source generalisation heterogeneity and selective reliability; the
STM32F411 work is a separate contribution and was making that paper do
two things at once.

Nothing here is regenerable from the manuscript build any more -- the
calls were removed from `scripts/s7_manuscript.py`. The analysis code is
intact and still rate-aware:

    scripts/s3b_int8.py                int8 PTQ (ONNX Runtime, QDQ)
    scripts/s8_latency.py              on-device timing over SWD
    scripts/s11_int8_preservation.py   dAUPRC and selective risk
    scripts/s12_confidence_lattice.py  the confidence lattice
    firmware/                          bare-metal timing firmware

Cached results stay under `results/study3b_100hz/` and
`results/study2_100hz/{int8_preservation,confidence_lattice}.json`, so
none of this needs retraining or reflashing to pick up again.

## What the material shows

- All six encoders run on an STM32F411 in int8: 44.8-508.8 KB Flash,
  27.3-53.5 KB SRAM, 222-4247 ms measured latency, RTF 0.022-0.425.
- Five of six fit a realistic envelope reserving 64 KB Flash / 16 KB
  SRAM; ResNet1D-Lite is Flash-limited at 508.8 of 512 KB.
- MACC mis-ranks measured latency by about 2x.
- int8 costs 0.0012-0.0032 macro AUPRC.
- **The strongest result:** int8 collapses confidence resolution from
  ~16,500 distinct values per fold to about 24, so the attainable
  selective-coverage lattice is coarse. Of 24 model-by-source folds, 20
  cannot reach a nominal 90% operating point within two percentage
  points, and in 8 the nearest alternative to withholding the
  least-confident tenth is withholding nothing. The float32 control
  shows coverage error 0.0000. Confidence resolution correlates with
  model size (r = +0.87 with AUPRC), so the cheapest encoders have the
  coarsest lattice.

That last point is the natural spine of the separate paper: quantisation
preserves what a model predicts and destroys how finely it can rank,
which bounds the abstention policies an MCU deployment can realise.
