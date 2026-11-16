# Benchmarking methodology

- Speed runs are separate from accuracy runs and use a fixed, seeded subset of the dataset (`speed.images`, default 32).
- Images are decoded once up front, so latency covers each adapter's preprocess + inference + postprocess but not disk I/O.
- The detector is unloaded and reloaded first so peak GPU memory (`torch.cuda.max_memory_allocated`) includes weights; then it is warmed up (`warmup()` plus `speed.warmup` untimed calls).
- Each timed call is wrapped in `torch.cuda.synchronize()` when running on CUDA. `speed.iters` timed single-image calls give mean, std, p50, p90, p99, min, max; every extra batch size in `speed.batch_sizes` gets its own timed calls and a throughput figure.
- The breakdown comes from the adapter's `last_timing` (Ultralytics reports it natively; torchvision folds NMS into inference).
- If CUDA is requested but unavailable the run falls back to CPU and records a note; nothing crashes.
- Every report stores CPU, cores, RAM, GPU, driver and CUDA versions, OS, Python and library versions (`hardware` in `results.json`).
- Latency on shared or thermally limited machines is noisy: compare the std and p99, close other programs, and rerun. Numbers are only comparable across models run in the same report.
