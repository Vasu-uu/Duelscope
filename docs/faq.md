# FAQ

**My GPU is not used.** Run `duelscope doctor`. The usual cause is a CPU-only PyTorch build; it prints the install command. ONNX needs `onnxruntime-gpu`.

**Why does my mAP differ from `model.val()`?** See [metrics.md](metrics.md): different letterbox padding and a slightly different AP integration.

**Class names do not match my dataset.** Add `class_map: {model_name: dataset_name}` to the config. Unmapped classes are listed in the warnings.

**Can I rerun fast?** Predictions are cached in `.duelscope_cache/` keyed by weights hash, dataset content and settings. `--no-cache` bypasses it; `--max-images 20` gives a quick smoke run; `duelscope report --from results.json` rebuilds reports with no inference.

**Does it need internet?** Only to download pretrained weights the first time.

**Is the HTML really standalone?** Yes: CSS, JS, charts and thumbnails are inline, with no CDN. The `gallery/` folder next to it is only needed to rebuild the report and for the Markdown links.
