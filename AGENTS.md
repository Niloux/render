# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Python 3D Gaussian rendering pipeline. `render_manager.py` is the public orchestration layer for model loading, `init`, and `render_frame`. Camera rendering lives in `camera_renderer.py`, lidar rendering in `lidar_renderer.py`, Gaussian buffers in `gaussian_buffers.py`, runtime validation/pose helpers in `render_runtime.py`, and low-level kernels in `render_kernel.py`. `data_types.py` defines request/response objects, while `rgb_decoder.py` and `mlp_decoder.py` implement decoder modules. Model abstractions are under `models/`. Utility scripts live in `scripts/`; `check_pth.py` validates checkpoints and `combine.py` copies actor components between checkpoints. `test_render.py` is the primary benchmark entry point. Generated artifacts such as `output/`, `*.pth`, `*.ply`, and local JSON data are ignored.

## Build, Test, and Development Commands

No package manager or project-level build file is currently defined. Use the project conda environment, `render`, which should provide `torch`, `numpy`, and `gsplat`.

```bash
conda activate render
python test_render.py --model-path 049_cnn_0421.pth --frames 10
python test_render.py --model-path 049_cnn_0421.pth --no-save --camera-count 1
python scripts/check_pth.py 049_cnn_0421.pth --components
python scripts/combine.py --source actors.pth --target scene.pth --output merged.pth --actors obj_010
```

`test_render.py` runs the render benchmark and optional first-frame output. `scripts/check_pth.py` checks fields required by `GSModel` and `RenderManager`. Prefer small frame counts during development.

## Coding Style & Naming Conventions

Use Python type hints for public functions and structured dictionaries, following the existing `TypedDict` and `torch.Tensor | None` style. Keep imports grouped as standard library, third-party, then local modules. Use 4-space indentation, `snake_case` for functions and variables, and `PascalCase` for classes. Existing comments and docstrings may be Chinese; keep new comments short and only where they clarify non-obvious rendering or coordinate-system logic.

## Testing Guidelines

There is no formal pytest suite yet. Treat `test_render.py` as the smoke/performance test. When changing checkpoint loading or rendering paths, run `--frames 1` or `--frames 10` and verify output generation unless using `--no-save`. For script changes, run the target script with a small local sample.

## Commit & Pull Request Guidelines

Git history uses short conventional prefixes such as `feat(...)`, `debug(...)`, `perf(...)`, and `chore:`; continue that pattern and include the affected module when useful, for example `perf(render_manager): reduce lidar buffer copies`. Pull requests should describe the rendering path changed, list the checkpoint/data used for validation, include benchmark numbers when performance is affected, and attach screenshots or output samples for visual changes. Never include large `.pth`, `.ply`, JSON sensor dumps, or generated `output/` files in PRs.
