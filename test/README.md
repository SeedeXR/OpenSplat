# test/

OpenSplat's automated test suite. **Off by default** — enable with `-DOPENSPLAT_BUILD_TESTS=ON`
so normal and CI builds are unaffected.

```
test/
├── CMakeLists.txt     # FetchContent(doctest); one target per suite, registered with ctest
├── unit/              # isolated logic — fast, no dataset needed
│   ├── test_utils.cpp            (InfiniteRandomIterator, parallel_for; header-only, no Torch)
│   ├── test_tensor_math.cpp      (quat/rodrigues/rotation matrices, pose scaling)
│   ├── test_ssim.cpp             (SSIM(x,x)=1, dissimilar<1, symmetry)
│   ├── test_optim_scheduler.cpp  (log-linear LR schedule endpoints/midpoint/monotonicity)
│   ├── test_kdtree.cpp           (PointsTensor::scales nearest-neighbour spacing)
│   └── test_cv_utils.cpp         (tensor <-> cv::Mat roundtrips)
├── regression/        # locked-in numeric behaviour (characterization) + per-bug tests
│   └── test_regression_math.cpp
├── integration/       # modules wired together (CPU backend)
│   └── test_pipeline.cpp         (PLY roundtrip; COLMAP load — DATA-GATED)
└── fixtures/          # tiny committed inputs (large data: scripts/fetch_test_data.sh)
```

## Build & run

```bash
scripts/build.sh --libtorch /path/to/libtorch --backend CPU -- -DOPENSPLAT_BUILD_TESTS=ON
ctest --test-dir build --output-on-failure         # all suites
ctest --test-dir build -R test_tensor_math         # a single suite
# data-gated integration: point at a scene (else it skips)
OPENSPLAT_TEST_DATA=data/db/drjohnson ctest --test-dir build -R test_integration --output-on-failure
```

## Levels (see ../memory/operating/test.md for the full methodology)

| Level | What it asserts | Data | Deps |
| ----- | --------------- | ---- | ---- |
| **unit** | A single function/class | none | Torch (utils: none) |
| **regression** | Stable numeric behaviour stays put | none | Torch |
| **integration** | Modules wired (IO roundtrip, load → InputData) | optional scene | full CPU pipeline |

## Conventions

- Framework: [doctest](https://github.com/doctest/doctest) (header-only, fast compile).
- Each test target links only the sources it exercises — keep dependencies minimal.
- Tests requiring a dataset must **skip gracefully** when `../data/` is absent.
- Performance/benchmark testing (the 2→64 image ladder) lives in `scripts/benchmark.sh`
  and writes to `../memory/profiles/`, separate from correctness tests here.

> Status: unit suites cover the leaf math/IO/utility modules (`tensor_math`, `ssim`,
> `optim_scheduler`, `kdtree_tensor`, `cv_utils`, `utils`), plus regression characterization
> and an integration suite (PLY roundtrip + data-gated COLMAP load). **Not yet compiled/run on
> the 16 GB dev machine** (needs LibTorch+OpenCV) — CI / first run is the authoritative check.
> Modules tied to the GPU backends (`model`, `project_gaussians`, `rasterize_gaussians`,
> `spherical_harmonics`) are exercised via the integration pipeline; dedicated unit tests for
> them are tracked in `../memory/operating/todo.md` (Phase 2).
