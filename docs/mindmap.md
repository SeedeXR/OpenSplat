# System Mindmap

High-level map of OpenSplat: modules, dependencies, data flow, and relationships. Detail lives
in [`architecture.md`](architecture.md); file locations in [`repo_organization.md`](repo_organization.md).

## Module mindmap

```mermaid
mindmap
  root((OpenSplat))
    app
      opensplat CLI
      simple_trainer
      visualizer optional
    io
      colmap
      nerfstudio
      opensfm
      openmvg
      point_io
      InputData
    model
      gaussian model
      spherical_harmonics
      optim_scheduler
      kdtree_tensor
    render
      project_gaussians
      rasterize_gaussians
      ssim loss
      gsplat.hpp switch
    rasterizer
      gsplat CUDA/HIP
      gsplat-cpu
      gsplat-metal
    common
      cv_utils
      tensor_math
      utils
      constants
```

## How the pieces connect

```mermaid
flowchart LR
    IMG["posed images + sparse points"] --> IO["src/io loaders"]
    IO --> ID["InputData"]
    ID --> APP["src/app driver"]
    APP --> MODEL["src/model (Gaussians)"]
    MODEL --> RENDER["src/render"]
    RENDER --> BE["rasterizer/ backend"]
    BE --> OUT[".ply / .splat + cameras.json"]
    COMMON["src/common (shared)"] -.-> IO & MODEL & RENDER & APP
```

## External dependencies

```mermaid
flowchart TD
    OS[OpenSplat] --> T[LibTorch<br/>tensors · autograd · optimizers]
    OS --> CV[OpenCV<br/>image IO]
    OS --> J[nlohmann_json]
    OS --> X[cxxopts<br/>CLI args]
    OS --> N[nanoflann<br/>KD-tree]
    OS --> G[glm<br/>CUDA/HIP only]
```

`nlohmann_json`, `nanoflann`, `cxxopts`, `glm` are fetched (or `find_package`d) in
`../CMakeLists.txt`. LibTorch + OpenCV are required external installs.

## Training data flow (summary)

`images + sparse points → InputData → initialize Gaussians → {project → rasterize → SSIM/L1 →
backprop → optimizer step → densify/prune} ×N → write splat`. Full sequence in
[`architecture.md`](architecture.md#4-training-data-flow).
