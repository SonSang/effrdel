# Efficient Restricted Delaunay Triangulation for Mesh Recovery

This repository provides an implementation of **Restricted Delaunay Triangulation (RDT)** for reconstructing a connected, self-intersection-free mesh from a potentially defective *triangle soup*.  
Given an input mesh with inconsistent connectivity or artifacts, we:

1. Build the **3D Delaunay triangulation** of the input vertices.
2. Select a **restricted subset of Delaunay faces** that captures the surface topology of the input.
3. Output a clean mesh.

The implementation is heavily optimized with a **BVH** and **CPU/GPU vectorization**.

## Installation

This library depends on [PyTorch](https://pytorch.org/). Install PyTorch first (matching your CUDA setup), then install this package:

```bash
pip install -e .
```

## Usage

Specify the input and output mesh paths and run:

```python
import rdel

rdel.run(
    INPUT_MESH_PATH,
    OUTPUT_MESH_PATH,
    verbose=False,   # Print timings and extra logs if True
    orient=False     # Try to consistently orient face normals if True
)
```

* `INPUT_MESH_PATH`: path to a triangle soup mesh (e.g., `.ply`, `.obj`).
* `OUTPUT_MESH_PATH`: path to save the cleaned mesh.
* `verbose`: enables timing and diagnostic prints.
* `orient`: attempts to make face normals as consistent as possible.