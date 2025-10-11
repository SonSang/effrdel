# Efficient Restricted Delaunay Triangulation for Mesh Recovery

This repo contains a code to recover connected mesh free from self-intersection using [Restricted Delaunay Triangulation](https://www.cs.purdue.edu/homes/tamaldey/book/Delmesh/chapter13-old.pdf).
For the given input triangle soup, which could be defective, it constructs Delaunay Triangulation of the vertices of the triangle soup.
Then, it finds the subset of faces in the Delaunay Triangulation that resembles the overall topology of the input mesh.
The algorithm is highly optimized using BVH and CPU/GPU vectorization.

## Installation

This code requires [PyTorch](https://pytorch.org/). After installing it, please install this library using

```
pip install -e .
```

## Usage

You can specify the path to the input mesh and output mesh to run the algorithm.

```
import rdel

rdel.run(
    INPUT_MESH_PATH,
    OUTPUT_MESH_PATH,
    verbose=False,          # whether or not to print timings
    orient=False            # whether or not to orient the face normals consistently as much as possible
)
```