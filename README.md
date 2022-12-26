# TeukEvolution.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JLRipley314.github.io/TeukEvolution.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JLRipley314.github.io/TeukEvolution.jl/dev)
[![Build Status](https://github.com/JLRipley314/TeukEvolution.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JLRipley314/TeukEvolution.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JLRipley314/TeukEvolution.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JLRipley314/TeukEvolution.jl)

**Buyer Beware: this code is a work in progress**

`TeukEvolution.jl` contains routines to evolve the Teukolsky equation in
a horizon penetrating, hyperboloidally compactified system of coordinates.

## Initial data

There are two choices of initial data.

* Gaussian pulse initial data.

* Quasinormal mode initial data. These need to be read in from HDF5 files that
are generated with 
[TeukolskyQNMFunctions.jl](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl). 

## Visualization

The output is currently saved as `.csv` files, which stored in columns at
a fixed time (R,Y,value). 
We use the 2d plotter in the [sci-vis](https://github.com/JLRipley314/sci-vis)
to visualize code output.

## Contact

ripley[at]illinois[dot]edu

## Citation 

If you use this code, the best thing to cite right now would be

```
	@article{Ripley:2020xby,
	    author = "Ripley, Justin L. and Loutrel, Nicholas and Giorgi, Elena and Pretorius, Frans",
	    title = "{Numerical computation of second order vacuum perturbations of Kerr black holes}",
	    eprint = "2010.00162",
	    archivePrefix = "arXiv",
	    primaryClass = "gr-qc",
	    doi = "10.1103/PhysRevD.103.104018",
	    journal = "Phys. Rev. D",
	    volume = "103",
	    pages = "104018",
	    year = "2021"
	}
```
