# TeukEvolution.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JLRipley314.github.io/TeukEvolution.jl/dev)
[![Build Status](https://github.com/JLRipley314/TeukEvolution.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JLRipley314/TeukEvolution.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JLRipley314/TeukEvolution.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JLRipley314/TeukEvolution.jl)

**Buyer Beware: this code is a work in progress**

`TeukEvolution.jl` contains routines to evolve the Teukolsky equation in
a horizon penetrating, hyperboloidally compactified system of coordinates.
One day, this code may replace the second order Teukolsky code
[teuk-fortran-2020](https://github.com/JLRipley314/teuk-fortran-2020).
At the moment, this code can only do **linear** evolution of fields
right now. If you want to compute the second-order perturbation of
a black hole, you will have to use the teuk-fortran-2020 code.

## Initial data

There are two choices of initial data.

1. Gaussian pulse initial data.

2. Quasinormal mode initial data. These need to be read in from HDF5 files that
are generated with 
[TeukolskyQNMFunctions.jl](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl). 

See the `examples` directory for two example paramter files.

## Visualization

The output is currently saved as `.csv` files, which stored in columns at
a fixed time (R,Y,value). 
We use the 2d plotter in the [sci-vis](https://github.com/JLRipley314/sci-vis)
to visualize code output.

For example, once you've cloned the sci-vis repository, you can open
the 2D plotter via
```
python3 ~/sci-vis/plotters/plotter_2d.py
```
This should open a gui, from which you can open a field file, e.g.
`lin_f_re_2.csv`, which is the m=2 component of the real part of the
linear field (e.g. $\Psi_4$).

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

If you make use of quasinormal mode intial data, please also cite
```
@article{Ripley:2022ypi,
    author = "Ripley, Justin L.",
    title = "{Computing the quasinormal modes and eigenfunctions for the Teukolsky equation using horizon penetrating, hyperboloidally compactified coordinates}",
    eprint = "2202.03837",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1088/1361-6382/ac776d",
    journal = "Class. Quant. Grav.",
    volume = "39",
    number = "14",
    pages = "145009",
    year = "2022"
}
```

