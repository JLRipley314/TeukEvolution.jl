# TeukEvolution.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://JLRipley314.github.io/TeukEvolution.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://JLRipley314.github.io/TeukEvolution.jl/dev)
[![Build Status](https://github.com/JLRipley314/TeukEvolution.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JLRipley314/TeukEvolution.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JLRipley314/TeukEvolution.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JLRipley314/TeukEvolution.jl)


`TeukEvolution.jl` contains routines to evolve the Teukolsky equation in
a horizon penetrating, hyperboloidally compactified system of coordinates.

## Initial data

There are choices of initial data.

* Gaussian pulse initial data.

* Quasinormal mode initial data. This is generated with 
[TeukolskyQNMFunctions.jl](https://github.com/JLRipley314/TeukolskyQNMFunctions.jl). 

## Visualization

The output is currently saved as `.csv` files, which stored in columns at
a fixed time (R,Y,value). 
I have used [paraview](https://www.paraview.org/) visualize the output files.
