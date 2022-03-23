# Exploring Julia
This presentation explores the GPU programming features of Julia

## Interacting with Julia
Load the Julia module with "module load julia-1.6.3"

To run a program from the command line, use "julia foo.jl Arg1 Arg2 ..."

To enter interactive mode, use "julia"

To exit interactive mode, press Ctrl+D

To install packages, press "]" in interactive mode and use "add ..."

To exit package mode, press Ctrl+C

## cudaLU.jl
Implementation of the pivoted LU with register shuffling. Run interactively
with "ARGS=[N_total M]; include("cudaLU.jl")".

## axpby.jl
Performs and times the axpby operation for random N-length vectors using
CUDA.jl's kernel and vector programming features. Run from the command line with
"julia axpby.jl N" or interactively with "ARGS=["N"]; include("cudaLU.jl")"
