#__precompile__()

module SpinLattice
using DiffEqBase, OrdinaryDiffEq, RecursiveArrayTools, DiffEqCallbacks
using JLD, HDF5, DataStructures
using RecipesBase
using SparseArrays, SharedArrays, Printf
using LinearAlgebra, Random, Distributed

include("compatibility.jl")
include("sharedsparse.jl")
include("spin_funcs.jl")
include("Interactions.jl")
include("Lattice.jl")
include("Approximations.jl")
include("SPOperators_parallel.jl")
include("RHS.jl")
include("Observables.jl")
include("InitialConditions.jl")
include("Models.jl")
include("Propagation.jl")
include("plotting.jl")
include("hdisorder.jl")


export Lattice, SpinArray, Interaction, ConvRules, CFVals, MDI, nearest_neighbours
export Exact, Clustered, PureClassical, Hybrid, AbstractModel
export ExactApprox, ClusteredApprox, PureClassicalApprox, HybridApprox
export ExactObservable, ClusteredObservable, PureClassicalObservable, HybridObservable, ObservablesSet
export build_Hamiltonian, build_Approximation, build_RHS_function, build_Spin_Operator, set_CorrFuncs, calculateCorrFunc!, build_Observable
export get_string, translate_indices, get_q_spins, get_cl_spins, get_all_spins, get_central_spins, get_spins_by_name
export build_problem, build_problem0, simulate, set_Logging, output, randomState, LProblem, similar
export get_positions, get_Dh, QuantumObservable, ClassicalObservable, axis_dict

end
