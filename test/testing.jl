module myTesting
using SpinLattice
using PyCall, RecursiveArrayTools
unshift!(PyVector(pyimport("sys")["path"]), "")
@pyimport mkl
mkl.set_num_threads(1)
@pyimport madness as oop
@pyimport scipy.sparse as pysparse
@pyimport scipy.sparse.linalg as slinalg

L1d_1 = ((15,), (-0.41,-0.41,0.82), Interaction(nearest_neighbours), (0.,0.,0.))
L1d_2 = ((15,), (1.0,1.0,1.0), Interaction(nearest_neighbours), (0.,0.,0.))
L1d_3 = ((15,), (0.707,0.707,0.0), Interaction(nearest_neighbours), (0.,0.,0.))
L1d_4 = ((15,), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.,0.,0.))
L1d_5 = ((15,), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.207,0.,0.))
L1d_6 = ((15,), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.207,0.,0.100))
L1d_7 = ((15,), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.207,0.,0.207))

L2d_1 = ((9,9), (-0.41,-0.41,0.82), Interaction(nearest_neighbours), (0.,0.,0.))
L2d_2 = ((9,9), (1.0,1.0,1.0), Interaction(nearest_neighbours), (0.,0.,0.))
L2d_3 = ((9,9), (0.707,0.707,0.0), Interaction(nearest_neighbours), (0.,0.,0.))
L2d_4 = ((9,9), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.,0.,0.))
L2d_5 = ((9,9), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.207,0.,0.))
L2d_6 = ((9,9), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.207,0.,0.100))
L2d_7 = ((9,9), (0.518, 0.830, 0.207), Interaction(nearest_neighbours), (0.207,0.,0.207))

L3d_1 = ((9,9,9), (-0.41,-0.41,0.82), Interaction(nearest_neighbours), (0.,0.,0.))
Lfid0x0x1 = ((9,9,9), (-0.5,-0.5,1.0), Interaction(MDI(0,0,1)), (0.,0.,0.), [(1,1,i) for i in 1:9], [(1,1,i) for i in 1:9])
Lfid0x1x1 = ((9,9,9), (-0.5,-0.5,1.0), Interaction(MDI(0,1,1)), (0.,0.,0.), [(i,1,1) for i in 1:9], [(i,1,1) for i in 1:9])
Lfid1x1x1 = ((9,9,9), (-0.5,-0.5,1.0), Interaction(MDI(1,1,1)), (0.,0.,0.), [(i,i,i) for i in 1:9], [(i,i,i) for i in 1:9])

L1d = [L1d_1, L1d_2, L1d_3, L1d_4, L1d_5, L1d_6, L1d_7]
L2d = [L2d_1, L2d_2, L2d_3, L2d_4, L2d_5, L2d_6, L2d_7]
L3d = [L3d_1, Lfid0x0x1, Lfid0x1x1, Lfid1x1x1]

d1block = (5,)
d2block = (3,3)
d3block = (:cross, sort([(2,2,2),(2,2,1),(2,1,2),(1,2,2),(2,2,3),(2,3,2),(3,2,2)]), [(2,2,2)])
d3block_py = ([Tuple(reverse(collect(ind) .- 1)) for ind in d3block[2]], [Tuple(reverse(collect(ind) .- 1)) for ind in d3block[3]])
get_block(::Type{Val{1}}) = d1block
get_block(::Type{Val{2}}) = d2block
get_block(::Type{Val{3}}) = d3block
get_block_py(x::Type{Val{D}}) where D = get_block(x)
get_block_py(::Type{Val{3}}) = d3block_py

jlmat2pymat(S::SparseMatrixCSC) = pysparse.csc_matrix((S.nzval, S.rowval .- 1, S.colptr .- 1), shape=size(S))
function py_initialize(M::Exact, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64}; kwargs...) where {D}
    sp = oop.UniformLattice(reverse(dims), hs, Js; kwargs...)
    return sp
end
function py_initialize(M::Exact, dims::NTuple{2,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64}; kwargs...)
    sp = oop.UniformLattice(reverse((4,4)), hs, Js; kwargs...)
    return sp
end
function py_initialize(M::Clustered, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64}; kwargs...) where {D}
    dims_r = reverse(dims)
    block_dims = reverse(get_block_py(Val{D}))
    blocks = Tuple(div.(collect(dims_r), collect(block_dims)))
    sp = oop.ClusteredLattice(block_dims, blocks, hs, Js; kwargs...)
    return sp
end
function py_initialize(M::PureClassical, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64}; kwargs...) where {D}
    sp = oop.ClassicalLattice(reverse(dims), hs, Js; kwargs...)
    return sp
end
function py_initialize(M::Hybrid, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64}; kwargs...) where {D}
    dims_r = reverse(dims)
    block_dims = reverse(get_block_py(Val{D}))
    sp = oop.HybridLattice(block_dims, dims_r, hs, Js; kwargs...)
    return sp
end
function py_initialize(M::Hybrid, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64}; kwargs...)
    q_spins, middle_spins = d3block_py
    q_spins_r = [reverse(x) for x in q_spins]
    sp = oop.HybridLatticeIR(q_spins_r, reverse(dims), middle_spins, hs, Js; kwargs...)
    return sp
end
function py_initialize(M::Hybrid, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{MDI{3}}, hs::NTuple{3,Float64}; kwargs...)
    q_spins, middle_spins = d3block_py
    q_spins_r = [reverse(x) for x in q_spins]
    sp = oop.FIDLattice(3, reverse(dims), q_spins_r, middle_spins, direction = reverse(I.func.dir); kwargs...)
    return sp
end
function py_initialize(M::Hybrid, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{MDI{3}}, hs::NTuple{3,Float64}, q_spins::Vector{NTuple{3,Int64}}, middle_spins::Vector{NTuple{3,Int64}}; kwargs...)
    q_spins_r = [Tuple(reverse(collect(ind) .- 1)) for ind in q_spins]
    middle_spins_r = [Tuple(reverse(collect(ind) .- 1)) for ind in middle_spins]
    sp = oop.FIDLattice(3, reverse(dims), q_spins_r, middle_spins_r, direction = reverse(I.func.dir); kwargs...)
    return sp
end
function py_initialize(M::PureClassical, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{MDI{3}}, hs::NTuple{3,Float64}; kwargs...)
    sp = oop.FIDLatticeClassical(3, reverse(dims), direction = reverse(I.func.dir); kwargs...)
    return sp
end
@inline function py_initialize(M::PureClassical, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}, q_spins::Vector{NTuple{3,Int64}}, middle_spins::Vector{NTuple{3,Int64}}; kwargs...) where {D,F}
    py_initialize(M::PureClassical, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}; kwargs...)
end

@inline default_links(L::Lattice, M::Model) where Model<:Union{Exact, PureClassical} = (:all,:all)
@inline default_links(L::Lattice{D}, M::Model) where {D, Model<:Union{Clustered,Hybrid}} = (:all,:central)
@inline default_links(L::Lattice{3}, M::Hybrid) = (:all, SpinArray(L, [(2,2,2)], :central))

function julia_initialize(M::Model, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where {D, Model<:Union{Exact,PureClassical}} where F
    L = Lattice(dims, Js, I, hs)
    A = build_Approximation(M, L)
    RHS = build_RHS_function(A)
    rules, OSET = set_CorrFuncs(A, default_links(L,M))
    return A, RHS, OSET, rules
end
@inline function julia_initialize(M::PureClassical, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}, q_spins::Vector{NTuple{3,Int64}}, middle_spins::Vector{NTuple{3,Int64}}) where {D, F}
    julia_initialize(M, dims, Js, I, hs)
end
function julia_initialize(M::Exact, dims::NTuple{2,Int64}, Js::NTuple{3,Float64}, I::Interaction{typeof(nearest_neighbours)}, hs::NTuple{3,Float64})
    L = Lattice((4,4), Js, I, hs)
    A = build_Approximation(M, L)
    RHS = build_RHS_function(A)
    rules, OSET = set_CorrFuncs(A, default_links(L,M))
    return A, RHS, OSET, rules
end

function julia_initialize(M::Model, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where {D, Model<:Union{Clustered,Hybrid}, F}
    L = Lattice(dims, Js, I, hs)
    A = build_Approximation(M, L, get_block(Val{D}))
    RHS = build_RHS_function(A)
    rules, OSET = set_CorrFuncs(A, default_links(L,M))
    return A, RHS, OSET, rules
end

function julia_initialize(M::Hybrid, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where F
    L = Lattice(dims, Js, I, hs)
    name, q_spins, middle_spins = d3block
    sarray = SpinArray(L, q_spins, name)
    A = build_Approximation(M, L, sarray)
    RHS = build_RHS_function(A)
    rules, OSET = set_CorrFuncs(A, default_links(L,M))
    return A, RHS, OSET, rules
end
function julia_initialize(M::Hybrid, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}, q_spins::Vector{NTuple{3,Int64}}, middle_spins::Vector{NTuple{3,Int64}}) where F
    L = Lattice(dims, Js, I, hs)
    name = :column
    sarray = SpinArray(L, q_spins, name)
    A = build_Approximation(M, L, sarray)
    RHS = build_RHS_function(A)
    rules, OSET = set_CorrFuncs(A, (:all, SpinArray(L, middle_spins, :central)))
    return A, RHS, OSET, rules
end

function test_julia_init()
    Models = [Exact(), PureClassical(), Clustered(), Hybrid()]
    for model in Models
        for lat in L1d
            try
                julia_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
    for model in Models
        for lat in L2d
            try
                julia_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
    for model in Models[[2,4]]
        for lat in L3d
            try
                julia_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
end
function test_py_init()
    Models = [Exact(), PureClassical(), Clustered(), Hybrid()]
    for model in Models
        for lat in L1d
            try
                py_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
    for model in Models
        for lat in L2d
            try
                py_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
    for model in Models[[2,4]]
        for lat in L3d
            try
                py_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
end
function test_py_init_3d()
    Models = [Exact(), PureClassical(), Clustered(), Hybrid()]
    for model in Models[[2,4]]
        for lat in L3d
            try
                py_initialize(model, lat...)
                print("all green\n\n")
            catch
                print("$model   $lat\n\n")
            end
        end
    end
end

const toler = 1e-14

function convert_state(A::ExactApprox, sp)
    v = Vector{Vector{Complex{Float64}}}()
    push!(v, copy(sp[:psi]))
    for i = 1:size(sp[:psi_aid],1)
        push!(v, sp[:psi_aid][i,:])
    end
    return VectorOfArray(v)
end

function test_operators(M::Exact, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where {D, F}
    A,RHS,OSET, rules = julia_initialize(M, dims, Js, I, hs)
    sp = py_initialize(M, dims, Js, I, hs)
    H_j = jlmat2pymat(RHS.H)
    Ms_j = Dict()
    for Obs in OSET.Observables
        axis = axis_dict[Obs.axis]-1
        Ms_j[axis] = jlmat2pymat(Obs.QO.O)
    end
    sp[:initState]()
    u0 = copy(sp[:psi])
    du = copy(u0)
    tj = -time()
    RHS(0.0,u0,du)
    tj += time()
    tp = -time()
    du2 = sp[:rhs](sp[:psi])
    tp += time()
    print("$((dims, Js, I, hs))\n")
    @printf("Check Hamiltonian:   %e\n", slinalg.norm(sp[:H]-H_j)/slinalg.norm(sp[:H]))
    @printf("Check tot magnetiz: %e   %e   %e\n", [slinalg.norm(sp[:Ms][i]-Ms_j[i])/slinalg.norm(sp[:Ms][i]) for i in 0:2]...)
    @printf("Check RHS: %e\n", norm(du.*(2.0^-7)-du2)/norm(du2))
    @printf("Check time: %10e", tj/tp)
    print("\n\n")

end

function convert_state(A::ClusteredApprox, sp)
    cluster_num = size(sp[:psi],1)
    u0 = VectorOfArray([copy(sp[:psi][i,:]) for i in 1:cluster_num])
    return u0
end

function test_operators(M::Clustered, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where {D, F}
    d = Dict([(:all,"Global"), (:central,"Local")])
    A,RHS,OSET, rules = julia_initialize(M, dims, Js, I, hs)
    sp = py_initialize(M, dims, Js, I, hs)
    H_j = jlmat2pymat(RHS.H)
    ops_j = Dict()
    for Obs in OSET.Observables
        index = (string(Obs.axis), d[Obs.name])
        ops_j[index] = jlmat2pymat(Obs.QO.O)
    end
    num_of_espins = RHS.num_of_espins
    taus_j = Matrix(num_of_espins, 3)
    for espin in 1:num_of_espins for sigma in 1:3
        taus_j[espin, sigma] = jlmat2pymat(RHS.edgeSpinOperators[sigma, espin])
    end end
    sp[:initState]()
    cluster_num = size(sp[:psi],1)
    u0 = VectorOfArray([copy(sp[:psi][i,:]) for i in 1:cluster_num])
    du = VectorOfArray([copy(sp[:psi][i,:]) for i in 1:cluster_num])
    tj = -time()
    RHS(0.0,u0,du)
    tj += time()
    tp = -time()
    du2 = sp[:rhs](sp[:psi])
    tp += time()

    print("$((dims, Js, I, hs))\n")
    @printf("Check Hamiltonian:   %e\n", slinalg.norm(sp[:H]-H_j)/slinalg.norm(sp[:H]))
    print("Check spin operators:\n")
    for espin in 1:num_of_espins
        for sigma in 1:3
            @printf("%10e ", slinalg.norm(sp[:taus][espin, sigma]-taus_j[espin, sigma])/slinalg.norm(sp[:taus][espin, sigma]))
        end
        print("\n")
    end
    ###################
    links = sp[:links]
    noe = num_of_espins
    cl_num = cluster_num
    len = noe*cl_num
    IMp = zeros(len, len)
    for link in links
        p1 = sub2ind((noe,cl_num), link[1][2]+1, link[1][1]+1)
        p2 = sub2ind((noe,cl_num), link[2][2]+1, link[2][1]+1)
        IMp[p1,p2] += 1.0
    end
    IMp .= .-IMp.*0.25.*sqrt(sp[:N])
    @printf("Check intrablock matrix: %10e\n", norm(IMp-RHS.IM)/norm(IMp))
    ####################
    print("Check observables:\n")
    print("Local: ")
    for axis in ["x","y","x"]
        index = (axis,"Local")
        @printf("%10e ", slinalg.norm(sp[:ops][index] - ops_j[index])/slinalg.norm(sp[:ops][index]))
    end
    print("\n")
    print("Global: ")
    for axis in ["x","y","x"]
        index = (axis,"Global")
        @printf("%10e ", slinalg.norm(sp[:ops][index] - ops_j[index])/slinalg.norm(sp[:ops][index]))
    end
    print("\n")
    ###################
    print("Check rhs: ")
    for i in 1:cluster_num
        @printf("%10e ", norm(du[:,i].*2.0^-7-du2[i,:])/norm(du2[i,:]))
    end
    print("\n")
    @printf("Check time: %10e", tj/tp)
    print("\n\n")
end

function convert_state(A::HybridApprox, sp)
    u0 = ArrayPartition(copy(sp[:psi]), VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3]))
    return u0
end

function test_operators(M::Hybrid, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where {D, F}
    d = Dict([(:all,"Global"), (:central,"Local")])
    A,RHS,OSET, rules = julia_initialize(M, dims, Js, I, hs)
    sp = py_initialize(M, dims, Js, I, hs)
    H_j = jlmat2pymat(RHS.H)
    ops_j = Dict()
    for Obs in OSET.Observables
        index = (axis_dict[Obs.axis]-1, d[Obs.name])
        ops_j[index] = (jlmat2pymat(Obs.QO.O),Obs)
    end
    num_of_espins = size(RHS.edgeSpinOperators,1)
    taus_j = Matrix(num_of_espins, 3)
    for espin in 1:num_of_espins for sigma in 1:3
        taus_j[espin, sigma] = jlmat2pymat(RHS.edgeSpinOperators[espin, sigma])
    end end
    sp[:initState]()
    sp[:updateCorrelator]()
    u0 = ArrayPartition(copy(sp[:psi]), VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3]))
    du = ArrayPartition(copy(sp[:psi]), VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3]))
    tj = -time()
    RHS(0.0,u0,du)
    tj += time()
    tp = -time()
    du2 = sp[:rhs](sp[:psi], sp[:clState])
    tp += time()

    print("$((dims, Js, I, hs))\n")
    @printf("Check Hamiltonian:   %e\n", slinalg.norm(sp[:H]-H_j)/slinalg.norm(sp[:H]))
    print("Check spin operators:\n")
    for espin in 1:num_of_espins
        for sigma in 1:3
            @printf("%10e ", slinalg.norm(sp[:taus][espin, sigma]-taus_j[espin, sigma])/slinalg.norm(sp[:taus][espin, sigma]))
        end
        print("\n")
    end
    ###################
    print("Check observable ops:\n")
    print("Local: ")
    for axis in [0,1,2]
        index = (axis,"Local")
        @printf("%10e ", slinalg.norm(sp[:ops][index] - ops_j[index][1])/slinalg.norm(sp[:ops][index]))
    end
    print("\n")
    print("Global: ")
    for axis in [0,1,2]
        index = (axis,"Global")
        @printf("%10e ", slinalg.norm(sp[:ops][index] - ops_j[index][1])/slinalg.norm(sp[:ops][index]))
    end
    print("\n")
    ###################
    print("Check observable vals:\n")
    print("Local: ")
    for axis in [0,1,2]
        index = (axis,"Local")
        @printf("%10e ", norm(sp[:means][index][end] - ops_j[index][2](u0))/norm(sp[:means][index][end]))
    end
    print("\n")
    print("Global: ")
    for axis in [0,1,2]
        index = (axis,"Global")
        @printf("%10e ", norm(sp[:means][index][end] - ops_j[index][2](u0))/norm(sp[:means][index][end]))
    end
    print("\n")
    ###################
    IM_j = jlmat2pymat(RHS.IM)
    q2cl_j = jlmat2pymat(RHS.q2cl)
    cl2q_j = jlmat2pymat(RHS.cl2q)
    q2cl_p = sp[:Q2CL]*0.5*sqrt(get_Dh(A))
    cl2q_p = sp[:CL2Q]*-0.5
    @printf("Check IM: %10e\n", slinalg.norm(sp[:FM]-IM_j)/slinalg.norm(sp[:FM]))
    @printf("Check Q2CL: %10e\n", slinalg.norm(q2cl_p-q2cl_j)/slinalg.norm(q2cl_p))
    @printf("Check CL2Q: %10e\n", slinalg.norm(cl2q_p-cl2q_j)/slinalg.norm(cl2q_p))
    ###################
    print("Check rhs:\n")
    @printf("%10e\n", norm(du2[1] - du.x[1].*2.0^-7)/norm(du2[1]))
    for sigma in 1:3
        @printf("%10e ", norm(du2[2][sigma,:]-du.x[2][:,sigma].*2.0^-7)/norm(du2[2][sigma,:]))
    end
    print("\n")
    @printf("Check time: %10e", tj/tp)
    print("\n\n")
end

function test_operators(M::Hybrid, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}, q_spins::Vector{NTuple{3,Int64}}, middle_spins::Vector{NTuple{3,Int64}}) where {F}
    d = Dict([(:all,"Global"), (:central,"Local")])
    A,RHS,OSET, rules = julia_initialize(M, dims, Js, I, hs, q_spins, middle_spins)
    sp = py_initialize(M, dims, Js, I, hs, q_spins, middle_spins)
    H_j = jlmat2pymat(RHS.H)
    ops_j = Dict()
    for Obs in OSET.Observables
        if Obs.axis == :x
            index = d[Obs.name]
            ops_j[index] = (jlmat2pymat(Obs.QO.O),Obs)
        end
    end
    num_of_espins = size(RHS.edgeSpinOperators,1)
    taus_j = Matrix(num_of_espins, 3)
    for espin in 1:num_of_espins for sigma in 1:3
        taus_j[espin, sigma] = jlmat2pymat(RHS.edgeSpinOperators[espin, sigma])
    end end
    sp[:initState]()
    sp[:updateCorrelator]()
    u0 = ArrayPartition(copy(sp[:psi]), VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3]))
    du = ArrayPartition(copy(sp[:psi]), VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3]))
    tj = -time()
    RHS(0.0,u0,du)
    tj += time()
    tp = -time()
    du2 = sp[:rhs](sp[:psi], sp[:clState])
    tp += time()

    print("$((dims, Js, I, hs))\n")
    @printf("Check Hamiltonian:   %e\n", slinalg.norm(sp[:H]-H_j)/slinalg.norm(sp[:H]))
    print("Check spin operators:\n")
    for espin in 1:num_of_espins
        for sigma in 1:3
            @printf("%10e ", slinalg.norm(sp[:taus][espin, sigma]-taus_j[espin, sigma])/slinalg.norm(sp[:taus][espin, sigma]))
        end
        print("\n")
    end
    ###################
    print("Check observable ops:\n")
    index = "Local"
    @printf("Local: %10e\n", slinalg.norm(sp[:ops][index] - ops_j[index][1])/slinalg.norm(sp[:ops][index]))
    index = "Global"
    @printf("Global: %10e\n", slinalg.norm(sp[:ops][index] - ops_j[index][1])/slinalg.norm(sp[:ops][index]))
    ###################
    print("Check observable vals:\n")
    index = "Local"
    @printf("Local: %10e\n", norm(sp[:means][index][end] - ops_j[index][2](u0))/norm(sp[:means][index][end]))
    index = "Global"
    @printf("Global: %10e\n", norm(sp[:means][index][end] - ops_j[index][2](u0))/norm(sp[:means][index][end]))
    ###################
    IM_j = RHS.IM
    q2cl_j = RHS.q2cl
    cl2q_j = RHS.cl2q
    q2cl_p = sp[:Q2CL]
    cl2q_p = -sp[:CL2Q]
    @printf("Check IM: %10e\n", norm(sp[:FM]-IM_j)/norm(sp[:FM]))
    @printf("Check Q2CL: %10e\n", norm(q2cl_p-q2cl_j)/norm(q2cl_p))
    @printf("Check CL2Q: %10e\n", norm(cl2q_p-cl2q_j)/norm(cl2q_p))
    ###################
    print("Check rhs:\n")
    @printf("%10e\n", norm(du2[1] - du.x[1].*2.0^-7)/norm(du2[1]))
    for sigma in 1:3
        @printf("%10e ", norm(du2[2][sigma,:]-du.x[2][:,sigma].*2.0^-7)/norm(du2[2][sigma,:]))
    end
    print("\n")
    @printf("Check time: %10e", tj/tp)
    print("\n\n")
end

function convert_state(A::PureClassicalApprox, sp)
    u0 = VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3])
    return u0
end

function test_operators(M::PureClassical, dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}) where {D, F}
    A,RHS,OSET, rules = julia_initialize(M, dims, Js, I, hs)
    sp = py_initialize(M, dims, Js, I, hs)
    ops_j = Dict()
    for Obs in OSET.Observables
        index = axis_dict[Obs.axis]-1
        ops_j[index] = Obs
    end
    sp[:initState]()
    sp[:updateCorrelator]()
    u0 = VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3])
    du = VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3])
    tj = -time()
    RHS(0.0,u0,du)
    tj += time()
    tp = -time()
    du2 = sp[:rhs](sp[:clState])
    tp += time()

    print("$((dims, Js, I, hs))\n")
    ###################
    print("Check observable vals: ")
    for axis in [0,1,2]
        @printf("%10e ", norm(sp[:means][axis][end] - ops_j[axis](u0))/norm(sp[:means][axis][end]))
    end
    print("\n")
    ###################
    IM_j = jlmat2pymat(RHS.IM)
    @printf("Check IM: %10e\n", slinalg.norm(sp[:FM]-IM_j)/slinalg.norm(sp[:FM]))
    ###################
    print("Check rhs: ")
    for sigma in 1:3
        @printf("%10e ", norm(du2[sigma,:]-du[:,sigma].*2.0^-7)/norm(du2[sigma,:]))
    end
    print("\n")
    @printf("Check time: %10e", tj/tp)
    print("\n\n")
end

function test_operators(M::PureClassical, dims::NTuple{3,Int64}, Js::NTuple{3,Float64}, I::Interaction{F}, hs::NTuple{3,Float64}, q_spins, middle_spins) where {F}
    A,RHS,OSET, rules = julia_initialize(M, dims, Js, I, hs)
    sp = py_initialize(M, dims, Js, I, hs)
    ops_j = OSET.Observables[1]
    for Obs in OSET.Observables
        if Obs.axis == :x
            ops_j = Obs
        end
    end
    sp[:initState]()
    sp[:updateCorrelator]()
    u0 = VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3])
    du = VectorOfArray([copy(sp[:clState][i,:]) for i in 1:3])
    tj = -time()
    RHS(0.0,u0,du)
    tj += time()
    tp = -time()
    du2 = sp[:rhs](sp[:clState])
    tp += time()

    print("$((dims, Js, I, hs))\n")
    ###################
    @printf("Check observable val: %10e\n", norm(sp[:means][end] - ops_j(u0))/norm(sp[:means][end]))
    ###################
    IM_j = RHS.IM
    @printf("Check IM: %10e\n", norm(sp[:FM]-IM_j)/norm(sp[:FM]))
    ###################
    print("Check rhs: ")
    for sigma in 1:3
        @printf("%10e ", norm(du2[sigma,:]-du[:,sigma].*2.0^-7)/norm(du2[sigma,:]))
    end
    print("\n")
    @printf("Check time: %10e", tj/tp)
    print("\n\n")
end

@inline store_state(A::ExactApprox, sp) = (deepcopy(sp[:psi]),deepcopy(sp[:psi_aid]))
@inline store_state(A::ClusteredApprox, sp) = (deepcopy(sp[:psi]),)
@inline store_state(A::HybridApprox, sp) = (deepcopy(sp[:psi]), deepcopy(sp[:clState]))
@inline store_state(A::PureClassicalApprox, sp) = (deepcopy(sp[:clState]),)

function test_propagation(M::AbstractModel, args...; Tmax::Float64 = 10.0, tstep::Float64 = 2.0^-7, delimiter = 10)
    A, ARHS, OSET, rules = julia_initialize(M, args...)
    sp = py_initialize(M, args...; :Tmax => Tmax, :tstep => tstep)
    sp[:initState]()
    u0 = convert_state(A, sp)
    state = store_state(A, sp)
    sp[:propagate](state...)
    ####
    l = length(OSET.Observables)
    tspan = (0.0,Tmax*delimiter)
    t_means = collect(0.0:tstep:(Tmax*delimiter))
    t_cors = collect(0.0:tstep:Tmax)
    saved_values = SpinLattice.SavedValues(t_means, [zeros(l) for i in eachindex(t_means)])
    prob = SpinLattice.ODEProblem(ARHS, u0, tspan)
    cb = SpinLattice.SavingCallback(OSET, saved_values, u0; saveat = t_means)
    #cb = SavingCallback(OSET, saved_values; save_everystep=true)
    cf_vals = CFVals(rules, t_cors)
    LP = LProblem(A, cb, prob, rules, cf_vals, Tmax, tstep, delimiter)
    ####
    cf_vals, integrator = simulate(LP, 1, SpinLattice.RK4(); adaptive = false, dt = LP.tstep)

    return LP, cf_vals, integrator, sp
end


end
