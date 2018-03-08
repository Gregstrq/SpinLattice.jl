abstract type AbstractApproximation{Ltype} end
abstract type AbstractQuantumApproximation{Ltype} <: AbstractApproximation{Ltype} end

struct ExactApprox{Ltype} <: AbstractQuantumApproximation{Ltype}
    L::Ltype
    Dh::Int64
    ExactApprox(L::Ltype) where {Ltype<:Lattice} = new{Ltype}(L, 2^L.tot_spin_num)
end

struct ClusteredApprox{Ltype,D1,D2} <: AbstractQuantumApproximation{Ltype}
    L::Ltype
    inner_cluster_dims::NTuple{D1,Int64}
    outer_cluster_dims::NTuple{D1,Int64}
    transpose_dims::NTuple{D2, Int64}
    cluster_cell_num::Int64
    cluster_num::Int64
    function ClusteredApprox(L::Lattice{D,F,T,C}, inner_cluster_dims::NTuple{D,Int64}) where {D,F,T,C}
        for i in 1:D
            assert(L.lattice_dims[i]%inner_cluster_dims[i] == 0)
        end
        outer_cluster_dims = tuple([div(L.lattice_dims[i], inner_cluster_dims[i]) for i in 1:D]...)
        transpose_dims = Vector{Int64}()
        for i in 1:D
            push!(transpose_dims, inner_cluster_dims[i])
            push!(transpose_dims, outer_cluster_dims[i])
        end
        transpose_dims = tuple(transpose_dims...)
        cluster_cell_num = prod(inner_cluster_dims)
        cluster_num = prod(outer_cluster_dims)
        new{Lattice{D,F,T,C},D,length(transpose_dims)}(L, inner_cluster_dims, outer_cluster_dims, transpose_dims, cluster_cell_num, cluster_num)
    end
end

struct PureClassicalApprox{Ltype} <: AbstractApproximation{Ltype}
    L::Ltype
    PureClassicalApprox(L::Ltype) where {Ltype<:Lattice} = new{Ltype}(L)
end

struct HybridApprox{Ltype} <: AbstractQuantumApproximation{Ltype}
    L::Ltype
    q_spins::Vector{Tuple{Int64,Int64}}
    cl_spins::Vector{Tuple{Int64,Int64}}
    q_num::Int64
    cl_num::Int64
    name::Symbol
    function HybridApprox(L::Ltype, q_spins::Vector{Tuple{Int64,Int64}}, name::Symbol) where {Ltype<:Lattice}
        q_num = length(q_spins)
        for i in 1:q_num
            assert(1 <= q_spins[i][1] <= L.tot_cell_num)
            assert(1 <= q_spins[i][2] <= L.cell_size)
        end
        sort!(q_spins)
        cl_num = L.tot_spin_num - q_num
        cl_spins = Vector{Tuple{Int64,Int64}}()
        for s in 1:L.tot_cell_num for c in 1:L.cell_size
            if !((s,c) in q_spins)
                push!(cl_spins, (s,c))
            end
        end end
        sort!(cl_spins)
        new{Ltype}(L, q_spins, cl_spins, q_num, cl_num, name)
    end
end
function HybridApprox(L::Lattice, q_spins::Vector{Int64}, name=:undef)
    qq_spins = translate_indices(L, q_spins)
    return HybridApprox(L, qq_spins, name)
end
@inline HybridApprox(L::Lattice, q_spins::SpinArray) = HybridApprox(L, q_spins.spins, q_spins.name)
@inline function HybridApprox(L::Lattice{D}, q_block::NTuple{D, Int64}) where {D}
    name = :block__
    i = 1
    name = Symbol(name, q_block[i])
    i += 1
    while i<=length(q_block)
        name = Symbol(name, :_, q_block[i])
        i += 1
    end
    HybridApprox(L, translate_indices(L, CartesianRange(tuple(q_block..., L.cell_size))), name)
end

#############################################
get_string(Approx::A) where A<:ExactApprox = @sprintf("ul LD%s", "$(Approx.L.lattice_dims)")
get_string(Approx::A) where A<:ClusteredApprox = "cll LD$(Approx.L.lattice_dims) BD$((Approx.inner_cluster_dims))"
get_string(Approx::A) where A<:PureClassicalApprox = @sprintf("pcl LD%s", "$(Approx.L.lattice_dims)")
function get_string(Approx::A) where A<:HybridApprox
    data = Approx.name==:undef ? "$((Approx.q_spins))" : "$(Approx.name)"
    return "hl LD$(Approx.L.lattice_dims) QS:$data"
end

function get_value(A::AbstractApproximation, spin1::S, spin2::S) where {S<:Union{Integer,Tuple{Integer,Integer}}}
    return get_value(A.L, spin1::S, spin2::S)
end

get_q_spins(A::ExactApprox) = translate_indices(A.L, CartesianRange(tuple(A.L.lattice_dims..., A.L.cell_size)))
get_q_spins(A::ClusteredApprox) = translate_indices(A.L, CartesianRange(tuple(A.inner_cluster_dims..., A.L.cell_size)))
@inline get_q_spins(A::HybridApprox) = A.q_spins

get_Dh(A::AbstractQuantumApproximation) = 2^length(get_q_spins(A))

get_cl_spins(A::PureClassicalApprox) = translate_indices(A.L, CartesianRange(tuple(A.L.lattice_dims..., A.L.cell_size)))
@inline get_cl_spins(A::HybridApprox) = A.cl_spins

@inline get_all_spins(A::AbstractApproximation) = get_all_spins(A.L)

@inline get_central_spins(A::Approx) where Approx<:Union{ExactApprox, PureClassicalApprox} = get_central_spins(A.L, A.L.lattice_dims)
@inline get_central_spins(A::ClusteredApprox) = get_central_spins(A.L, A.inner_cluster_dims)
@inline function get_central_spins(A::HybridApprox)
    name = string(A.name)
    if contains(name, "block__")
        ds = split(name[8:end], '_')
        dims = map(x->parse(Int64, x), ds)
        return get_central_spins(A.L, Tuple(dims))
    else
        error("you need to define central spins for this case yourself")
    end
end

function get_spins_by_name(A::Approx, name::Symbol) where Approx<:AbstractApproximation
    if name==:all
        return get_all_spins(A)
    elseif name==:central
        return get_central_spins(A)
    end
end


################## helper functions for ClusteredApprox
function lin2cluster(A::ClusteredApprox{Lattice{D,F,T,C},D,D2}, ind::Int64) where {D,F,T,C,D2}
    i = ind2sub(A.transpose_dims, ind)
    return i[1:2:end]::NTuple{D,Int64}, i[2:2:end]::NTuple{D,Int64}
end
function cluster2lin(A::ClusteredApprox{Lattice{D,F,T,C},D,D2}, ind1::NTuple{D, Int64}, ind2::NTuple{D, Int64}) where {D,F,T,C,D2}
    nInd = Vector{Int64}()
    for i in 1:D
        push!(nInd, ind1[i])
        push!(nInd, ind2[i])
    end
    return sub2ind(A.transpose_dims, nInd...)
end

function get_first_cluster(A::ClusteredApprox{Lattice{D,F,T,C}}) where {D,F,T,C}
    q_spins = get_q_spins(A)
    cluster_spins = Vector{Tuple{NTuple{D,Int64}, Int64}}(length(q_spins))
    for i in eachindex(q_spins)
        cluster_spins[i] = (lin2cluster(A,q_spins[i][1])[1], q_spins[i][2])
        i+=1
    end
    return cluster_spins
end

#############################################

function get_positions(A::Approx, spins::Vector{NTuple{2,Int64}}) where Approx<:Union{ExactApprox, PureClassicalApprox}
    poss = Vector{Int64}(length(spins))
    l1 = A.L.cell_size
    l2 = A.L.tot_cell_num
    for i in eachindex(spins)
        poss[i] = sub2ind((l1,l2), spins[i][2], spins[i][1])
    end
    return sort!(poss)
end
function get_positions(A::ClusteredApprox, spins::Vector{NTuple{2,Int64}})
    poss = Vector{NTuple{2,Int64}}(length(spins))
    lc = A.L.cell_size
    sh = A.inner_cluster_dims
    for i in eachindex(spins)
        n,c = spins[i]
        in_index, out_index = lin2cluster(A, n)
        in_pos = sub2ind((lc, sh...),c,in_index...)
        out_pos = sub2ind(A.outer_cluster_dims, out_index...)
        poss[i] = (in_pos, out_pos)
    end
    return sort!(poss)
end
function get_positions(A::HybridApprox, spins::Vector{NTuple{2,Int64}})
    q_poss = Vector{Int64}()
    cl_poss = Vector{Int64}()
    for spin in spins
        pos1 = findfirst(A.q_spins, spin)
        pos2 = findfirst(A.cl_spins, spin)
        if pos1!=0
            push!(q_poss, pos1)
        elseif pos2!=0
            push!(cl_poss, pos2)
        else
            error("Something is very very wrong...")
        end
    end
    return sort!(q_poss), sort!(cl_poss)
end
