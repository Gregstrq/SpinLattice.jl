mutable struct Lattice{D,F,T,C}
    lattice_dims::NTuple{D,Int64}
    basis_vectors::Matrix{T}
    cell_vectors::Vector{Vector{T}}
    increments::Vector{Vector{T}}
    Js::NTuple{3,Float64}
    hs::NTuple{3,Float64}
    Jfunc::Interaction{F}
    cell_size::Int64
    tot_cell_num::Int64
    tot_spin_num::Int64
    basis_name::Symbol
    cell_name::Symbol
    ccell_cartesian::NTuple{D,Int64}
    ccell_linear::Int64
    complex::Val{C}
    function Lattice(lattice_dims::NTuple{D,Int64}, Js::NTuple{3,Float64}, Jfunc::Interaction{F}, hs::NTuple{3,Float64}, basis_vectors::Matrix{T}, cell_vectors::Vector{Vector{T}}, basis_name::Symbol, cell_name::Symbol) where {D,F,T}
        Dim = length(lattice_dims)
        assert(det(basis_vectors) != 0)
        if findfirst(cell_vectors, zeros(T,Dim))==0
            push!(cell_vectors,zeros(T,Dim))
        end
        cell_size = length(cell_vectors)
        increments = Vector{Vector{Float64}}()
        for incr in product([[-1,0,1] for i in 1:Dim]...)
            increment = zeros(Float64, Dim)
            for i in 1:Dim
                @. increment += basis_vectors[:,i]*incr[i]*lattice_dims[i]
            end
            push!(increments, increment)
        end
        tot_cell_num = prod(lattice_dims)
        tot_spin_num = tot_cell_num*cell_size
        cc = hs[2]!=0.
        ccell_cartesian = div.(lattice_dims,2).+1
        ccell_linear = sub2ind(lattice_dims, ccell_cartesian...)
        new{D,F,T,cc}(lattice_dims, basis_vectors, cell_vectors, increments, Js, hs, Jfunc, cell_size, tot_cell_num, tot_spin_num, basis_name, cell_name, ccell_cartesian, ccell_linear, Val{cc}())
    end
end

function Lattice(dims, Js, Jfunc, hs, T::Type = Int64)
    basis_vectors = eye(T, length(dims), length(dims))
    cell_vectors = Vector{Vector{T}}()
    Lattice(dims, Js, Jfunc, hs, basis_vectors, cell_vectors, :cubic, :simple)
end
Lattice(dims, Js, Jfunc, hs, basis_vector::Matrix{T}, basis_name::Symbol) where {T} = Lattice(dims, Js, Jfunc, hs, basis_vector, Vector{Vector{T}}(), basis_name, :simple)

function get_vector(L::Lattice, spin::I) where {I}
    L.basis_vectors*collect(ind2sub(L.lattice_dims, spin))
end
function get_vector(L::Lattice, spin::Tuple{I, I}) where {I}
    L.basis_vectors*collect(ind2sub(L.lattice_dims, spin[1])) + L.cell_vectors[spin[2]]
end

function check_spin(L::Lattice, spin::I) where {I}
    assert(0<spin<=L.tot_spin_num)
end
function check_spin(L::Lattice, spin::Tuple{I, I}) where {I}
    assert((0<spin[1]<=prod(L.lattice_dims)) && (0<spin[2]<=L.cell_size))
end

function get_patch_vector(L::Lattice, spin1::S, spin2::S) where {S<:Tuple{Integer,Integer}}
    check_spin(L, spin1)
    check_spin(L, spin2)
    new_s2_cart = ind2sub(L.lattice_dims, spin2[1]) .- ind2sub(L.lattice_dims, spin1[1]) .+ L.ccell_cartesian
    new_s2_cart = mod.(new_s2_cart.-1, L.lattice_dims).+1
    return get_vector(L, (sub2ind(L.lattice_dims, new_s2_cart...), spin2[2])) .- get_vector(L, (L.ccell_linear, spin1[2]))
end

@inline get_patch_vector(L::Lattice, spin1::S, spin2::S) where {S<:Integer} = get_patch_vector(L, (spin1, 1), (spin2, 1))

function get_smallest_vector(L::Lattice, spin1::S, spin2::S) where {S<:Union{Integer,Tuple{Integer,Integer}}}
    check_spin(L, spin1)
    check_spin(L, spin2)
    vector = get_vector(L, spin2) .- get_vector(L, spin1)
    smallest = deepcopy(vector)
    temp_vector = deepcopy(vector)
    r = norm(vector)
    for incr in L.increments
        temp_vector .= vector .+ incr
        temp_r = norm(temp_vector)
        if temp_r<r
            smallest = deepcopy(temp_vector)
            r = temp_r
        end
    end
    return smallest
end

function get_value(L::Lattice, spin1::S, spin2::S) where {S<:Union{Integer,Tuple{Integer,Integer}}}
    return L.Jfunc(get_patch_vector(L, spin1, spin2))
end

function get_string(L::Ltype) where {Ltype<:Lattice}
    @sprintf("D%d d%d S:%s C:%s Js(%.3f,%.3f,%.3f) Jf%s H(%.3f,%.3f,%.3f)", length(L.lattice_dims), L.cell_size, string(L.basis_name), string(L.cell_name), L.Js[1], L.Js[2], L.Js[3], func_type(L.Jfunc), L.hs[1], L.hs[2], L.hs[3])
end

function get_string(vec::Vector)
    str = "("
    i = 1
    str *= "$(vec[i])"
    i += 1
    while i<= length(vec)
        str *= ", $(vec[i])"
        i += 1
    end
    str *= ")"
    return str
end

function translate_indices(L::Lattice, indices::Vector{Tuple{Int64, Int64}})
    sort!(indices)
    return indices
end
function translate_indices(L::Lattice, indices::AbstractVector{Int64})
    nIndices = Vector{Tuple{Int64,Int64}}()
    for index in indices for c in 1:L.cell_size
        push!(nIndices, (index, c))
    end end
    sort!(nIndices)
    return nIndices
end
function translate_indices(L::Lattice{D,F,T,C}, indices::CartesianRange{CartesianIndex{D2}}) where {D,F,T,C,D2}
    assert(length(first(indices).I)==length(L.lattice_dims)+1)
    nIndices = Vector{NTuple{2, Int64}}()
    for index in indices
        push!(nIndices, (sub2ind(L.lattice_dims, index.I[1:D]...), index.I[end]))
    end
    sort!(nIndices)
    return nIndices
end

function read_indices(L::Lattice{D,F,T,C}, indices::Vector{NTuple{D, Int64}}) where {D,F,T,C}
    fff(index::NTuple{D}) = sub2ind(L.lattice_dims, index...)
    return translate_indices(L, map(fff, indices))
end

get_all_spins(L::Lattice) = NTuple{2,Int64}[(l,c) for l in Base.OneTo(L.tot_cell_num) for c in Base.OneTo(L.cell_size)]

function get_central_spins(L::Lattice, dims::NTuple{N,Int64}) where {N}
    v = Vector{Vector{Int64}}()
    for i in 1:N
        s = div(dims[i],2)
        if dims[i]%2==0
            push!(v, [s,s+1])
        else
            push!(v, [s+1])
        end
    end
    c_cells = product(v...)
    return translate_indices(L, [sub2ind(L.lattice_dims, idxs...) for idxs in product(v...)])
end

change_dims(dims1, dims2, index) = ind2sub(dims2, sub2ind(dims1, index...))

##########################3

struct SpinArray
    name::Symbol
    spins::Vector{NTuple{2,Int64}}
end
@inline SpinArray(spins::Vector{NTuple{2,Int64}}, name = :undef) = SpinArray(name, spins)
@inline SpinArray(L::Lattice{D}, spins::Vector{NTuple{D,Int64}}, name = :undef) where {D} = SpinArray(name, read_indices(L, spins))
function SpinArray(L::Lattice{D}, spins::Vector{T}, name = :undef) where T<:Tuple{NTuple{D, Int64}, Int64} where D
    fff(index::T) = (sub2ind(L.lattice_dims, index[1]...), index[2])
    return SpinArray(name, map(fff, spins))
end

@inline get_string(s::Symbol) = string(s)
@inline get_string(s::SpinArray) = s.name==:undef ? "$(s.spins)" : string(s.name)
get_string(link::Tuple{T1,T2}) where {T1,T2} = @sprintf("%s_%s", get_string(link[1]), get_string(link[2]))


Base.isless(s1::SpinArray, s2::Symbol) = Base.isless(s1.name, s2)
Base.isless(s1::Symbol, s2::SpinArray) = Base.isless(s1, s2.name)
Base.isless(s1::SpinArray, s2::SpinArray) = Base.isless(s1.name, s2.name)
