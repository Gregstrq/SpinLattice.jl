import Base.getindex

struct SharedSparseMatrixCSC{Tv, Ti<:Integer} <: AbstractSparseMatrix{Tv, Ti}
    m::Int
    n::Int
    colptr::SharedVector{Ti}
    rowval::SharedVector{Ti}
    nzval::SharedVector{Tv}

    function SharedSparseMatrixCSC{Tv,Ti}(m::Integer, n::Integer, colptr::SharedVector{Ti}, rowval::SharedVector{Ti},
                                    nzval::SharedVector{Tv}) where {Tv,Ti<:Integer}
        m < 0 && throw(ArgumentError("number of rows (m) must be ≥ 0, got $m"))
        n < 0 && throw(ArgumentError("number of columns (n) must be ≥ 0, got $n"))
        new(Int(m), Int(n), colptr, rowval, nzval)
end
end

function SharedSparseMatrixCSC(m::Integer, n::Integer, colptr::SharedVector, rowval::SharedVector, nzval::SharedVector)
    Tv = eltype(nzval)
    Ti = promote_type(eltype(colptr), eltype(rowval))
    SharedSparseMatrixCSC{Tv,Ti}(m, n, colptr, rowval, nzval)
end

Base.size(S::SharedSparseMatrixCSC) = (S.m, S.n)

function A_mul_B!(alpha::Number, A::SharedSparseMatrixCSC, B::StridedVecOrMat, beta::Number, C::StridedVecOrMat)
    A.n == size(B, 1) || throw(DimensionMismatch())
    A.m == size(C, 1) || throw(DimensionMismatch())
    size(B, 2) == size(C, 2) || throw(DimensionMismatch())
    nzv = A.nzval
    rv = A.rowval
    if beta != 1
        beta != 0 ? scale!(C, beta) : fill!(C, zero(eltype(C)))
    end
    for k = 1:size(C, 2)
        for col = 1:A.n
            alphaxj = alpha*B[col,k]
            @inbounds for j = A.colptr[col]:(A.colptr[col + 1] - 1)
                C[rv[j], k] += nzv[j]*alphaxj
            end
        end
    end
    C
end

A_mul_B!(C::StridedVecOrMat, A::SharedSparseMatrixCSC, B::StridedVecOrMat) = A_mul_B!(one(eltype(B)), A, B, zero(eltype(C)), C)

convert(::Type{SharedSparseMatrixCSC}, M::Matrix) = sparse(M)
convert(::Type{SharedSparseMatrixCSC}, M::AbstractMatrix{Tv}) where {Tv} = convert(SharedSparseMatrixCSC{Tv,Int}, M)
convert(::Type{SharedSparseMatrixCSC{Tv}}, M::AbstractMatrix{Tv}) where {Tv} = convert(SharedSparseMatrixCSC{Tv,Int}, M)
function convert(::Type{SharedSparseMatrixCSC{Tv,Ti}}, M::AbstractMatrix) where {Tv,Ti}
    (I, J, V) = findnz(M)
    eltypeTiI = Base.convert(Vector{Ti}, I)
    eltypeTiJ = Base.convert(Vector{Ti}, J)
    eltypeTvV = Base.convert(Vector{Tv}, V)
    return shared_sparse_IJ_sorted(eltypeTiI, eltypeTiJ, eltypeTvV, size(M)...)
end

shared_sparse(A::AbstractMatrix{Tv}) where {Tv} = convert(SharedSparseMatrixCSC{Tv,Int}, A)

shared_sparse_IJ_sorted(I,J,V,m,n) = shared_sparse_IJ_sorted(I,J,V,m,n,+)
function shared_sparse_IJ_sorted(I::AbstractVector{Ti}, J::AbstractVector{Ti},
                                  V::AbstractVector{Tv},
                           m::Integer, n::Integer, combine::Function) where Ti<:Integer where Tv
    m = m < 0 ? 0 : m
    n = n < 0 ? 0 : n
    if isempty(V); return spzeros(eltype(V),Ti,m,n); end

    cols = zeros(Ti, n+1)
    cols[1] = 1  # For cumsum purposes
    cols[J[1] + 1] = 1

    lastdup = 1
    ndups = 0
    I_lastdup = I[1]
    J_lastdup = J[1]
    L = length(I)

    @inbounds for k=2:L
        if I[k] == I_lastdup && J[k] == J_lastdup
            V[lastdup] = combine(V[lastdup], V[k])
            ndups += 1
        else
            cols[J[k] + 1] += 1
            lastdup = k-ndups
            I_lastdup = I[k]
            J_lastdup = J[k]
            if ndups != 0
                I[lastdup] = I_lastdup
                V[lastdup] = V[k]
            end
        end
    end

    colptr = cumsum!(similar(cols), cols)

    # Allow up to 20% slack
    if ndups > 0.2*L
        numnz = L-ndups
        deleteat!(I, (numnz+1):L)
        deleteat!(V, (numnz+1):length(V))
    end

    return SharedSparseMatrixCSC(m, n, SharedVector{Ti}(colptr), SharedVector{Ti}(I), SharedVector{Tv}(V))
end

function getindex(A::SharedSparseMatrixCSC{T}, i0::Int, i1::Int) where T
    if !(1 <= i0 <= A.m && 1 <= i1 <= A.n); throw(BoundsError()); end
    first = A.colptr[i1]
    last = A.colptr[i1+1]-1
    while first <= last
        mid = (first + last) >> 1
        t = A.rowval[mid]
        if t == i0
            return A.nzval[mid]
        elseif t > i0
            last = mid - 1
        else
            first = mid + 1
        end
    end
    return zero(T)
end
