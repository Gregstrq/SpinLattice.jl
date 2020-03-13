sub2ind(dims::NTuple{N,Integer}, I::Integer...) where {N} = LinearIndices(dims)[I...]
ind2sub(dims::NTuple{N,Integer}, ind::Integer) where {N} = Tuple(CartesianIndices(dims)[ind])

findfirst_c(A,v) = something(findfirst(isequal(v), A), 0)

#function findnz(A::AbstractMatrix)
#    I = findall(!iszero, A)
#    return (getindex.(I, 1), getindex.(I, 2), A[I])
#end
