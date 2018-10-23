function InterMap(L::Lattice{D}, cr::CartesianRange{CartesianIndex{D}}, filename::AbstractString, cutoff::Int64, origin::Vector{Float64}) where D
    vals = Vector{Float64}()
    distances = Vector{Float64}()
    indices = Vector{Tuple{NTuple{D,Int64},Int64}}()
    vectors = Vector{Vector{Float64}}()
    for ci in cr
        for c = 1:L.cell_size
            push!(indices, (ci.I, c))
            vector = L.basis_vectors*collect(ci.I) + L.cell_vectors[c]-origin
            push!(distances, norm(vector))
            push!(vals, L.Jfunc(vector))
            push!(vectors, vector)
        end
    end
    p = sortperm(vals; by = abs, rev = true)
    permute!(vals, p)
    permute!(distances, p)
    permute!(indices, p)
    permute!(vectors, p)
    io = open(filename, "w")
    for i = 1:min(cutoff, length(vals))
        print(io, vals[i], "\t\t", distances[i], "\t\t", vectors[i], "\t\t", indices[i][1], "\t", indices[i][2], "\n")
    end
    close(io)
end

function get_num_eff(L::Lattice{D}, cr::CartesianRange{CartesianIndex{D}}, origin::Vector{Float64}) where D
    jj = 0.0
    j = 0.0
    for ci in cr
        for c = 1:L.cell_size
            vector = L.basis_vectors*collect(ci.I) + L.cell_vectors[c]-origin
            if norm(vector) != 0.0
                val = L.Jfunc(vector)^2
                jj += val
                j += val^2
            end
        end
    end
    return jj^2/j
end
