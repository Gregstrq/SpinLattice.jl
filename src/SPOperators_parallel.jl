const tolerance = 1e-16

const QSpins = Union{AbstractVector{Int64}, AbstractVector{NTuple{2, Int64}}, CartesianIndices}

function build_quantum_links(L::Lattice, quantum_spins::AbstractVector{NTuple{2, Int64}})
    q_links = Vector{Tuple{Int64,Int64,Float64}}()
    num_of_spins = length(quantum_spins)
    Dh = 2^num_of_spins
    for i in 1:num_of_spins-1
        for j in i+1:num_of_spins
            val = get_value(L, quantum_spins[i], quantum_spins[j])*0.25
            if val!=0.
                push!(q_links, (i, j, val))
            end
        end
    end
    return q_links
end

#function to calculate number of nnz matrix elements
#arising from $S_i^\sigma S_j^\sigma$ for $\sigma = x,y0$
#if J_x == J_y
function calculate_nnz_xx_yy(Dh::Int64, q_links::Vector{Tuple{Int64,Int64,Float64}})
    nnz_xx_yy = zero(Dh)
    for vec in 1:Dh
        for link in q_links
            nnz_xx_yy += div(1-getSign(vec, link[1], link[2]),2)
        end
    end
    return nnz_xx_yy
end

function build_Hamiltonian(L::Lattice, quantum_spins::AbstractVector{NTuple{2, Int64}}, T::Type)
    q_links = build_quantum_links(L, quantum_spins)
    num_of_links = length(q_links)
    num_of_spins = length(quantum_spins)
    Dh = 2^num_of_spins
    #calculate number of nonzero elements to store in the matrix
    @assert (L.Js |> collect |> norm) + (L.hs |> collect |> norm) > 0
    num_of_nnzs = zero(Dh)
    if ([L.Js[3], L.hs[3]] |> norm) > 0
        num_of_nnzs += Dh
    end
    if L.Js[1] == L.Js[2] == 0
        #do nothing
    elseif L.Js[1] == L.Js[2]
        num_of_nnzs += calculate_nnz_xx_yy(Dh, q_links)
    elseif L.Js[1] == -L.Js[2]
        num_of_nnzs += Dh*num_of_links - calculate_nnz_xx_yy(Dh, q_links)
    else
        num_of_nnzs += Dh*num_of_links
    end
    num_of_nnzs += Dh*num_of_spins*((L.hs[1]!=0) + (L.hs[2]!=0))
    #define arrays to store data
    data = SharedVector{T}(num_of_nnzs)
    rowind = SharedVector{Int64}(num_of_nnzs)
    colptr = SharedVector{Int64}(Dh + 1)
    #main cycle
    index = 1
    for vec in 1:Dh
        colptr[vec] = index
        data_temp = 0.
        for pos in 1:num_of_spins
            val, vec2 = pauliZ(vec, pos)
            data_temp -= 0.5*L.hs[3]*val
        end
        for link in q_links
            sign = getSign(vec, link[1], link[2])
            data_temp -= link[3]*L.Js[3]*sign
        end
        if abs(data_temp)>tolerance
            rowind[index] = vec
            data[index] = data_temp
            index+=1
        end
        for link in q_links
            vec2 = invert(vec, link[1], link[2])
            sign = getSign(vec, link[1], link[2])
            rowind_temp, data_temp = vec2, (L.Js[2]*sign-L.Js[1])*link[3]
            if abs(data_temp)>tolerance
                rowind[index] = rowind_temp
                data[index] = data_temp
                index += 1
            end
        end
        if L.hs[1]!=0.
            for pos in 1:num_of_spins
                val, rowind[index] = pauliX(vec, pos)
                data[index] = -0.5*val*L.hs[1]
                index += 1
            end
        end
        if L.hs[2]!=0.
            for pos in 1:num_of_spins
                val, rowind[index] = pauliY(vec, pos)
                data[index] = vec, -0.5*val*L.hs[1]
                index += 1
            end
        end
        colind = @view rowind[colptr[vec]:index-1]
        coldata = @view data[colptr[vec]:index-1]
        perm = sortperm(colind)
        permute!(colind, perm)
        permute!(coldata, perm)
    end
    colptr[end] = index
    #if(index<num_of_nnzs+1)
    #    resize!(rowind, index-1)
    #    resize!(data, index-1)
    #end
    return SharedSparseMatrixCSC(Dh, Dh, colptr, rowind, data)
end
build_Hamiltonian(L::Lattice{D,F,T,true}, quantum_spins::AbstractVector{NTuple{2, Int64}}) where {D,F,T} = build_Hamiltonian(L, quantum_spins, Complex{Float64})
build_Hamiltonian(L::Lattice{D,F,T,false}, quantum_spins::AbstractVector{NTuple{2, Int64}}) where {D,F,T} = build_Hamiltonian(L, quantum_spins, Float64)
build_Hamiltonian(A::Approx) where Approx<:AbstractQuantumApproximation = build_Hamiltonian(A.L, get_q_spins(A))


function build_Spin_Operator(Dh::Int64, spin::Int64, sigma::Int64)
    @assert 1<=sigma<=3
    num_of_nnzs = Dh
    #define arrays to store data
    data = SharedVector{Complex{Float64}}(num_of_nnzs)
    rowind = SharedVector{Int64}(num_of_nnzs)
    colptr = SharedVector{Int64}(Dh + 1)
    #main cycle
    index = 1
    for vec in 1:Dh
        colptr[vec] = index
        data[index], rowind[index] = pauli[sigma](vec, spin)
        index += 1
    end
    colptr[end] = index
    return SharedSparseMatrixCSC(Dh, Dh, colptr, rowind, data)
end
function build_Spin_Operator(Dh::Int64, spins::Vector{Int64}, sigma::Int64)
    if sigma==3
        num_of_nnzs = Dh
        #define arrays to store data
        data = SharedVector{Complex{Float64}}(num_of_nnzs)
        rowind = SharedVector{Int64}(num_of_nnzs)
        colptr = SharedVector{Int64}(Dh + 1)
        #main cycle
        index = 1
        for vec in 1:Dh
            colptr[vec] = index
            data_temp = 0.
            for pos in spins
                val, vec2 = pauliZ(vec,pos)
                data_temp += val
            end
            rowind[index] = vec
            data[index] = data_temp
            index+=1
        end
        colptr[end] = index
        return SharedSparseMatrixCSC(Dh, Dh, colptr, rowind, data)
    else
        num_of_nnzs = Dh*length(spins)
        #define arrays to store data
        data = SharedVector{Complex{Float64}}(num_of_nnzs)
        rowind = SharedVector{Int64}(num_of_nnzs)
        colptr = SharedVector{Int64}(Dh + 1)
        #main cycle
        index = 1
        for vec in 1:Dh
            colptr[vec] = index
            for pos in spins
                data[index], rowind[index] = pauli[sigma](vec, pos)
                index += 1
            end
            colind = @view rowind[colptr[vec]:index-1]
            coldata = @view data[colptr[vec]:index-1]
            perm = sortperm(colind)
            permute!(colind, perm)
            permute!(coldata, perm)
        end
        colptr[end] = index
        return SharedSparseMatrixCSC(Dh, Dh, colptr, rowind, data)
    end
end

function build_ClassicalInteractions(L::Lattice{D,typeof(nearest_neighbours),T,C}, cl_spins::Vector{Tuple{Int64,Int64}}) where {D,T,C}
    cl_num = length(cl_spins)
    IM = zeros((cl_num, cl_num))
    for i in 1:cl_num-1 for j in i+1:cl_num
        IM[i,j] = get_value(L, cl_spins[i], cl_spins[j])
        IM[j,i] = IM[i,j]
    end end
    return shared_sparse(IM)::SharedSparseMatrixCSC{Float64, Int64}
end
function build_ClassicalInteractions(L::Lattice{D,MDI{D2},T,C}, cl_spins::Vector{Tuple{Int64,Int64}}) where {D,D2,T,C}
    cl_num = length(cl_spins)
    IM = SharedMatrix{Float64}((cl_num, cl_num))
    fill!(IM,0.0)
    for i in 1:cl_num-1 for j in i+1:cl_num
        IM[i,j] = get_value(L, cl_spins[i], cl_spins[j])
        IM[j,i] = IM[i,j]
    end end
    print(issymmetric(IM),"\n")
    return Symmetric(IM)::Symmetric{Float64, SharedMatrix{Float64}}
end
function build_ClassicalInteractions(L::Lattice{D,MDI{D2},T,C}, cl_spins::Vector{Tuple{Int64,Int64}}, factor::Float64) where {D,D2,T,C}
    cl_num = length(cl_spins)
    IM = SharedMatrix{Float64}((cl_num, cl_num))
    fill!(IM,0.0)
    for i in 1:cl_num-1 for j in i+1:cl_num
        IM[i,j] = get_value(L, cl_spins[i], cl_spins[j])*factor
        IM[j,i] = IM[i,j]
    end end
    print(issymmetric(IM),"\n")
    return Symmetric(IM)::Symmetric{Float64, SharedMatrix{Float64}}
end
build_ClassicalInteractions(A::Approx) where Approx<:Union{PureClassicalApprox, HybridApprox} = build_ClassicalInteractions(A.L, get_cl_spins(A))
build_ClassicalInteractions(A::PureClassicalApprox, factor::Float64) = build_ClassicalInteractions(A.L, get_cl_spins(A), factor)

@inline build_InterClusterInteractions(A::ClusteredApprox) = _build_InterClusterInteractions(A)
function build_InterClusterInteractions(A::ClusteredApprox{Lattice{D,typeof(nearest_neighbours),T,C}}) where {D,T,C}
    edge_spins, IM = _build_InterClusterInteractions(A)
    return edge_spins::Vector{Int64}, shared_sparse(IM)::SharedSparseMatrixCSC{Float64,Int64}
end

function _build_InterClusterInteractions(A::ClusteredApprox)
    edge_spins = Set{Int64}()
    cluster_spins = get_first_cluster(A)
    links = Vector{Tuple{Int64,Int64,Int64,Float64}}()
    cluster_pos1 = ind2sub(A.outer_cluster_dims, 1)
    cl_pos1 = collect(cluster_pos1)
    for sp1 in eachindex(cluster_spins)
        index1 = (cluster2lin(A, cluster_spins[sp1][1],cluster_pos1), cluster_spins[sp1][2])
        for cluster in 2:A.cluster_num
            cluster_pos2 = ind2sub(A.outer_cluster_dims, cluster)
            for sp2 in eachindex(cluster_spins)
                index2 = (cluster2lin(A, cluster_spins[sp2][1], cluster_pos2), cluster_spins[sp2][2])
                val = get_value(A, index1, index2)
                if val!=0.0
                    push!(edge_spins, sp1)
                    push!(links, (sp1, sp2, cluster, val))
                end
            end
        end
    end
    io = open("test.txt", "w")
    #print(io, "$edge_spins\n$links\n")
    eedge_spins = Int64[edge_spins...]
    sort!(eedge_spins)
    len = length(eedge_spins)*A.cluster_num
    IM = SharedMatrix{Float64}(len, len)
    fill!(IM, 0.0)
    for cluster in 1:A.cluster_num
        cl_pos = ind2sub(A.outer_cluster_dims, cluster)
        delta = collect(cl_pos) - cl_pos1
        #print(io,"$delta\n\n")
        for link in links
            p1 = sub2ind((length(eedge_spins), A.cluster_num), findfirst_c(eedge_spins, link[1]), cluster)
            cl_p2 = (collect(ind2sub(A.outer_cluster_dims, link[3])) .+ delta .- 1).%collect(A.outer_cluster_dims) .+ 1
            #print(io,"$cl_p2\n")
            #print(io,"$link\n")
            p2 = sub2ind((length(eedge_spins), A.cluster_num), findfirst_c(eedge_spins, link[2]), sub2ind(A.outer_cluster_dims, cl_p2...))
            #print(io,p1,"\t",p2,"\n")
            IM[p1,p2] = link[4]
        end
        #print(io,"\n\n\n")
    end
    close(io)
    IM .= .-IM.*sqrt(2^(length(cluster_spins))+1).*0.25
    return eedge_spins::Vector{Int64}, IM::SharedMatrix{Float64}
end

function _build_QuantumClassicalInteractions(A::HybridApprox)
    edge_spins = Set{Float64}()
    q_spins = get_q_spins(A)
    cl_spins = get_cl_spins(A)
    links = Vector{Tuple{Int64,Int64,Float64}}()
    for q in eachindex(q_spins)
        for cl in eachindex(cl_spins)
            val = get_value(A, q_spins[q], cl_spins[cl])*0.5
            if val!= 0.0
                push!(edge_spins, q)
                push!(links, (q,cl,val))
            end
        end
    end
    edge_spins = Int64[edge_spins...]
    sort!(edge_spins)
    q2cl = SharedMatrix{Float64}(length(cl_spins), length(edge_spins))
    fill!(q2cl, 0.0)
    for link in links
        q2cl[link[2], findfirst_c(edge_spins,link[1])] += link[3]
    end
    cl2q = SharedMatrix{Float64}(length(edge_spins), length(cl_spins))
    transpose!(cl2q,q2cl)
    rmul!(cl2q, -1.0)
    q2cl .*= sqrt(2^length(q_spins)+1)
    return edge_spins, q2cl, cl2q
end
@inline build_QuantumClassicalInteractions(A::HybridApprox{Lattice{D,MDI{D2},T,C,B}}) where {D,D2,T,C,B} = _build_QuantumClassicalInteractions(A)

function build_QuantumClassicalInteractions(A::HybridApprox{Lattice{D,typeof(nearest_neighbours),T,C,B}}) where {D,T,C,B}
    edge_spins = Set{Float64}()
    q_spins = get_q_spins(A)
    cl_spins = get_cl_spins(A)
    links = Vector{Tuple{Int64,Int64,Float64}}()
    for q in eachindex(q_spins)
        for cl in eachindex(cl_spins)
            val = get_value(A, q_spins[q], cl_spins[cl])*0.5
            if val!= 0.0
                push!(edge_spins, q)
                push!(links, (q,cl,val))
            end
        end
    end
    edge_spins = Int64[edge_spins...]
    sort!(edge_spins)
    q2cl = Matrix{Float64}(undef, length(cl_spins), length(edge_spins))
    fill!(q2cl, 0.0)
    for link in links
        q2cl[link[2], findfirst_c(edge_spins,link[1])] += link[3]
    end
    cl2q = -transpose(q2cl)
    q2cl .*= sqrt(2^length(q_spins)+1)
    return edge_spins, shared_sparse(q2cl)::SharedSparseMatrixCSC{Float64,Int64}, shared_sparse(cl2q)::SharedSparseMatrixCSC{Float64,Int64}
end

#####################################
#What is written here works only for long-range coupling.


function build_InterLattice_Couplings(A::CompositeApproximation{Approx1,Approx2}, edge_spins_len::Int64) where {Approx1<:HybridApprox, Approx2<:PureClassicalApprox}
    q1tocl2, cl2toq1 = build_InterLattice_q2cl(A.A1, A.A2, edge_spins_len, A.gamma)
    cl1tocl2, cl2tocl1 = build_InterLattice_cl2cl(A.A1, A.A2, A.gamma)
    return q1tocl2, cl2toq1, cl1tocl2, cl2tocl1
end

function build_InterLattice_q2cl(A1::Approx1, A2::Approx2, edge_spins_len::Int64, gamma::Float64) where {Approx1<:HybridApprox, Approx2<:PureClassicalApprox}
    q_spins = get_q_spins(A1)
    @assert length(q_spins)==edge_spins_len
    cl_spins = get_cl_spins(A2)
    L1 = A1.L
    L2 = A2.L
    q1tocl2 = SharedMatrix{Float64}(length(cl_spins), edge_spins_len)
    fill!(q1tocl2, 0.0)
    for q in eachindex(q_spins)
        for cl in eachindex(cl_spins)
            q1tocl2[cl, q] = get_value(L1, L2, q_spins[q], cl_spins[cl])*0.5*gamma
        end
    end
    cl2toq1 = SharedMatrix{Float64}(edge_spins_len, length(cl_spins))
    transpose!(cl2toq1,q1tocl2)
    scale!(cl2toq1,-1.0)
    q1tocl2 .*= sqrt(2^edge_spins_len+1)
    return q1tocl2, cl2toq1
end

function build_InterLattice_cl2cl(A1::Approx1, A2::Approx2, gamma::Float64) where {Approx1<:HybridApprox, Approx2<:PureClassicalApprox}
    cl_spins1 = get_cl_spins(A1)
    cl_spins2 = get_cl_spins(A2)
    L1 = A1.L
    L2 = A2.L
    cl1tocl2 = SharedMatrix{Float64}(length(cl_spins2), length(cl_spins1))
    fill!(cl1tocl2, 0.0)
    for cl2 in eachindex(cl_spins2)
        for cl1 in eachindex(cl_spins1)
            cl1tocl2[cl2, cl1] = get_value(L1, L2, cl_spins1[cl1], cl_spins2[cl2])*gamma
        end
    end
    cl2tocl1 = SharedMatrix{Float64}(length(cl_spins1), length(cl_spins2))
    transpose!(cl2tocl1, cl1tocl2)
    return cl1tocl2, cl2tocl1
end
