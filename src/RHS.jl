abstract type AbstractRHSFunction end
function build_RHS_function end

@inline A_mul_B!(α::T, A::SharedMatrix{T}, B::Vector{T}, β::T, C::Vector{T}) where T<:Union{Float64, Complex{Float64}} = BLAS.gemv!('N', α, A, B, β, C)
@inline A_mul_B!(α::T, A::Symmetric{T}, B::Vector{T}, β::T, C::Vector{T}) where T<:Union{Float64, Complex{Float64}} = BLAS.symv!(A.uplo, α, A.data, B, β, C)

@inline Base.similar(VA::VectorOfArray, ::Type{T}, dims::Base.Dims{N}) where {T,N} = similar(VA, T)
@inline Base.similar(VA::VectorOfArray, dims::Base.Dims{N}) where {N} = similar(VA)

############# Exact Approximation
struct ExactRHSFunction{TS<:AbstractSparseMatrix} <: AbstractRHSFunction
    H::TS
end
function build_RHS_function(A::Approx) where {Approx<:ExactApprox}
    ExactRHSFunction(build_Hamiltonian(A))
end

function (EF::ExactRHSFunction)(t, state::T, dstate::T) where {T<:AbstractVector{Complex{Float64}}}
    A_mul_B!(-1im, EF.H, state, zero(Complex{Float64}), dstate)
end
function (EF::ExactRHSFunction)(t, u::T, du::T) where {T<:VectorOfArray{Complex{Float64},2,Array{T1,1}}} where T1<:AbstractVector{Complex{Float64}}
    @inbounds for i in eachindex(u)
        A_mul_B!(-1im, EF.H, u[i], zero(Complex{Float64}), du[i])
    end
end

############# Structured Approximation

struct ClusteredRHSFunction{TH<:AbstractSparseMatrix, TIM<:Union{AbstractMatrix,AbstractSparseMatrix}, T3<:AbstractSparseMatrix} <: AbstractRHSFunction
    Js::NTuple{3, Float64}
    H::TH
    edgeSpinOperators::Matrix{T3}###row - spin position, col - spin projection
    average_values::VectorOfArray{Float64, 2, Array{Vector{Float64},1}}
    #average_values_arr::VectorOfArray{Float64, 3, Array{Matrix{Float64},1}}
    cl_field::VectorOfArray{Float64, 2, Array{Vector{Float64},1}}
    #cl_field_arr::VectorOfArray{Float64, 3, Array{Matrix{Float64},1}}
    cluster_temp_state::Vector{Complex{Float64}}
    IM::TIM
    num_of_espins::Int64
end
function build_RHS_function(A::Approx) where {Approx<:ClusteredApprox}
    H = build_Hamiltonian(A)
    Dh = H.m
    edge_spins, IM = build_InterClusterInteractions(A)
    average_values = VectorOfArray([zeros(Float64, size(IM,1)), zeros(Float64, size(IM,1)), zeros(Float64, size(IM,1))])
    #average_values_arr = VectorOfArray([reshape(average_values[i], (length(edge_spins), A.cluster_num)) for i in eachindex(average_values)])
    cl_field = VectorOfArray([zeros(Float64, size(IM,1)), zeros(Float64, size(IM,1)), zeros(Float64, size(IM,1))])
    #cl_field_arr = VectorOfArray([reshape(cl_field[i], (length(edge_spins), A.cluster_num)) for i in eachindex(cl_field)])
    edgeSpinOperators = Matrix{SharedSparseMatrixCSC{Complex{Float64}, Int64}}(3,length(edge_spins))
    for i in eachindex(edge_spins) for sigma in 1:3
        edgeSpinOperators[sigma, i] = build_Spin_Operator(Dh, edge_spins[i], sigma)
    end end
    cluster_temp_state = Vector{Complex{Float64}}(Dh)
    ClusteredRHSFunction(A.L.Js, H, edgeSpinOperators, average_values, cl_field, cluster_temp_state, IM, length(edge_spins))
end

function (ClF::ClusteredRHSFunction)(t, u::T, du::T) where T<:VectorOfArray{Complex{Float64},2,Array{Array{Complex{Float64},1},1}}
    dims = (ClF.num_of_espins, length(u))
    @inbounds for cluster in eachindex(u)
        cl_norm = norm(u[cluster])
        for espin in 1:ClF.num_of_espins for sigma in 1:3
            A_mul_B!(ClF.cluster_temp_state, ClF.edgeSpinOperators[sigma, espin], u[cluster])
            ClF.average_values[sigma][sub2ind(dims, espin, cluster)] = real(dot(u[cluster], ClF.cluster_temp_state))/cl_norm
        end end
    end
    @inbounds for sigma in 1:3
        A_mul_B!(ClF.Js[sigma], ClF.IM, ClF.average_values[sigma], zero(Float64), ClF.cl_field[sigma])
    end
    @inbounds for cluster in eachindex(u)
        A_mul_B!(du[cluster], ClF.H, u[cluster])
        for espin in 1:ClF.num_of_espins for sigma in 1:3
            A_mul_B!(ClF.cl_field[sigma][sub2ind(dims, espin, cluster)], ClF.edgeSpinOperators[sigma, espin], u[cluster], one(Complex{Float64}), du[cluster])
        end end
        du[:,cluster] .*= -1im
    end
end

############# PureClassical Approximation

struct PureClassicalRHSFunction{T<:AbstractMatrix} <: AbstractRHSFunction
    Js::NTuple{3,Float64}
    hs::NTuple{3, Float64}
    IM::T
    cl_field::VectorOfArray{Float64, 2, Vector{Vector{Float64}}}
end
function build_RHS_function(A::Approx) where Approx<:PureClassicalApprox
    cl_field = deepcopy(randomState(A))
    IM = build_ClassicalInteractions(A)
    PureClassicalRHSFunction{typeof(IM)}(A.L.Js, A.L.hs, IM, cl_field)
end
function build_RHS_function(A::PureClassicalApprox, factor::Float64)
    cl_field = deepcopy(randomState(A))
    IM = build_ClassicalInteractions(A, factor)
    PureClassicalRHSFunction{typeof(IM)}(A.L.Js, A.L.hs, IM, cl_field)
end

function (PCF::PureClassicalRHSFunction)(t, u::T, du::T) where T<:VectorOfArray{Float64, 2, Vector{Vector{Float64}}}
    @inbounds for sigma in 1:3
        fill!(PCF.cl_field[sigma], PCF.hs[sigma])
        A_mul_B!(PCF.Js[sigma], PCF.IM, u[sigma], one(Float64), PCF.cl_field[sigma])
    end
    @. du[:,1] = u[2]*PCF.cl_field[3] - u[3]*PCF.cl_field[2]
    @. du[:,2] = u[3]*PCF.cl_field[1] - u[1]*PCF.cl_field[3]
    @. du[:,3] = u[1]*PCF.cl_field[2] - u[2]*PCF.cl_field[1]
end

############# Hybrid Approximation

struct HybridRHSFunction{TH<:AbstractSparseMatrix, T1<:AbstractMatrix, T2<:AbstractMatrix, T3<:AbstractSparseMatrix} <: AbstractRHSFunction
    Js::NTuple{3,Float64}
    hs::NTuple{3, Float64}
    H::TH
    IM::T1
    q2cl::T2
    cl2q::T2
    average_values::VectorOfArray{Float64, 2, Vector{Vector{Float64}}}
    edgeSpinOperators::Matrix{T3}
    cl_field::ArrayPartition{Float64, NTuple{2,VectorOfArray{Float64, 2, Vector{Vector{Float64}}}}}
    q_temp_state::Vector{Complex{Float64}}
end
function build_RHS_function(A::HybridApprox)
    H = build_Hamiltonian(A)
    Dh = H.m
    IM = build_ClassicalInteractions(A)
    edge_spins, q2cl, cl2q = build_QuantumClassicalInteractions(A)
    average_values = random_cl_state(length(edge_spins))
    cl_field = ArrayPartition(random_cl_state(length(edge_spins)), random_cl_state(A.cl_num))
    edgeSpinOperators = Matrix{SharedSparseMatrixCSC{Complex{Float64}, Int64}}(length(edge_spins),3)
    for espin in eachindex(edge_spins)
        for sigma in 1:3
            edgeSpinOperators[espin, sigma] = build_Spin_Operator(Dh, edge_spins[espin], sigma)
        end
    end
    q_temp_state = Vector{Complex{Float64}}(Dh)
    HybridRHSFunction(A.L.Js, A.L.hs, H, IM, q2cl, cl2q, average_values, edgeSpinOperators, cl_field, q_temp_state)
end

function (HF::HybridRHSFunction)(t, u::T, du::T) where T<:ArrayPartition{Complex{Float64}, Tuple{Vector{Complex{Float64}}, VectorOfArray{Float64, 2, Vector{Vector{Float64}}}}}
    qnorm = norm(u.x[1])
    @inbounds for sigma in 1:3
        for q in 1:size(HF.edgeSpinOperators,1)
            A_mul_B!(HF.q_temp_state, HF.edgeSpinOperators[q, sigma], u.x[1])
            HF.average_values[q, sigma] = real(dot(HF.q_temp_state, u.x[1]))/qnorm
        end
    end
    A_mul_B!(du.x[1], HF.H, u.x[1])
    @inbounds for sigma in 1:3
        A_mul_B!(HF.Js[sigma], HF.cl2q, u.x[2][sigma], zero(Float64), HF.cl_field.x[1][sigma])
        for q in 1:size(HF.edgeSpinOperators, 1)
            A_mul_B!(HF.cl_field.x[1][q,sigma], HF.edgeSpinOperators[q, sigma], u.x[1], one(Complex{Float64}), du.x[1])
        end
        fill!(HF.cl_field.x[2][sigma], HF.hs[sigma])
        A_mul_B!(HF.Js[sigma], HF.IM, u.x[2][sigma],  one(Float64), HF.cl_field.x[2][sigma])
        A_mul_B!(HF.Js[sigma], HF.q2cl, HF.average_values[sigma],  one(Float64), HF.cl_field.x[2][sigma])
    end
    du.x[1] .*= -1im
    @. du.x[2][:,1] = u.x[2][2]*HF.cl_field.x[2][3] - u.x[2][3]*HF.cl_field.x[2][2]
    @. du.x[2][:,2] = u.x[2][3]*HF.cl_field.x[2][1] - u.x[2][1]*HF.cl_field.x[2][3]
    @. du.x[2][:,3] = u.x[2][1]*HF.cl_field.x[2][2] - u.x[2][2]*HF.cl_field.x[2][1]
end


############# Composite Approximation

struct CompositeRHSFunction{RHSFunction1, RHSFunction2} <: AbstractRHSFunction
    RHS1::RHSFunction1
    RHS2::RHSFunction2
    q1tocl2::SharedMatrix{Float64}
    cl2toq1::SharedMatrix{Float64}
    cl1tocl2::SharedMatrix{Float64}
    cl2tocl1::SharedMatrix{Float64}
end

function build_RHS_function(A::CompositeApproximation{Approx1,Approx2}) where {Approx1<:HybridApprox, Approx2<:PureClassicalApprox}
    RHS1 = build_RHS_function(A.A1)
    RHS2 = build_RHS_function(A.A2, A.gamma^2)
    q1tocl2, cl2toq1, cl1tocl2, cl2tocl1 = build_InterLattice_Couplings(A, size(RHS1.edgeSpinOperators)[1])
    CompositeRHSFunction(RHS1,RHS2, q1tocl2, cl2toq1, cl1tocl2, cl2tocl1)
end

function (CF::CompositeRHSFunction)(t, u::T, du::T) where T<:ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},1},VectorOfArray{Float64,2,Array{Array{Float64,1},1}},VectorOfArray{Float64,2,Array{Array{Float64,1},1}}}}
    ############## Unpacking
    HF = CF.RHS1
    PCF = CF.RHS2
    ############## Quantum averages
    qnorm = norm(u.x[1])
    @inbounds for sigma in 1:3
        for q in 1:size(HF.edgeSpinOperators,1)
            A_mul_B!(HF.q_temp_state, HF.edgeSpinOperators[q, sigma], u.x[1])
            HF.average_values[q, sigma] = real(dot(HF.q_temp_state, u.x[1]))/qnorm
        end
    end
    ############## Calculation of effective magnetic fields
    @inbounds for sigma in 1:3
        A_mul_B!(HF.Js[sigma], HF.cl2q, u.x[2][sigma], zero(Float64), HF.cl_field.x[1][sigma]) # quantum part
        fill!(HF.cl_field.x[2][sigma], HF.hs[sigma]) # init L1 fields
        fill!(PCF.cl_field[sigma], PCF.hs[sigma]) # init L2 fields
        A_mul_B!(HF.Js[sigma], HF.IM, u.x[2][sigma], one(Float64), HF.cl_field.x[2][sigma]) # cl2cl L1 fields
        A_mul_B!(PCF.Js[sigma], PCF.IM, u.x[3][sigma], one(Float64), PCF.cl_field[sigma]) # cl2cl L2 fields
        A_mul_B!(HF.Js[sigma], HF.q2cl, HF.average_values[sigma], one(Float64), HF.cl_field.x[2][sigma]) #cl21 L1 fields#
    end
    #### Cross fields
    A_mul_B!(PCF.Js[3], CF.cl2toq1, u.x[3][3], one(Float64), HF.cl_field.x[1][3])
    A_mul_B!(PCF.Js[3], CF.cl2tocl1, u.x[3][3], one(Float64), HF.cl_field.x[2][3])
    A_mul_B!(HF.Js[3], CF.q1tocl2, HF.average_values[3], one(Float64), PCF.cl_field[3])
    A_mul_B!(HF.Js[3], CF.cl1tocl2, u.x[2][3], one(Float64), PCF.cl_field[3])
    ############## Apply fields
    A_mul_B!(du.x[1], HF.H, u.x[1])
    @inbounds for sigma in 1:3
        for q in 1:size(HF.edgeSpinOperators, 1)
            A_mul_B!(HF.cl_field.x[1][q,sigma], HF.edgeSpinOperators[q, sigma], u.x[1], one(Complex{Float64}), du.x[1])
        end
    end
    du.x[1] .*= -1im
    @. du.x[2][:,1] = u.x[2][2]*HF.cl_field.x[2][3] - u.x[2][3]*HF.cl_field.x[2][2]
    @. du.x[2][:,2] = u.x[2][3]*HF.cl_field.x[2][1] - u.x[2][1]*HF.cl_field.x[2][3]
    @. du.x[2][:,3] = u.x[2][1]*HF.cl_field.x[2][2] - u.x[2][2]*HF.cl_field.x[2][1]
    @. du.x[3][:,1] = u.x[3][2]*PCF.cl_field[3] - u.x[3][3]*PCF.cl_field[2]
    @. du.x[3][:,2] = u.x[3][3]*PCF.cl_field[1] - u.x[3][1]*PCF.cl_field[3]
    @. du.x[3][:,3] = u.x[3][1]*PCF.cl_field[2] - u.x[3][2]*PCF.cl_field[1]
end
