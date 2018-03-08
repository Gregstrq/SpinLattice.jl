@inline DiffEqBase.ODE_DEFAULT_NORM(u::RecursiveArrayTools.ArrayPartition) = sqrt(sum(DiffEqBase.ODE_DEFAULT_NORM, u.x)/length(u.x))
@inline DiffEqBase.ODE_DEFAULT_NORM(u::RecursiveArrayTools.AbstractVectorOfArray) = sqrt(sum(DiffEqBase.ODE_DEFAULT_NORM, u)/length(u))

@inline Base.broadcast(/, VA::VectorOfArray, b::Number) = VectorOfArray(broadcast(/, VA.u, b))

function random_q_state(Dh::Int64, RNG::AbstractRNG = Base.GLOBAL_RNG)
    threshold = (2.0^-7)::Float64
    q_state = Vector{Complex{Float64}}(Dh)
    amps = rand(RNG, Dh) .* (1-threshold) .+ threshold
    @. amps = sqrt(-log(amps)/Dh)
    amps .= amps./norm(amps)
    phases = rand(RNG, Float64, Dh)
    @. q_state = amps*exp(2*pi*1im*phases)
    return q_state
end

function random_cl_state(cl_num, RNG::AbstractRNG = Base.GLOBAL_RNG)
    zs = rand(RNG, cl_num) .* 2.0 .- 1.0
    rs = sqrt.(1.0 .- zs.^2)
    angles = rand(RNG, cl_num) .* 2.0 .* pi
    cl_state = VectorOfArray([rs.*cos.(angles), rs.*sin.(angles), zs])
    cl_state .*= sqrt(3)/2.0
    return cl_state
end

function randomState(A::Approx, RNG::AbstractRNG = Base.GLOBAL_RNG) where {Approx<:ExactApprox}
    Dh = 2^A.L.tot_spin_num
    return random_q_state(Dh, RNG)
end
function randomState(A::ExactApprox, OSET::ObservablesSet, RNG::AbstractRNG = Base.GLOBAL_RNG)
    Dh = get_Dh(A)
    states = Vector{Vector{Complex{Float64}}}()
    push!(states, random_q_state(Dh, RNG))
    for i = 1:length(OSET.Observables)
        push!(states, similar(states[1]))
        A_mul_B!(states[end], OSET.Observables[i].QO.O, states[1])
    end
    return VectorOfArray(states)
end
function randomState(A::Approx, RNG::AbstractRNG = Base.GLOBAL_RNG) where {Approx<:ClusteredApprox}
    Dh = 2^(A.cluster_cell_num*A.L.cell_size)
    return VectorOfArray([random_q_state(Dh, RNG) for i in 1:A.cluster_num])
end
function randomState(A::Approx, RNG::AbstractRNG = Base.GLOBAL_RNG) where {Approx<:PureClassicalApprox}
    return random_cl_state(A.L.tot_spin_num, RNG)
end
function randomState(A::Approx, RNG::AbstractRNG = Base.GLOBAL_RNG) where {Approx<:HybridApprox}
    Dh = 2^A.q_num
    return ArrayPartition(random_q_state(Dh, RNG), random_cl_state(A.cl_num, RNG))
end

@inline randomState(A::AbstractApproximation, OSET::ObservablesSet, RNG::AbstractRNG = Base.GLOBAL_RNG) = randomState(A, RNG)

function initRNG(d::Int)
    return randjump(Base.GLOBAL_RNG, d)
end
