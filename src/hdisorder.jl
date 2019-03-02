function dh_simulation(LP::LProblem{T}, p::Float64, nTrials::Int64, alg::OrdinaryDiffEqAlgorithm, logger::Logger{T1,T2} = Logger{:no, :cmd}(), interval::Int64 = 1, offset::Int64 = 0; links = (:all,:all), axes = [:x, :y, :z], kwargs...) where {T<:HybridApprox,T1,T2}
    assert(nTrials > 0)
    RNGarray = initRNG(nprocs()+1)
    path_part = @sprintf("Dis Tm%.2f dt%.3f del%d Obs%s Pr%.3f/", LP.Tmax, LP.tstep, LP.delimiter, get_string(LP.rules.str_vec), p)
    filenames = Vector{String}()
    A = HybridApprox(LP.A.L, copy(LP.A.q_spins), LP.A.name)
    l = LP.cb.affect!.save_func.len
    tspan = (0.0, LP.Tmax*LP.delimiter)
    t_means = collect(0.0:LP.tstep:(LP.Tmax*LP.delimiter))
    saved_values = SavedValues(t_means, [zeros(l) for i in eachindex(t_means)])
    for thread = (offset+1):(offset+nprocs())
        push!(filenames, get_string(nTrials, thread, alg))
    end
    log_o, save_os = set_Logging(logger, LP.A, path_part, filenames)
    cf_vals0 = LP.cf_vals
    @sync for i in workers()
        @spawnat i global cf_vals = cf_vals0
    end
    iter = get_status(log_o)
    offset = get_offset(log_o)
    if iter > 0
        load_data!(cf_vals0, save_os[1])
        for i in workers()
            @spawnat i load_data!(cf_vals, save_os[i])
        end
    end
    t0 = time()-offset
    while iter<nTrials
        t1 = time()
        iter += 1
        q_spins = randsubseq(RNGarray[end], LP.A.q_spins, 1-p)
        A.q_spins = q_spins
        A.q_num = length(q_spins)
        ARHS = build_RHS_function(A)
        rules, OSET = set_CorrFuncs(A, links, axes)
        @sync begin
            for i in workers()
                @spawnat i dh_run(iter, A, ARHS, OSET, rules, t_means, saved_values, tspan, p, cf_vals, alg, save_os[i], RNGarray[i]; kwargs...)
            end
            @spawnat 1 dh_run(iter, A, ARHS, OSET, rules, t_means, saved_values, tspan, p, cf_vals0, alg, save_os[1], RNGarray[1]; kwargs...)
        end
        t2 = time()
        output(log_o, @sprintf("%10d %20.5f %20.5f %10d", iter, t2-t1, (t2-t0)/iter, nTrials - iter))
    end
end

function dh_simulation(A::CompositeApproximation{T1,T2}, p::Float64, nTrials::Int64, alg::OrdinaryDiffEqAlgorithm, logger::Logger{T3,T4} = Logger{:cluster, :file}, interval::Int64 = 1, offset1::Int64 = 0, Tmax::Float64 = 3.5, tstep::Float64 = 2.0^-7, delimiter::Int64 = 10, links = (:all,:allq), axes = [:x]; kwargs...) where {T1<:HybridApprox, T2<:PureClassicalApprox, T3,T4}
    A_temp = CompositeApproximation(HybridApprox(A.A1.L, copy(A.A1.q_spins), A.A1.name), A.A2, A.gamma)
    rules, OSET = set_CorrFuncs(A.A1, links, axes)
    l = length(OSET.Observables)
    tspan = (0.0, Tmax*delimiter)
    t_means = collect(0.0:tstep:(Tmax*delimiter))
    t_cors = collect(0.0:tstep:Tmax)
    saved_values = SavedValues(t_means, [zeros(l) for i in eachindex(t_means)])
    ##############################
    assert(nTrials > 0)
    RNGarray = initRNG(nprocs()+1)
    path_part = @sprintf("Dis Tm%.2f dt%.3f del%d Obs%s Pr%.3f/", Tmax, tstep, delimiter, get_string(rules.str_vec), p)
    filenames = Vector{String}()
    for thread = (offset1+1):(offset1+nprocs())
        push!(filenames, get_string(nTrials, thread, alg))
    end
    log_o, save_os = set_Logging(logger, A, path_part, filenames)
    cf_vals0 = CFVals(rules, t_cors)
    @sync for i in workers()
        @spawnat i global cf_vals = cf_vals0
    end
    iter = get_status(log_o)
    offset = get_offset(log_o)
    if iter > 0
        load_data!(cf_vals0, save_os[1])
        for i in workers()
            @spawnat i load_data!(cf_vals, save_os[i])
        end
    end
    t0 = time()-offset
    ##############################
    while iter<nTrials
        t1 = time()
        iter += 1
        q_spins = randsubseq(RNGarray[end], A.A1.q_spins, 1-p)
        A_temp.A1.q_spins = q_spins
        A_temp.A1.q_num = length(q_spins)
        print(length(q_spins), "\n")
        ARHS = build_RHS_function(A_temp)
        rules, OSET = set_CorrFuncs(A_temp.A1, links, axes)
        @sync begin
            for i in workers()
                @spawnat i dh_run(iter, A_temp, ARHS, OSET, rules, t_means, saved_values, tspan, p, cf_vals, alg, save_os[i], RNGarray[i]; kwargs...)
            end
            @spawnat 1 dh_run(iter, A_temp, ARHS, OSET, rules, t_means, saved_values, tspan, p, cf_vals0, alg, save_os[1], RNGarray[1]; kwargs...)
        end
        t2 = time()
        output(log_o, @sprintf("%10d %20.5f %20.5f %10d", iter, t2-t1, (t2-t0)/iter, nTrials - iter))
    end
end



function dh_run(iter::Int64, A::abstractApproximation, ARHS::AbstractRHSFunction, OSET::ObservablesSet, rules::ConvRules, t_means::Vector{Float64}, saved_values::SavedValues, tspan::NTuple{2, Float64}, p::Float64, cf_vals::CFVals, alg::OrdinaryDiffEqAlgorithm, save_o::Saver{:file}, RNG::AbstractRNG; abstol_i = 1e-14, reltol_i = 1e-6, kwargs...)
    u0 = randomDState(A, p, RNG)
    prob = ODEProblem(ARHS, u0, tspan)
    cb = SavingCallback(OSET, saved_values, u0; saveat = t_means)
    solve(prob, alg; save_everystep = false, callback = cb, abstol = abstol_i, reltol = reltol_i, kwargs...)
    calculateCorrFunc!(A, saved_values, rules, cf_vals, iter)
    save_data(save_o, cf_vals, iter)
end

function randomDState(A::HybridApprox, p::Float64, RNG::AbstractRNG = Base.GLOBAL_RNG)
    Dh = 2^A.q_num
    cl_state = random_cl_state(A.cl_num, RNG)
    discarded = randsubseq(RNG, collect(1:A.cl_num), p)
    for i in discarded
        cl_state[i,:] .= 0.0
    end
    return ArrayPartition(random_q_state(Dh, RNG), cl_state)
end

function randomDState(A::CompositeApproximation{T1,T2}, p::Float64, RNG::AbstractRNG = Base.GLOBAL_RNG) where {T1<:HybridApprox, T2<:PureClassicalApprox}
    Dh = 2^A.A1.q_num
    cl_state = random_cl_state(A.A1.cl_num, RNG)
    discarded = randsubseq(RNG, collect(1:A.A1.cl_num), p)
    for i in discarded
        cl_state[i,:] .= 0.0
    end
    ArrayPartition(random_q_state(Dh, RNG), cl_state, randomState(A.A2, RNG))
end

export dh_simulation
