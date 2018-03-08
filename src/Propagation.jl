struct LProblem{Approx, SCB, PP}
    A::Approx
    cb::SCB
    prob::PP
    rules::ConvRules
    cf_vals::CFVals
    Tmax::Float64
    tstep::Float64
    delimiter::Int64
end

function build_problem(L::Lattice, M::AbstractModel, args...; Tmax::Float64 = 10.0, tstep::Float64 = 2.0^-7, delimiter::Int64 = 10, links = [(:all,:all)], axes = [:x, :y, :z])
    A = build_Approximation(M, L, args...)
    ARHS = build_RHS_function(A)
    rules, OSET = set_CorrFuncs(A, links, axes)
    l = length(OSET.Observables)
    tspan = (0.0,Tmax*delimiter)
    t_means = collect(0.0:tstep:(Tmax*delimiter))
    t_cors = collect(0.0:tstep:Tmax)
    saved_values = SavedValues(t_means, [zeros(l) for i in eachindex(t_means)])
    u0 = randomState(A, OSET)
    prob = ODEProblem(ARHS, u0, tspan)
    cb = SavingCallback(OSET, saved_values, u0; saveat = t_means)
    #cb = SavingCallback(OSET, saved_values; save_everystep=true)
    cf_vals = CFVals(rules, t_cors)
    return LProblem(A, cb, prob, rules, cf_vals, Tmax, tstep, delimiter)
end

struct Logger{T1,T2} end
struct Saver{T}
    filename::String
end
Saver() = Saver{:no}("")
Saver(filename::String) = Saver{:file}(filename)

@inline save_data(s::Saver{:no}, cf_vals::CFVals) = nothing
@inline function save_data(s::Saver{:file}, cf_vals::CFVals)
    jldopen(s.filename, "w") do file
        for i in Base.OneTo(length(cf_vals.str_vec))
            write(file, cf_vals.str_vec[i], cf_vals.data[i])
        end
        write(file, "ts", cf_vals.ts)
    end
end
function load_data(filename::AbstractString)
    jdata = load(filename)
    keys_ar = (jdata |> keys |> collect)
    deleteat!(keys_ar, findfirst(keys_ar, "ts"))
    return CFVals(keys_ar, jdata["ts"], VectorOfArray([jdata[key] for key in keys_ar]))
end
function load_data!(cf_vals::CFVals, filename::AbstractString)
    jdata = load(filename)
    keys_ar = (jdata |> keys |> collect)
    deleteat!(keys_ar, findfirst(keys_ar, "ts"))
    for key in keys_ar
        i = findfirst(cf_vals.str_vec, key)
        cf_vals.data[:,i] .= jdata[key]
    end
end
@inline load_data!(cf_vals::CFVals, saver::Saver{:file}) = load_data!(cf_vals, saver.filename)

abstract type AbstractLog end
struct CMDLog <: AbstractLog end

@inline output(log_o::CMDLog, args...) = (print(args...);print("\n"))
@inline Base.close(log_o::CMDLog) = nothing

struct FileLog
    status::Int64
    offset::Float64
    filename::String
end

function FileLog(filename::AbstractString)
    if isfile(filename)
        io = open(filename, "r+")
        lc = countlines(io)
        if lc>0
            skip(io, -64)
            line = split(readline(io))
            status = parse(Int64, line[1])
            offset = status*parse(Float64, line[3])
            seekend(io)
        else
            status = 0
            offset = 0.0
        end
    else
        io = open(filename, "w")
        status = 0
        offset = 0.0
    end
    close(io)
    FileLog(status, offset, filename)
end

@inline output(log_o::FileLog, args...) = (io = open(log_o.filename, "a"); print(io, args...);print(io, "\n"); close(io))

@inline get_status(log_o::CMDLog) = 0
@inline get_status(log_o::FileLog) = log_o.status
@inline get_offset(log_o::CMDLog) = 0.0
@inline get_offset(log_o::FileLog) = log_o.offset

@inline get_string(nTrials::Int64, thread::Int64, alg::OrdinaryDiffEqAlgorithm) = @sprintf("nTr%d Alg(%s) thr%d", nTrials, alg, thread)

function set_Logging(logger::Logger{:no, :cmd}, A::AbstractApproximation, filename::NTuple{2,String})
    save_o = Saver()
    log_o = CMDLog()
    return log_o, save_o
end
function set_Logging(logger::Logger{:local, :cmd}, A::AbstractApproximation, filename::NTuple{2,String})
    path = "./Data/" * get_string(A.L) * "/" * get_string(A) * "/" * filename[1]
    mkpath(path)
    save_o = Saver(path * filename[2] * ".jld")
    log_o = CMDLog()
    return log_o, save_o
end
function set_Logging(logger::Logger{:local, :file}, A::AbstractApproximation, filename::NTuple{2,String})
    path = "./Data/" * get_string(A.L) * "/" * get_string(A) * "/" * filename[1]
    mkpath(path)
    save_o = Saver(path * filename[2] * ".jld")
    log_o = FileLog(path * filename[2] * ".log")
    return log_o, save_o
end
function set_Logging(logger::Logger{:cluster, :file}, A::AbstractApproximation, filename::NTuple{2,String})
    path = ENV["WORK_DIR"] * "/Data/" * get_string(A.L) * "/" * get_string(A) * "/" * filename[1]
    mkpath(path)
    save_o = Saver(path * filename[2] * ".jld")
    log_o = FileLog(path * filename[2] * ".log")
    return log_o, save_o
end

function simulate(LP::LProblem, nTrials::Int64, alg::OrdinaryDiffEqAlgorithm, logger::Logger{T1,T2} = Logger{:no, :cmd}(), thread::Int64 = 1, interval::Int64 = 1, RNG::AbstractRNG=Base.GLOBAL_RNG; abstol_i = 1e-14, reltol_i = 1e-6, kwargs...) where {T1,T2}
    assert(nTrials > 0)
    saved_values = LP.cb.affect!.saved_values
    rules = LP.rules
    cf_vals = LP.cf_vals
    A = LP.A
    OSET = LP.cb.affect!.save_func
    filename = (@sprintf("Tm%.2f dt%.3f del%d Obs%s/", LP.Tmax, LP.tstep, LP.delimiter, get_string(LP.rules.str_vec)), get_string(nTrials, thread, alg))
    log_o, save_o = set_Logging(logger, A, filename)
    iter = get_status(log_o)
    offset = get_offset(log_o)
    if iter > 0
        load_data!(cf_vals, save_o)
    end
    if iter == nTrials && (T1 == :local || T1 == :cluster)
        nTrials = 2*nTrials
        #close(log_o)
        filename = (@sprintf("Tm%.2f dt%.3f del%d Obs%s/", LP.Tmax, LP.tstep, LP.delimiter, get_string(LP.rules.str_vec)), get_string(nTrials, thread, alg))
        log_o, save_o = set_Logging(logger, A, filename)
    end
    iter += 1
    t1 = time()
    t0 = t1-offset
    integrator = init(LP.prob, alg; save_everystep = false, callback = LP.cb, abstol = abstol_i, reltol = reltol_i, kwargs...)
    solve!(integrator)
    calculateCorrFunc!(A, saved_values, rules, cf_vals)
    save_data(save_o, cf_vals)
    t2 = time()
    output(log_o, @sprintf("%10d %20.5f %20.5f %10d", iter, t2-t1, (t2-t0)/iter, nTrials - iter))
    iter += 1
    while iter <= nTrials
        t1 = time()
        reinit!(integrator, randomState(A,OSET,RNG))
        solve!(integrator)
        calculateCorrFunc!(A, saved_values, rules, cf_vals)
        t2 = time()
        if iter%interval == 0
            save_data(save_o, cf_vals)
            output(log_o, @sprintf("%10d %20.5f %20.5f %10d", iter, t2-t1, (t2-t0)/iter, nTrials - iter))
        end
        iter += 1
    end
    save_data(save_o, cf_vals)
    output(log_o, @sprintf("%10d %20.5f %20.5f %10d", nTrials, t2-t0, (t2-t0)/(iter-1), 0))
    #close(log_o)
    return cf_vals
end

export Logger


function parallel_simulate(LP::LProblem, nTrials::Int64, alg::OrdinaryDiffEqAlgorithm, logger::Logger{T1,T2} = Logger{:no, :cmd}(), interval::Int64 = 1, offset::Int64 = 0; kwargs...) where {T1,T2}
    RNGarray = initRNG(nprocs())
    @sync begin
        for i in workers()
            @spawnat i simulate(LP,nTrials, alg, logger, i+offset, interval, RNGarray[i]; kwargs...)
        end
        @spawnat 1 simulate(LP,nTrials, alg, logger, 1+offset, interval, RNGarray[1]; kwargs...)
    end
end

export parallel_simulate
