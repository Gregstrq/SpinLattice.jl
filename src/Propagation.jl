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

@inline save_data(s::Saver{:no}, cf_vals::CFVals, iter = 0) = nothing
@inline save_data(s::Saver{:file}, cf_vals::CFVals, iter) = save_data(s.filename, cf_vals, iter)
function save_data(filename::AbstractString, cf_vals::CFVals, iter)
    jldopen(filename, "w") do file
        gdata = g_create(file, "data")
        gerrors = g_create(file, "errors")
        for i in Base.OneTo(length(cf_vals.str_vec))
            write(gdata, cf_vals.str_vec[i], cf_vals.data[i])
            write(gerrors, cf_vals.str_vec[i], cf_vals.errors[i])
        end
        write(file, "ts", cf_vals.ts)
        write(file, "str_vec", cf_vals.str_vec)
        write(file, "nTrials", iter)
    end
end
function load_data(filename::AbstractString)
    jdata = load(filename)
    data = VectorOfArray([jdata["data"][key] for key in jdata["str_vec"]])
    errors = VectorOfArray([jdata["errors"][key] for key in jdata["str_vec"]])
    return CFVals(jdata["str_vec"], jdata["ts"], data, errors), jdata["nTrials"]
end
function load_data!(cf_vals::CFVals, filename::AbstractString)
    jdata = load(filename)
    for i in eachindex(jdata["str_vec"])
        key = jdata["str_vec"][i]
        cf_vals.data[:,i] = jdata["data"][key]
        cf_vals.errors[:,i] = jdata["errors"][key]
    end
    return jdata["nTrials"]
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
function set_Logging(logger::Logger{:local, :cmd}, A::AbstractApproximation, path_part::String, filenames::Vector{String})
    path = "./Data/" * get_string(A.L) * "/" * get_string(A) * "/" * path_part
    mkpath(path)
    log_o = CMDLog()
    save_os = Vector{Saver{:file}}()
    for filename in filenames
        push!(save_os, Saver(path * filename * ".jld"))
    end
    return log_o, save_os
end
function set_Logging(logger::Logger{:local, :file}, A::AbstractApproximation, filename::NTuple{2,String})
    path = "./Data/" * get_string(A.L) * "/" * get_string(A) * "/" * filename[1]
    mkpath(path)
    save_o = Saver(path * filename[2] * ".jld")
    log_o = FileLog(path * filename[2] * ".log")
    return log_o, save_o
end
function set_Logging(logger::Logger{:local, :file}, A::AbstractApproximation, path_part::String, filenames::Vector{String})
    path = "./Data/" * get_string(A.L) * "/" * get_string(A) * "/" * path_part
    mkpath(path)
    log_o = FileLog(path * filenames[1] * ".log")
    save_os = Vector{Saver{:file}}()
    for filename in filenames
        push!(save_os, Saver(path * filename * ".jld"))
    end
    return log_o, save_os
end
function set_Logging(logger::Logger{:cluster, :file}, A::AbstractApproximation, filename::NTuple{2,String})
    path = ENV["WORK_DIR"] * "/Data/" * get_string(A.L) * "/" * get_string(A) * "/" * filename[1]
    mkpath(path)
    save_o = Saver(path * filename[2] * ".jld")
    log_o = FileLog(path * filename[2] * ".log")
    return log_o, save_o
end
function set_Logging(logger::Logger{:cluster, :file}, A::AbstractApproximation, path_part::String, filenames::Vector{String})
    path = ENV["WORK_DIR"] * "/Data/" * get_string(A.L) * "/" * get_string(A) * "/" * path_part
    mkpath(path)
    log_o = FileLog(path * filenames[1] * ".log")
    save_os = Vector{Saver{:file}}()
    for filename in filenames
        push!(save_os, Saver(path * filename * ".jld"))
    end
    return log_o, save_os
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
    calculateCorrFunc!(A, saved_values, rules, cf_vals, iter)
    save_data(save_o, cf_vals, iter)
    t2 = time()
    output(log_o, @sprintf("%10d %20.5f %20.5f %10d", iter, t2-t1, (t2-t0)/iter, nTrials - iter))
    iter += 1
    while iter <= nTrials
        t1 = time()
        reinit!(integrator, randomState(A,OSET,RNG))
        solve!(integrator)
        calculateCorrFunc!(A, saved_values, rules, cf_vals, iter)
        t2 = time()
        if iter%interval == 0
            save_data(save_o, cf_vals, iter)
            output(log_o, @sprintf("%10d %20.5f %20.5f %10d", iter, t2-t1, (t2-t0)/iter, nTrials - iter))
        end
        iter += 1
    end
    save_data(save_o, cf_vals, iter)
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

function parse_dir(dirname::AbstractString = pwd())
    corfs = Vector{CFVals}()
    Ns = Vector{Int64}()
    for filename in readdir(dirname)
        file = joinpath(dirname, filename)
        if filename != "aggregate.jld" && isfile(file) && file[end-3:end] == ".jld"
            corf, Ntr = load_data(file)
            push!(corfs, corf)
            push!(Ns, Ntr)
        end
    end
    return corfs, Ns
end

function parse_dir(dirlist::Tuple{AbstractString,Vararg{AbstractString}})
    corfs = Vector{CFVals}()
    Ns = Vector{Int64}()
    for dirname in dirlist
        for filename in readdir(dirname)
            file = joinpath(dirname, filename)
            if filename != "aggregate.jld" && isfile(file) && file[end-3:end] == ".jld"
                corf, Ntr = load_data(file)
                push!(corfs, corf)
                push!(Ns, Ntr)
            end
        end
    end
    return corfs, Ns
end

function aggregate(dirname::AbstractString = pwd())
    corfs, Ns = parse_dir(dirname)
    return aggregate(corfs, Ns, dirname)
end

function aggregate(dirlist::Tuple{AbstractString,Vararg{AbstractString}}, destdir::AbstractString = pwd())
    corfs, Ns = parse_dir(dirlist)
    mkpath(destdir)
    return aggregate(corfs, Ns, destdir)
end


function aggregate(corfs::Vector{CFVals}, Ns::Vector{Int64}, dirname::AbstractString = pwd(), agrname = "aggregate.jld")
    aggregate = zeros(first(corfs))
    N_total = sum(Ns)
    for i in eachindex(corfs)
        aggregate.data   .= aggregate.data  .+ corfs[i].data .* Ns[i]
        aggregate.errors .= aggregate.errors .+ corfs[i].errors
    end
    for j = 2:length(corfs), i = 1:j
        coef = Ns[i]*Ns[j]/N_total
        aggregate.errors .= aggregate.errors .+ (corfs[i].data - corfs[j].data).^2 .* coef
    end
    correction = N_total/(N_total - 1)
    for i in 1:length(aggregate.data)
        coef = 1/aggregate.data[1,i]
        aggregate.data[:,i]   .= aggregate.data[:,i] .* coef
        aggregate.errors[:,i] .= sqrt.(aggregate.errors[:,i] .* correction) .* coef
    end
    save_data(joinpath(dirname, agrname), aggregate, N_total)
    return aggregate, N_total
end

function separate(Ns::Vector{Int64})
    i = length(Ns)
    lm = div(i,2)
    l1 = Vector{Int64}()
    l2 = Vector{Int64}()
    sum1 = 0
    sum2 = 0
    while (length(l1)<lm) && (length(l2)<lm)
        el = Ns[i]
        if sum1>sum2
            sum2 += el
            push!(l2, i)
        else
            sum1 += el
            push!(l1, i)
        end
        i -=1
    end
    if sum1 > sum2
        append!(l2, 1:i)
    else
        append!(l1, 1:i)
    end
    return l1, l2
end

function aggregate2(dirname::AbstractString = pwd())
    corfs, Ns = parse_dir(dirname)
    l1, l2 = separate(Ns)
    agr1, Ntot1 = aggregate(corfs[l1], Ns[l1], dirname, "aggregate1.jld")
    agr2, Ntot2 = aggregate(corfs[l2], Ns[l2], dirname, "aggregate2.jld")
    return agr1, Ntot1, agr2, Ntot2
end


function aggregate2(dirlist::Tuple{AbstractString,Vararg{AbstractString}}, destdir::AbstractString = pwd())
    corfs, Ns = parse_dir(dirlist)
    l1, l2 = separate(Ns)
    mkpath(destdir)
    agr1, Ntot1 = aggregate(corfs[l1], Ns[l1], destdir, "aggregate1.jld")
    agr2, Ntot2 = aggregate(corfs[l2], Ns[l2], destdir, "aggregate2.jld")
    return agr1, Ntot1, agr2, Ntot2
end
