abstract type AbstractObservable end
abstract type AbstractSimpleObservable <: AbstractObservable end
abstract type AbstractCompositeObservable <: AbstractObservable end

const axis_dict = Dict{Symbol,Int64}(:x=>1, :y=>2, :z=>3)

struct QuantumObservable{TO<:AbstractSparseMatrix} <: AbstractSimpleObservable
    O::TO
    temp_vector::Vector{Complex{Float64}}
end
function QuantumObservable(Dh::Int64, q_spins::Vector{Int64}, sigma::Int64)
    assert(sigma in 1:3)
    for i in eachindex(q_spins)
        assert(2^q_spins[i]<=Dh)
    end
    QuantumObservable(build_Spin_Operator(Dh, q_spins, sigma), Vector{Complex{Float64}}(Dh))
end

struct ClassicalObservable <: AbstractSimpleObservable
    sigma::Int64
    cl_spins::Vector{Int64}
end

struct EmptyObservable <: AbstractSimpleObservable end

################

function (QO::QuantumObservable)(v::Vector{Complex{Float64}})
    A_mul_B!(QO.temp_vector, QO.O, v)
    return real(dot(v, QO.temp_vector))
end
function (QO::QuantumObservable)(v1::Vector{Complex{Float64}}, v2::Vector{Complex{Float64}})
    A_mul_B!(QO.temp_vector, QO.O, v2)
    return real(dot(v1, QO.temp_vector))
end
function (CO::ClassicalObservable)(v::VectorOfArray{Float64, 2, Vector{Vector{Float64}}})
    s = zero(Float64)
    sigma = CO.sigma
    for cl_spin in CO.cl_spins
        s += v[cl_spin, sigma]
    end
    return s
end
function (CO::ClassicalObservable)(v::VectorOfArray{T, 2, Vector{Vector{T}}}) where T<:Complex{Float64}
    s = zero(Float64)
    sigma = CO.sigma
    for cl_spin in CO.cl_spins
        s += real(v[cl_spin, sigma])
    end
    return s
end

@inline (EO::EmptyObservable)(v) = 0.0

################

convert_axis(axis) = error("no method for such input type of axis")
convert_axis(axis::Symbol) = axis_dict[axis]

struct ExactObservable{T1,T2,TO} <: AbstractCompositeObservable
    axis::T1
    name::T2
    QO::QuantumObservable{TO}
    id::Int64
    function ExactObservable(A::ExactApprox, spins::Vector{NTuple{2,Int64}}, axis::T1, name::T2, id::Int64) where{T1,T2}
        QO = QuantumObservable(get_Dh(A), get_positions(A, spins), convert_axis(axis))
                               new{T1,T2,typeof(QO.O)}(axis, name, QO, id)
    end
end
@inline ExactObservable(A, spins, axis, name) = ExactObservable(A, spins, axis, name, axis_dict[axis]+1)

struct ClusteredObservable{T1,T2,TO} <: AbstractCompositeObservable
    axis::T1
    name::T2
    QO::QuantumObservable{TO}
    function ClusteredObservable(A::ClusteredApprox, spins::Vector{NTuple{2,Int64}}, axis::T1, name::T2) where{T1,T2}
        spins_c = Set{Int64}()
        for pos in get_positions(A, spins)
            push!(spins_c, pos[1])
        end
        spins_cc = Int64[spins_c...]
        sort!(spins_cc)
        QO = QuantumObservable(get_Dh(A), spins_cc, convert_axis(axis))
        new{T1,T2,typeof(QO.O)}(axis, name, QO)
    end
end

struct PureClassicalObservable{T1,T2} <: AbstractCompositeObservable
    axis::T1
    name::T2
    CO::ClassicalObservable
    function PureClassicalObservable(A::PureClassicalApprox, spins::Vector{NTuple{2,Int64}}, axis::T1, name::T2) where{T1,T2}
        new{T1,T2}(axis, name, ClassicalObservable(convert_axis(axis), get_positions(A, spins)))
    end
end

struct HybridObservable{T1,T2, O1<:Union{QuantumObservable, EmptyObservable},O2<:Union{ClassicalObservable, EmptyObservable}} <: AbstractCompositeObservable
    axis::T1
    name::T2
    c::Float64
    QO::O1
    CO::O2
    function HybridObservable(A::HybridApprox, spins::Vector{NTuple{2,Int64}}, axis::T1, name::T2) where{T1,T2}
        q_poss, cl_poss = get_positions(A, spins)
        if !isempty(q_poss)
            QO = QuantumObservable(get_Dh(A), q_poss, convert_axis(axis))
        else
            QO = EmptyObservable()
        end
        if !isempty(cl_poss)
            CO = ClassicalObservable(convert_axis(axis), cl_poss)
        else
            CO = EmptyObservable()
        end
        c = sqrt(get_Dh(A)+1)/2
        new{T1,T2, typeof(QO), typeof(CO)}(axis, name, c, QO, CO)
    end
end


########################################

@inline (EO::ExactObservable)(v::Vector{Complex{Float64}}) = EO.QO(v)
@inline (EO::ExactObservable)(voa::VectorOfArray{Complex{Float64},2,Array{Array{Complex{Float64},1},1}}) = EO.QO(voa[1],voa[EO.id])

@inline (PCO::PureClassicalObservable)(v::VectorOfArray{Float64, 2, Vector{Vector{Float64}}}) = PCO.CO(v)
function (CLO::ClusteredObservable)(v::VectorOfArray{Complex{Float64}, 2, Vector{Vector{Complex{Float64}}}})
    s = zero(Float64)
    for i in eachindex(v)
        @inbounds s = s + CLO.QO(v[i])
    end
    return s
end
@inline function (HO::HybridObservable)(v::ArrayPartition{Complex{Float64}, Tuple{Vector{Complex{Float64}}, VectorOfArray{T, 2, Vector{Vector{T}}}}}) where T<:Union{Float64,Complex{Float64}}
    return HO.c*HO.QO(v.x[1]) + HO.CO(v.x[2])
end

@inline function (HO::HybridObservable)(v::ArrayPartition{Complex{Float64},Tuple{Array{Complex{Float64},1},VectorOfArray{Float64,2,Array{Array{Float64,1},1}},VectorOfArray{Float64,2,Array{Array{Float64,1},1}}}})
    return HO.c*HO.QO(v.x[1]) + HO.CO(v.x[2])
end

########################################

@inline build_Observable(A::ExactApprox, spins::Vector{NTuple{2,Int64}}, axis, name) = ExactObservable(A, spins, axis, name)
@inline build_Observable(A::ExactApprox, spins::Vector{NTuple{2,Int64}}, axis, name, id::Int64) = ExactObservable(A, spins, axis, name, id)
@inline build_Observable(A::ClusteredApprox, spins::Vector{NTuple{2,Int64}}, axis, name) = ClusteredObservable(A, spins, axis, name)
@inline build_Observable(A::PureClassicalApprox, spins::Vector{NTuple{2,Int64}}, axis, name) = PureClassicalObservable(A, spins, axis, name)
@inline build_Observable(A::HybridApprox, spins::Vector{NTuple{2,Int64}}, axis, name) = HybridObservable(A, spins, axis, name)

@inline build_Observable(A::AbstractApproximation, name::Symbol, axis::Symbol) = build_Observable(A, get_spins_by_name(A, name), axis, name)
@inline build_Observable(A::AbstractApproximation, s_array::SpinArray, axis::Symbol) = build_Observable(A, s_array.spins, axis, s_array.name)

@inline build_Observable(A::ExactApprox, name::Symbol, axis::Symbol, id::Int64) = build_Observable(A, get_spins_by_name(A, name), axis, name, id)
@inline build_Observable(A::ExactApprox, s_array::SpinArray, axis::Symbol, id::Int64) = build_Observable(A, s_array.spins, axis, s_array.name, id)

########################################

struct ObservablesSet{T<:Tuple} <: AbstractObservable
    Observables::T
    len::Int64
    ObservablesSet(O_tuple::T) where {T} = new{T}(O_tuple, length(O_tuple))
end
ObservablesSet(O::AbstractCompositeObservable) = ObservablesSet((O,))

@inline ObservablesSet(obs...) = ObservablesSet(obs)
@inline function (OSET::ObservablesSet)(t, v, integrator)
    data = Vector{Float64}(OSET.len)
    for i in eachindex(OSET.Observables)
        data[i] = OSET.Observables[i](v)
    end
    return data
end

@inline return_type(O::AbstractCompositeObservable) = Float64
@inline return_type(O::ObservablesSet) = Vector{Float64}

#########################################
#Extra machinery to define couplings for calculating correlation functions

struct ConvRules
    str_vec::Vector{String}
    rules::Vector{NTuple{2,Int64}}
end

@inline get_rules(cfrules::ConvRules) = cfrules.rules

#For Exact Approximation I should make a special type of CFVals and dispatch on ExactApprox to choose this particular type.

struct CFVals
    str_vec::Vector{String}
    ts::Vector{Float64}
    data::VectorOfArray{Float64, 2, Vector{Vector{Float64}}}
    errors::VectorOfArray{Float64, 2, Vector{Vector{Float64}}}
end

import Base.similar
import Base.zeros

@inline similar(cf::CFVals) = CFVals(copy(cf.str_vec), copy(cf.ts), similar(cf.data), similar(cf.errors))

function zeros(cf::CFVals)
    temp = similar(cf)
    fill!(temp.data, 0.0)
    fill!(temp.errors, 0.0)
    return temp
end


function CFVals(rules::ConvRules, ts::Vector{Float64})
    data = VectorOfArray([zeros(length(ts)) for rule in get_rules(rules)])
    errors = VectorOfArray([zeros(length(ts)) for rule in get_rules(rules)])
    CFVals(rules.str_vec, ts, data, errors)
end

function make_oset(A::AbstractApproximation, links::NTuple{D, NTuple{2,Union{Symbol, SpinArray}}}, axes::Vector{Symbol}) where D
    os = Set{Tuple{Union{Symbol, SpinArray},Symbol}}()
    for axis in axes
        for link in links
            push!(os, (link[1], axis))
            push!(os, (link[2], axis))
        end
    end
    os_a = [os...]
    sort!(os_a)
    return os_a
end

function make_oset(A::ExactApprox, linkss::NTuple{D, NTuple{2,Union{Symbol, SpinArray}}}, axes::Vector{Symbol}) where D
    links = ((:all,:all),)
    os = Set{Tuple{Symbol,Symbol,Int64}}()
    for axis_i in eachindex(axes)
        for link in links
            push!(os, (link[1], axes[axis_i], axis_i+1))
        end
    end
    os_a = [os...]
    sort!(os_a)
    return os_a
end

function set_CorrFuncs(A::AbstractApproximation, links::NTuple{D, NTuple{2,Union{Symbol, SpinArray}}}, axes::Vector{Symbol}) where D
    os_a = make_oset(A, links, axes)
    print("$os_a\n")
    OSET = ObservablesSet([build_Observable(A, O...) for O in os_a]...)
    rules = Vector{NTuple{2,Int64}}()
    links_str = Vector{String}()
    for axis in axes
        for i in eachindex(links)
            push!(links_str, get_string((links[i], axis)))
            push!(rules, ( findfirst(os_a, (links[i][1], axis)) , findfirst(os_a, (links[i][2], axis)) ) )
        end
    end
    return ConvRules(links_str, rules), OSET
end
@inline set_CorrFuncs(A::AbstractApproximation, links::NTuple{2,Union{Symbol, SpinArray}} = (:all,:all), axes::Vector{Symbol} = [:x, :y,:z]) = set_CorrFuncs(A, tuple(links), axes)

function calculateCorrFunc!(A::ExactApprox, saved_values::SavedValues, rules::ConvRules, vals::CFVals)
    savevals = saved_values.saveval
    rule = get_rules(rules)
    for i in eachindex(rule)
        r1 = rule[i][1]
        @inbounds for j = 1:length(vals.data[1])
            vals.data[j,i] += savevals[j][i]
        end
    end
end

function calculateCorrFunc!(A::AbstractApproximation, saved_values::SavedValues, rules::ConvRules, vals::CFVals, iter)
     meanvals = Vector{Vector{Float64}}()
     savevals = saved_values.saveval
     l = length(savevals)
     nmeans = length(savevals[1])
     for m in Base.OneTo(nmeans)
         v = Vector{Float64}(l)
         for i in Base.OneTo(l)
             v[i] = savevals[i][m]
         end
         push!(meanvals, v)
     end
     #####
     data = vals.data
     errors = vals.errors
     rule = get_rules(rules)
     for i in eachindex(rule)
        r1 = rule[i][1]
        r2 = rule[i][2]
        if r1 == r2
            convolute_s!(meanvals[r1], data[i], errors[i], iter)
        else
            convolute_d!(meanvals[r1], meanvals[r2], data[i], errors[i], iter)
        end
     end
end

@inline calculateCorrFunc!(A::CompositeApproximation, saved_values::SavedValues, rules::ConvRules, vals::CFVals, iter) = calculateCorrFunc!(A.A1, saved_values, rules, vals, iter)

function convolute_d!(mean1::Vector{Float64}, mean2::Vector{Float64}, data::Vector{Float64})
    delta = 0
    stop = length(mean1)
    maxDelta = length(data) - 1
    @inbounds while delta <= maxDelta
        stmd = stop-delta
        val = dot(mean1, 1:stmd, mean2, (1+delta):stop)
        val += dot(mean2, 1:stmd, mean1, (1+delta):stop)
        data[delta+1] += val*0.5/stmd
        delta += 1
    end
end

function convolute_s!(mean::Vector{Float64}, data::Vector{Float64})
    delta = 0
    stop = length(mean)
    maxDelta = length(data) - 1
    @inbounds while delta <= maxDelta
        stmd = stop-delta
        data[delta+1] += dot(mean, 1:stmd, mean, (1+delta):stop)/stmd
        delta += 1
    end
end


function convolute_d!(mean1::Vector{Float64}, mean2::Vector{Float64}, data::Vector{Float64}, errors::Vector{Float64}, iter)
    delta = 1
    stop = length(mean1)
    maxDelta = length(data)
    @inbounds while delta <= maxDelta
        stmd = stop-delta+1
        val = dot(mean1, 1:stmd, mean2, delta:stop)
        val += dot(mean2, 1:stmd, mean1, delta:stop)
        val *= 0.5/stmd
        d1 = (val - data[delta])
        data[delta] += d1/iter
        d2 = (val - data[delta])
        errors[delta] += d1*d2
        delta += 1
    end
end

function convolute_s!(mean::Vector{Float64}, data::Vector{Float64}, errors::Vector{Float64}, iter)
    delta = 1
    stop = length(mean)
    maxDelta = length(data)
    @inbounds while delta <= maxDelta
        stmd = stop-delta+1
        val = dot(mean, 1:stmd, mean, delta:stop)/stmd
        d1 = (val-data[delta])
        data[delta] += d1/iter
        d2 = (val-data[delta])
        errors[delta] += d1*d2
        delta += 1
    end
end
