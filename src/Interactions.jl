struct Interaction{F}
    func::F
    func_type::String
end

function func_type end
func_type(I::Interaction{F}) where {F} = I.func_type
Interaction(func) = Interaction{typeof(func)}(func, func_type(func))
@inline (I::Interaction)(v::Vector{T}) where T<:Real = I.func(v)

function nearest_neighbours(v::Vector{T}) where T<:Real
    margin = 1e-9
    if abs(norm(v)-1.) < margin
        return 1.
    else
        return 0.
    end
end

struct MDI{D}
    dir::NTuple{D, Int64}
    ndir::Vector{Float64}
    function MDI(dir::NTuple{D, Int64}) where {D}
        ndir = collect(dir)/norm(collect(dir))
        new{D}(dir, ndir)
    end
end
@inline MDI(dir...) = MDI(dir)

function (mdi::MDI)(v::Vector{T}) where T<:Real
    cos_theta = dot(mdi.ndir, v)/norm(v)
    return (1-3*cos_theta^2)/norm(v)^3
end

function func_type(Func)
    if isequal(Func, nearest_neighbours)
        return "(nn)"
    end
    if typeof(Func)<:MDI
        return "(md)$(Func.dir)"
    else
        error("no string for this function\n")
    end
end
