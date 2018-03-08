using RecipesBase

@recipe function f(data::CFVals)
    :label := hcat(data.str_vec...)
    cfs = Array{eltype(data.data)}(size(data.data))
    for i in Base.OneTo(length(data.data))
        cfs[:,i] .= data.data[i]./data.data[1,i]
    end
    data.ts, cfs
end
