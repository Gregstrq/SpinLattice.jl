using RecipesBase

@recipe function f(cf::CFVals, key::AbstractString)
    i = findfirst(cf.str_vec, key)
    ribbons := cf.errors[i]
    fillalpha := 0.3
    cf.ts, cf.data[i]
end


@recipe function f(cf::CFVals, key::AbstractString, scale::Float64)
    i = findfirst(cf.str_vec, key)
    ribbons := cf.errors[i]
    fillalpha := 0.3
    cf.ts*scale, cf.data[i]
end

@recipe function f(cf::CFVals, key::AbstractString, scale::Float64, func::Function)
    i = findfirst(cf.str_vec, key)
    fillalpha := 0.3
    cf.ts*scale, func.(cf.data[i])
end
