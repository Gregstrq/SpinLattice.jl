using RecipesBase

@recipe function f(cf::CFVals, key::AbstractString)
    i = findfirst(cf.str_vec, key)
    ribbons := cf.errors[i]
    fillalpha := 0.3
    cf.ts, cf.data[i]
end
