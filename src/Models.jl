abstract type AbstractModel end

struct Exact <: AbstractModel end
struct Clustered <: AbstractModel end
struct PureClassical <: AbstractModel end
struct Hybrid <: AbstractModel end

build_Approximation(M::Exact, args...) = ExactApprox(args...)
build_Approximation(M::Clustered, args...) = ClusteredApprox(args...)
build_Approximation(M::PureClassical, args...) = PureClassicalApprox(args...)
build_Approximation(M::Hybrid, args...) = HybridApprox(args...)
