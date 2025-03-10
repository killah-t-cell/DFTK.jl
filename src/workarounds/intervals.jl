import SpecialFunctions: erfc
const Interval = IntervalArithmetic.Interval

# Monkey-patch a few functions for Intervals
# ... this is far from proper and a bit specific for our use case here
# (that's why it's not contributed upstream).
# should be done e.g. by changing  the rounding mode ...
erfc(i::Interval) = Interval(prevfloat(erfc(i.lo)), nextfloat(erfc(i.hi)))

# This is done to avoid using sincospi(x), called by cispi(x),
# which has not been implemented in IntervalArithmetic
# see issue #513 on IntervalArithmetic repository
cis2pi(x::Interval) = exp(2 * (pi * (im * x)))

function compute_Glims_fast(lattice::AbstractMatrix{<:Interval}, args...; kwargs...)
    # This is done to avoid a call like ceil(Int, ::Interval)
    # in the above implementation of compute_fft_size,
    # where it is in general cases not clear, what to do.
    # In this case we just want a reasonable number for Gmax,
    # so replacing the intervals in the lattice with
    # their midpoints should be good.
    compute_Glims_fast(IntervalArithmetic.mid.(lattice), args...; kwargs...)
end
function compute_Glims_precise(::AbstractMatrix{<:Interval}, args...; kwargs...)
    error("fft_size_algorithm :precise not supported with intervals")
end

function _is_well_conditioned(A::AbstractArray{<:Interval}; kwargs...)
    # This check is used during the lattice setup, where it frequently fails with intervals
    # (because doing an SVD with intervals leads to a large overestimation of the rounding error)
    _is_well_conditioned(IntervalArithmetic.mid.(A); kwargs...)
end

function symmetry_operations(lattice::AbstractMatrix{<:Interval}, atoms, positions,
                             magnetic_moments=[];
                             tol_symmetry=max(SYMMETRY_TOLERANCE, maximum(radius, lattice)))
    @assert tol_symmetry < 1e-2
    symmetry_operations(IntervalArithmetic.mid.(lattice), atoms, positions, magnetic_moments;
                        tol_symmetry)
end

function local_potential_fourier(el::ElementCohenBergstresser, q::T) where {T <: Interval}
    lor = round(q.lo, digits=5)
    hir = round(q.hi, digits=5)
    @assert iszero(round(lor - hir, digits=3))
    T(local_potential_fourier(el, IntervalArithmetic.mid(q)))
end

function estimate_integer_lattice_bounds(M::AbstractMatrix{<:Interval}, δ, shift=zeros(3))
    # As a general statement, with M a lattice matrix, then if ||Mx|| <= δ, 
    # then xi = <ei, M^-1 Mx> = <M^-T ei, Mx> <= ||M^-T ei|| δ.
    # Below code does not support non-3D systems.
    xlims = [norm(inv(M')[:, i]) * δ + shift[i] for i in 1:3]
    map(x -> ceil(Int, x.hi), xlims)
end
