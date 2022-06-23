

using DFTK
using Plots
using Unitful
using LinearAlgebra
using UnitfulAtomic
using Unitful: Å
using Optim
using Printf

function get_lattice(type, a)
    if type === :fcc
        return a / 2 * [[1.0 1.0 0]; [0 1.0 1.0]; [1.0 0 1.0]]
    end

    if type === :bcc
        return a / 2 * [[1.0 1.0 0]; [0 1.0 1.0]; [1.0 0 1.0]]
    end

    if type === :sc
        return a * I(3)
    end
end

"""
1. Perform calculations to determine whether Pt prefers the 
simple cubic, fcc, or hcp crystal structure. Compare your 
DFT-predicted lattice para- meter(s) of the preferred 
structure with experimental observations.
"""
# The lowest cohesive energy among the three structures would indicate which structure is optimal for Pt
# The optimal lattice constant is one that yields the lowest cohesive energy (E) per atom in the unit cell
# step 0 – set up problem
kgrid = [1, 1, 1]       # k-point grid
Ecut = 300u"eV"               # kinetic energy cutoff in Hartree
tol = 1e-8              # tolerance for the optimization routine
temperature = 0.01
Pt = ElementPsp(:Pt, psp = load_psp("hgh/lda/Pt-q10"));
atoms = [Pt]
position = [zeros(3)]

function compute_scfres(a; lat = :sc)
    model = model_LDA(get_lattice(lat, a), atoms, position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol = tol / 10)
    return scfres.energies.total
end;

# step 1 – find the optimal lattice constant for each of the proposed structure
a_res = optimize((args...) -> compute_scfres(args...; lat = :sc), 4.0, 8.0)
a0 = Optim.minimizer(a_res)
a_sc = auconvert(Å, a0)

a_res = optimize((args...) -> compute_scfres(args...; lat = :fcc), 4.0, 8.0)
a0 = Optim.minimizer(a_res)
a_fcc = auconvert(Å, a0)

a_res = optimize((args...) -> compute_scfres(args...; lat = :bcc), 4.0, 8.0)
a0 = Optim.minimizer(a_res)
a_bcc = auconvert(Å, a0)

# step 2 – Calculate the cohesive energy among the three structures
sc_E = compute_scfres(a_sc; lat = :sc)
fcc_E = compute_scfres(a_fcc; lat = :fcc)
bcc_E = compute_scfres(a_bcc; lat = :bcc)

# step 3 – Find the lowest cohesive energy among the three structures
min(sc_E, fcc_E, bcc_E)

# answer: FCC!!!!
model = model_LDA(get_lattice(:fcc, a_fcc), atoms, [zeros(3)]; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol = tol / 10)


"""
Hf is experimentally observed to be an hcp metal with c/a = 1.58. 
Perform calculations to predict the lattice parameters for Hf and 
compare them with experimental observations.
"""
# step 1 – set up system
hcp(a, c) = [[a / 2 0 0]; [a / 2 a / 2 * sqrt(3) 0]; [0 0 c]]
kgrid = [8, 8, 4]             # k-point grid
Ecut = 500u"eV"               # kinetic energy cutoff in Hartree
tol = 1e-8                    # tolerance for the optimization routine
temperature = 1e-3
Hf = ElementPsp(:Hf, psp = load_psp("hgh/lda/Hf-q12"));
atoms = [Hf]
positions = [zeros(3)]

# step 2 – set up function
function compute_latparams(a)
    model = model_LDA(hcp(a[1], a[2]), atoms, positions; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol = tol / 10)
    scfres.energies.total
end;

# step 3 – multivariable optimize
a0 = [3.0, 5.0]
ares = optimize(compute_latparams, a0, LBFGS(),
    Optim.Options(show_trace = true, f_tol = tol))
ares_min = Optim.minimizer(ares)
c_a = ares_min[2] / ares_min[1]
Hf_c_a = auconvert(Å, a0_scal)


"""
A large number of solids with stoichiometry AB form the CsCl structure. 
In this structure, atoms of A define a simple cubic structure and atoms of
B reside in the center of each cube of A atoms. Define the cell vectors
and fractional coordinates for the CsCl structure, then use this structure
to predict the lattice constant of ScAl.
"""
kgrid = [10, 10, 10]             # k-point grid
Ecut = 500u"eV"               # kinetic energy cutoff in Hartree
tol = 1e-8                    # tolerance for the optimization routine
temperature = 1e-3
Sc = ElementPsp(:Sc, psp = load_psp("hgh/lda/Sc-q3"))
Al = ElementPsp(:Al, psp = load_psp("hgh/lda/Al-q3"))
atoms = [Sc, Al]
positions = [ones(3) * 0.0, ones(3) * 0.5]

function compute_scfres_ScAl(a)
    model = model_LDA(get_lattice(:sc, a), atoms, positions; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol = tol / 10)
    return scfres.energies.total
end;

a_res = optimize(compute_scfres_ScAl, 0.5, 8.0)
a0_scal = Optim.minimizer(a_res)
a_scal = auconvert(Å, a0_scal)

