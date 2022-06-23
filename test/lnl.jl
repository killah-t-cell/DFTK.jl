
using DFTK
using Plots
using Unitful
using UnitfulAtomic
using LinearAlgebra
using Unitful: Å
using Optim

# Example 1
"""
Compute the LDA ground state of the silicon crystal
"""
# 1. Define lattice and atomic positions
a = 5.431u"angstrom"          # Silicon lattice constant
lattice = a / 2 * [[0 1 1.];  # Silicon lattice vectors
                   [1 0 1.];  # specified column by column
                   [1 1 0.]]

# Load HGH pseudopotential for Silicon
# The residual attractive interaction between an electron and an ion
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/Si-q4"))

# Specify type and positions of atoms
atoms     = [Si, Si]
positions = [ones(3)/8, -ones(3)/8]

# 2. Select model and basis
model = model_LDA(lattice, atoms, positions) # local density approximation
kgrid = [4, 4, 4]     # k-point grid (Regular Monkhorst-Pack grid)
Ecut = 7              # kinetic energy cutoff
# Ecut = 190.5u"eV"  # Could also use eV or other energy-compatible units
basis = PlaneWaveBasis(model; Ecut, kgrid)
# Note the implicit passing of keyword arguments here:
# this is equivalent to PlaneWaveBasis(model; Ecut=Ecut, kgrid=kgrid)

# 3. Run the SCF procedure to obtain the ground state
scfres = self_consistent_field(basis, tol=1e-8);
scfres.energies
hcat(scfres.eigenvalues...)
hcat(scfres.occupation...)
rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis
x = [r[1] for r in rvecs]                   # only keep the x coordinate
plot(x, scfres.ρ[1, :, 1, 1], label="", xlabel="x", ylabel="ρ", marker=2)
plot_bandstructure(scfres; kline_density=10)


"""
Model a magnesium lattice as a simple example for a metallic system
"""
# Example 2
a = 3.01794  # bohr
b = 5.22722  # bohr
c = 9.77362  # bohr
lattice = [[-a -a  0]; [-b  b  0]; [0   0 -c]]
Mg = ElementPsp(:Mg, psp=load_psp("hgh/pbe/Mg-q2"))
atoms     = [Mg, Mg]
positions = [[2/3, 1/3, 1/4], [1/3, 2/3, 3/4]];

kspacing = 0.945 / u"angstrom"        # Minimal spacing of k-points,
#                                      in units of wavevectors (inverse Bohrs)
Ecut = 5                              # Kinetic energy cutoff in Hartree
temperature = 0.01                    # Smearing temperature in Hartree
smearing = DFTK.Smearing.FermiDirac() # Smearing method

model = model_DFT(lattice, atoms, positions, [:gga_x_pbe, :gga_c_pbe];
                  temperature, smearing)
kgrid = kgrid_from_minimal_spacing(lattice, kspacing)
basis = PlaneWaveBasis(model; Ecut, kgrid);

scfres = self_consistent_field(basis, damping=0.8, mixing=KerkerMixing());
scfres.occupation[1]
scfres.energies
plot_dos(scfres)


# Example 3
"""
1. Perform calculations to determine whether Pt prefers the 
simple cubic, fcc, or hcp crystal structure. Compare your 
DFT-predicted lattice para- meter(s) of the preferred 
structure with experimental observations.
"""
# The lowest cohesive energy among the three structures would indicate which structure is optimal for Pt
# The optimal lattice constant is one that yields the lowest cohesive energy (E) per atom in the unit cell
# step 0 – set up problem
function get_lattice(type, a) 
    if type === :fcc
        return a / 2 * [[1. 1. 0]; [0 1. 1.]; [1. 0 1.]]
    end

    if type === :bcc
        return a / 2 * [[1. 1. 0]; [0 1. 1.]; [1. 0 1.]]
    end

    if type === :sc
        return a * I(3)
    end
end

kgrid = [1, 1, 1]       # k-point grid
Ecut = 300u"eV"               # kinetic energy cutoff in Hartree
tol = 1e-8              # tolerance for the optimization routine
temperature = 1e-3
Pt = ElementPsp(:Pt, psp=load_psp("hgh/lda/Pt-q10"));
atoms = [Pt]

function compute_scfres(a; lat=:sc)
    model = model_LDA(get_lattice(lat, a), atoms, [zeros(3)]; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol=tol / 10)
    return scfres.energies.total
end;

# step 1 – find the optimal lattice constant for each of the proposed structure
a_res = optimize((args...) -> compute_scfres(args...; lat=:sc), 4.0, 8.0) 
a0 = Optim.minimizer(a_res)
a_sc = auconvert(Å, a0)

a_res = optimize((args...) -> compute_scfres(args...; lat=:fcc), 4.0, 8.0) 
a0 = Optim.minimizer(a_res)
a_fcc = auconvert(Å, a0)

a_res = optimize((args...) -> compute_scfres(args...; lat=:bcc), 4.0, 8.0) 
a0 = Optim.minimizer(a_res)
a_bcc = auconvert(Å, a0)

# step 2 – Calculate the cohesive energy among the three structures
sc_E = compute_scfres(a_sc; lat=:sc)
fcc_E = compute_scfres(a_fcc; lat=:fcc)
bcc_E = compute_scfres(a_bcc; lat=:bcc)

# step 3 – Find the lowest cohesive energy among the three structures
min(sc_E,fcc_E, bcc_E)

# answer: FCC!!!!
model = model_LDA(get_lattice(:fcc, a_fcc), atoms, [zeros(3)]; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis; tol=tol / 10)
plot_dos(scfres)
plot_bandstructure(scfres)