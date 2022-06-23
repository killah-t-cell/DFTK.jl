using DFTK
using LinearAlgebra
setup_threading(n_blas=1)
using Plots

function aluminium_setup(repeat=1; Ecut=12.0, kgrid=[2, 2, 2])
    a = 7.65339
    lattice = a * Matrix(I, 3, 3)
    Al = ElementPsp(:Al, psp=load_psp("hgh/lda/al-q3"))
    atoms     = [Al, Al, Al, Al]
    positions = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]

    # Make supercell in ASE:
    # We convert our lattice to the conventions used in ASE
    # and then back ...
    supercell = ase_atoms(lattice, atoms, positions) * (repeat, 1, 1)
    lattice   = load_lattice(supercell)
    positions = load_positions(supercell)
    atoms = fill(Al, length(positions))

    # Construct an LDA model and discretise
    model = model_LDA(lattice, atoms, positions; temperature=1e-3, symmetries=false)
    PlaneWaveBasis(model; Ecut, kgrid)
end

"""
One can patch or extend the SCF procedure by 
replacing parts of the code with custom callback functions.

Goal: construct our own SCF solver
"""

p = plot(yaxis=:log)
density_differences = Float64[]
function plot_callback(info)
    if info.stage == :finalize
    #When done with SCF: make plot
    plot!(p, density_differences, label="|ρout - ρin|", markershape=:x)
    else    
    # add the density difference of this step
    push!(density_differences, norm(info.ρout - info.ρin))
    end

    info # pass to allow callback chaining
end

# chain custom callback with the default one
callback = DFTK.ScfDefaultCallback() ∘ plot_callback
scfres = self_consistent_field(aluminium_setup(); tol=1e-12, callback, mixing=SimpleMixing())




###

using DFTK

a = 10.26
lattice = a / 2 * [[0 1 1.];
                   [1 0 1.];
                   [1 1 0.]]
Si = ElementPsp(:Si, psp=load_psp("hgh/lda/si-q4"))
atoms = [Si, Si]
positions = [-ones(3)/8, ones(3)/8]

# Guess some inital magnetic moments
# (Need to be != 0 otherwise SCF makes the assumption that the ground state is not magnetic
#  to gain some performance ...)
magnetic_moments = [2, 2]

model  = model_LDA(lattice, atoms, positions; temperature=0.01, magnetic_moments)
basis  = PlaneWaveBasis(model; Ecut=13, kgrid=[2, 2, 2]);
ρ0     = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis; ρ=ρ0)
scfres.energies.total

a = 5.42352  # iron lattice constant in bohr
lattice = a / 2 * [[-1  1  1];
                   [ 1 -1  1];
                   [ 1  1 -1]]
Fe = ElementPsp(:Fe, psp=load_psp("hgh/lda/Fe-q8.hgh"))
atoms = [Fe]
positions = [zeros(3)]
magnetic_moments = [3]

model  = model_LDA(lattice, atoms, positions; temperature=0.01, magnetic_moments)
basis  = PlaneWaveBasis(model; Ecut=30, kgrid=[5,5,5]);
ρ0     = guess_density(basis, magnetic_moments)
scfres = self_consistent_field(basis; ρ=ρ0)


