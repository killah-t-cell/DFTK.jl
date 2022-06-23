"""
Develop supercells suitable for performing calculations with the 
(100), (110), and (111) surfaces of an fcc metal. What size 
surface unit cell is needed for each surface to examine surface 
relaxation of each surface?
"""
miller100 = (1, 0, 0)
miller110 = (1, 1, 0)
miller111 = (1, 1, 1)



"""
Extend your calculations from Exercise 1 to calculate the surface energy of Pt(100), Pt(110), and Pt(111).
"""

"""
Extend your calculations from Exercise 1 to calculate the surface energy of Pt(100), Pt(110), and Pt(111).
"""

"""
Pt(110) is known experimentally to reconstruct into the so-called missing-row reconstruction. 
In this reconstruction, alternate rows from the top layer of the surface in a (2 X 1) surface 
unit cell are missing. Use a supercell defined by a (2 􏰁 1) surface unit cell of Pt(110) 
to compute the surface energy of the unreconstructed and reconstructed surfaces. 
Why does this comparison need to be made based on surface energy rather than simply based on the 
total energy of the supercells? Are your results consistent with the experimental observations? 
Use similar calculations to predict whether a similar reconstruction would be expected to exist for Cu(110).
"""

"""
Perform calculations to determine the preferred binding sites for atomic O on Pt(111) 
using ordered overlayers with coverages of 0.25 and 0.33 ML. 
Does the adsorption energy increase or decrease as the surface coverage is increased?
"""

"""
Perform calculations similar to Exercise 4, but for the adsorption of hydroxyl groups, 
OH, on Pt(111). What tilt angle does the OH bond form with the surface normal in its preferred adsorption configuration? 
What numerical evidence can you provide that your calculations adequately explored the possible tilt angles?
"""

"""
Just testing
"""
miller = (1, 1, 0)   # Surface Miller indices
n_GaAs = 2           # Number of GaAs layers
n_vacuum = 4         # Number of vacuum layers
Ecut = 5             # Hartree
kgrid = (4, 4, 1);   # Monkhorst-Pack mesh -> use fewer k points for z because z is larger in real space and smaller in reciprocal space3

using PyCall

ase_build = pyimport("ase.build")
a = 5.6537  # GaAs lattice parameter in Ångström (because ASE uses Å as length unit)
gaas = ase_build.bulk("GaAs", "zincblende", a = a)
surface = ase_build.surface(gaas, miller, n_GaAs, 0, periodic = true)

d_vacuum = maximum(maximum, surface.cell) / n_GaAs * n_vacuum
surface = ase_build.surface(gaas, miller, n_GaAs, d_vacuum, periodic = true)

pyimport("ase.io").write("surface.png", surface * (3, 3, 1),
    rotation = "-90x, 30y, -75z")

using DFTK

positions = load_positions(surface)
lattice = load_lattice(surface)
atoms = map(load_atoms(surface)) do el
    if el.symbol == :Ga
        ElementPsp(:Ga, psp = load_psp("hgh/pbe/ga-q3.hgh"))
    elseif el.symbol == :As
        ElementPsp(:As, psp = load_psp("hgh/pbe/as-q5.hgh"))
    else
        error("Unsupported element: $el")
    end
end

model = model_DFT(lattice, atoms, positions, [:gga_x_pbe, :gga_c_pbe],
    temperature = 0.001, smearing = DFTK.Smearing.Gaussian())
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres = self_consistent_field(basis, tol = 1e-4, mixing = LdosMixing())