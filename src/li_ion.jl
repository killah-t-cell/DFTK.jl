"""
Find the equilibrium voltage of a lithium transition-metal oxide intercalation cathode 
with composition LiMO2 and a lithium metal anode with the cell reaction.
All that is required to compute the voltage are three independent first principles 
calculations for Lix1MO2, Lix2MO2, and Li, and the energy of BCC lithium is independent
of the cathode material and hence only needs to be computed once
"""

using DFTK
using Plots
using Unitful
using UnitfulAtomic
using LinearAlgebra
using Optim

###LiCoO2
kgrid = [8, 8, 8]       # k-point grid
tol = 1e-4              # tolerance for the optimization routine
Ecut = 540u"eV"
temperature = 1e-3
a = 2.82
c = 14.10
hcp_lattice = [[a/2 0 0]; [a/2 a/2*sqrt(3) 0]; [0 0 c]]

Li = ElementPsp(:Li, psp=load_psp("hgh/lda/li-q1"))
Co = ElementPsp(:Co, psp=load_psp("hgh/lda/co-q9"))
O = ElementPsp(:O, psp=load_psp("hgh/lda/o-q6"))

#LiCoO2
atoms = [Li, Co, O, O]
positions = [zeros(3), [0.,0., 0.5], [0.,0., 0.245], [0.,0., -0.245]]

model = model_LDA(hcp_lattice, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres_LiCoO2 = self_consistent_field(basis; tol=tol / 10)

#LiCoO2
atoms = [Co, O, O]
positions = [[0.,0., 0.5], [0.,0., 0.245], [0.,0., -0.245]]

model = model_LDA(hcp_lattice, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres_CoO2 = self_consistent_field(basis; tol=tol / 10)


#Li
atoms = [Li]
positions = [zeros(3)]

model = model_LDA(I(3)*c, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid=[1,1,1])
scfres_Li = self_consistent_field(basis; tol=tol / 10)

F = 96485
eq_v = - (scfres_LiCoO2.energies.total - scfres_CoO2.energies.total - scfres_Li.energies.total)/F # wrong results
###
#=

# Lattice
a = 2.82
c = 14.10
hcp_lattice = a * I(3)

Li = ElementPsp(:Li, psp=load_psp("hgh/lda/li-q1"))
Co = ElementPsp(:Co, psp=load_psp("hgh/lda/co-q9"))
O = ElementPsp(:O, psp=load_psp("hgh/lda/o-q6"))

#Li3Co3O6
atoms = [Li, Li, Li,Co, Co, Co, O, O, O, O, O, O]
positions = [zeros(3), [0.5,0., 0.0], [0.245,0., 0.], 
            [0.0, 0.5,0.0], [0.5, 0.5, 0.0], [0.245, 0.5, 0.0],
            [0.0, 0.5,0.5], [0.5, 0.5, 0.5], [0.245, 0.5, 0.5],[0.125, 0.5,0.5], [0.125, 0.5, 0.5], [0.245, 0.5, 0.5]]

model = model_LDA(hcp_lattice, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres_Li3Co3O6 = self_consistent_field(basis; tol=tol / 10)

#Li2Co3O6
atoms = [Li, Li,Co, Co, Co, O, O, O, O, O, O]
positions = [zeros(3), [0.,0., 0.5], [0.,0., 0.245], [0.,0., -0.245]]

model = model_LDA(hcp_lattice, atoms, positions; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres_Li2Co3O6 = self_consistent_field(basis; tol=tol / 10)
=#


