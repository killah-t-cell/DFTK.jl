
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
In the exercises for Chapter 2 we suggested calculations for several materials, 
including Pt in the cubic and fcc crystal structures and ScAl in the CsCl structure. 
Repeat these calculations, this time developing numerical evidence that your results 
are well converged in terms of sampling k space and energy cutoff.

ANTONIE LEVITT ON CONVERGENCE STUDY
This is the nasty part about DFT that ideally we'd like to automate. 
In general you have to do a convergence study: start with a reasonable grid, 
pick a property (eg energy), increase the grid until the property doesn't 
change too much and use that value. You have to do this study (separately) 
for kgrid, Ecut (and, in theory, temperature, though in practice you can 
often fix a reasonably small value and not care too much about it) 
Eg abinit has a nice tutorial on how to perform a convergence study.
"""
#The optimal value of kpoints is obtained when the total energy difference with 
# regard to the result of highest kpoints is not more than 0.02 eV

kgrid = [1, 1, 1]       # k-point grid
tol = 1e-2              # tolerance for the optimization routine
Ecut = 490u"eV"
temperature = 0.01
Pt = ElementPsp(:Pt, psp = load_psp("hgh/lda/Pt-q10"));
atoms = [Pt]
position = [zeros(3)]
a = 5.00 # in Bohr

# set up convergence functions
function compute_Ecut_convergence(Ecut; lat = :sc)
    model = model_LDA(get_lattice(lat, a), atoms, position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol = tol / 10)
    return scfres.energies.total
end

function compute_kgrid_convergence(k; lat = :sc)
    model = model_LDA(get_lattice(lat, a), atoms, position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid = [Int64(ceil(k)) for _ in 1:3])
    scfres = self_consistent_field(basis; tol = tol / 10)
    return scfres.energies.total
end

function get_converged_k(energies; tol)
    for i in eachindex(energies)
        @show energies[i] - energies[i+1]
        if abs(energies[i] - energies[i+1]) < tol
            return i
        end
    end
end

#get k
energies_w_diff_k = [compute_kgrid_convergence(k) for k in 1:10]
k_conv = get_converged_k(energies_w_diff_k; tol) # 5.0
plot(compute_kgrid_convergence, 1:1:10)

# get Ecut
opt_res = optimize(compute_Ecut_convergence, 7, 22)
E_cut_conv = Optim.minimizer(opt_res)
E_cut_conv_H = auconvert(u"eV", E_cut_conv)
plot(compute_Ecut_convergence, 7:1:22)

# get k optim – less efficient (overkill)
opt_res = optimize(compute_kgrid_convergence, 1, 10; abs_tol = 0.01)
k_conv = ceil(Optim.minimizer(opt_res)) # 7.0



"""
Use methods similar to those in Section 3.5.1 to optimize the geometry 
of H2O and hydrogen cyanide, HCN.
"""
### H2O ###
kgrid = [1, 1, 1]       # k-point grid
Ecut = 200u"eV"         # kinetic energy cutoff in Hartree
tol = 1e-3              # tolerance for the optimization routine
a = 10                  # lattice constant in Bohr
temperature = 0.001
lattice = a * I(3)
H = ElementPsp(:H, psp = load_psp("hgh/lda/h-q1"))
O = ElementPsp(:O, psp = load_psp("hgh/lda/o-q6"))
atoms = [H, O, H]

ψ = nothing
ρ = nothing
function compute_scfres(x)
    model = model_LDA(lattice, atoms, [x[1:3], x[4:6], x[7:9]]; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    global ψ, ρ
    if isnothing(ρ)
        ρ = guess_density(basis)
    end
    scfres = self_consistent_field(basis; ψ = ψ, ρ = ρ,
        tol = tol / 10, callback = info -> nothing)
    ψ = scfres.ψ
    ρ = scfres.ρ
    scfres
end;
function fg!(F, G, x)
    scfres = compute_scfres(x)
    if G != nothing
        grad = compute_forces(scfres)
        G .= -[grad[1]; grad[2]; grad[3]]
    end
    scfres.energies.total
end;
x0 = vcat(lattice \ [0.1, 0.0, 0.0], lattice \ [0, 0.0, 0.0], lattice \ [0.0, 0.1, 0.0])
xres = optimize(Optim.only_fg!(fg!), x0, LBFGS(),
    Optim.Options(show_trace = true, f_tol = tol))
xmin = Optim.minimizer(xres)
dmin = norm(lattice * xmin[1:3] - lattice * xmin[4:6])
@printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut dmin

### HCN ###
kgrid = [1, 1, 1]       # k-point grid
Ecut = 5                # kinetic energy cutoff in Hartree
tol = 1e-8              # tolerance for the optimization routine
a = 10                  # lattice constant in Bohr
lattice = a * I(3)
H = ElementPsp(:H, psp = load_psp("hgh/lda/h-q1"))
C = ElementPsp(:C, psp = load_psp("hgh/lda/c-q4"))
N = ElementPsp(:N, psp = load_psp("hgh/lda/n-q5"))

atoms = [H, C, N]

ψ = nothing
ρ = nothing
function compute_scfres(x)
    model = model_LDA(lattice, atoms, [x[1:3], x[4:6], x[7:9]]; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    global ψ, ρ
    if isnothing(ρ)
        ρ = guess_density(basis)
    end
    scfres = self_consistent_field(basis; ψ = ψ, ρ = ρ,
        tol = tol / 10, callback = info -> nothing)
    ψ = scfres.ψ
    ρ = scfres.ρ
    scfres
end;
function fg!(F, G, x)
    scfres = compute_scfres(x)
    if G != nothing
        grad = compute_forces(scfres)
        G .= -[grad[1]; grad[2]; grad[3]]
    end
    scfres.energies.total
end;
x0 = vcat(lattice \ [0.0, 0.0, 0.0], lattice \ [1.4, 0.0, 0.0], lattice \ [0, 1.4, 0.0])
xres = optimize(Optim.only_fg!(fg!), x0, LBFGS(),
    Optim.Options(show_trace = true, f_tol = tol))
xmin = Optim.minimizer(xres)
dmin = norm(lattice * xmin[1:3] - lattice * xmin[4:6])
@printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut dmin

"""
Perform the calculations necessary to estimate the energy difference associated 
with forming an ordered bcc CuPd alloy from fcc Pd and fcc Cu. The ordered alloy 
is formed by defining a bcc crystal with Cu atoms at the corners of each 
cube and Pd atoms in the center of each cube (or vice versa). 
This ordered alloy is known to be the favored low temperature crystal 
structure of Pd and Cu when they are mixed with this stoichiometry. 
What does this observation tell you about the sign of the energy difference 
you are attempting to calculate? To calculate this energy difference you 
will need to optimize the lattice constant for each material and pay 
careful attention to how your energy cutoffs and k points are chosen.
"""

# fcc Cu, fcc Pd and bcc CuPd
kgrid = [10, 10, 10]       # k-point grid
Ecut = 5000u"eV"               # kinetic energy cutoff in Hartree
tol = 1e-8              # tolerance for the optimization routine
temperature = 0.01
Cu = ElementPsp(:Cu, psp = load_psp("hgh/lda/cu-q11"))
Pd = ElementPsp(:Pd, psp = load_psp("hgh/lda/pd-q10"))
atoms_cu = [Cu]
atoms_pd = [Pd]
atoms_CuPd = [Cu, Pd]

function compute_scfres(a; lat = :fcc, atoms, position = [zeros(3)], kgrid = kgrid, Ecut = Ecut)
    model = model_LDA(get_lattice(lat, a), atoms, position; temperature)
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    scfres = self_consistent_field(basis; tol = tol / 10)
    return scfres.energies.total
end

# optimize Cu
cu_res = optimize((args...) -> compute_scfres(args...; lat = :fcc, atoms = atoms_cu), 2.0, 8.0)
a0 = Optim.minimizer(cu_res)
cu_fcc = auconvert(Å, a0)

# get energy Cu
model = model_LDA(get_lattice(:fcc, cu_fcc), atoms_cu, [zeros(3)]; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres_cu = self_consistent_field(basis; tol = tol / 10)
scfres_cu_total = scfres_cu.energies.total

# optimize Pd
pd_res = optimize((args...) -> compute_scfres(args...; lat = :fcc, atoms = atoms_pd), 2.0, 6.0)
a0 = Optim.minimizer(pd_res)
pd_fcc = auconvert(Å, a0)

# get energy Pd
model = model_LDA(get_lattice(:fcc, pd_fcc), atoms_pd, [zeros(3)]; temperature)
basis = PlaneWaveBasis(model; Ecut, kgrid)
scfres_cu = self_consistent_field(basis; tol = tol / 10)
scfres_cu_total = scfres_cu.energies.total

# optimize CuPd
a_init = 3
cu_res = optimize((args...) -> compute_scfres(args...;
        lat = :bcc,
        atoms = atoms_CuPd,
        position = [[0.0, 0.0, a_init], [a_init / 2, a_init / 2, a_init / 2]]
    ),
    2.0,
    6.0)
a0 = Optim.minimizer(cu_res)
cupd_bcc = auconvert(Å, a0)

# get energy Cu
a = a0
model = model_LDA(get_lattice(:bcc, a), atoms_CuPd, [[0.0, 0.0, a], [a / 2, a / 2, a / 2]]; temperature = 1e-3)
basis = PlaneWaveBasis(model; Ecut = 20, kgrid = [9, 9, 9])
scfres_cupd = self_consistent_field(basis; tol = tol / 10)
scfres_cupd_total = scfres_cupd.energies.total
