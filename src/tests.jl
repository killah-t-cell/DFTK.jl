using DFTK
using Optim
using LinearAlgebra
using Printf

kgrid = [1, 1, 1]       # k-point grid
Ecut = 5                # kinetic energy cutoff in Hartree
tol = 1e-8              # tolerance for the optimization routine
a = 10                  # lattice constant in Bohr
lattice = a * I(3)
H = ElementPsp(:H, psp=load_psp("hgh/lda/h-q1"));
atoms = [H, H]
ψ = nothing
ρ = nothing
function compute_scfres(x)
    model = model_LDA(lattice, atoms, [x[1:3], x[4:6]])
    basis = PlaneWaveBasis(model; Ecut, kgrid)
    global ψ, ρ
    if isnothing(ρ)
        ρ = guess_density(basis)
    end
    scfres = self_consistent_field(basis; ψ=ψ, ρ=ρ,
                                   tol=tol / 10, callback=info->nothing)
    ψ = scfres.ψ
    ρ = scfres.ρ
    scfres
end;
function fg!(F, G, x)
    scfres = compute_scfres(x)
    if G != nothing
        @show compute_forces(scfres)
        @show G
        grad = compute_forces(scfres)
        G .= -[grad[1]; grad[2]]
    end
    scfres.energies.total
end;
x0 = vcat(lattice \ [0., 0., 0.], lattice \ [1.4, 0., 0.])
xres = optimize(Optim.only_fg!(fg!), x0, LBFGS(),
                Optim.Options(show_trace=true, f_tol=tol))
xmin = Optim.minimizer(xres)
dmin = norm(lattice*xmin[1:3] - lattice*xmin[4:6])
@printf "\nOptimal bond length for Ecut=%.2f: %.3f Bohr\n" Ecut dmin



