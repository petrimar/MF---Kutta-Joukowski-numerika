from mpi4py import MPI
from dolfinx import mesh, fem, io
from meshify import generate_mesh

comm = MPI.COMM_WORLD

# diskretizacija domene
    # datoteka koju prima generate_mesh() mora sadržavati samo koordinate točaka (po redovima) koje u danom poretku "neprekidno" opisuju rub aeroprofila
    # https://m-selig.ae.illinois.edu/ads/coord_seligFmt/ - sadrži >1600 takvih datoteka iz kojih se samo treba ukloniti prvi red i eventualno zadnji red (ako je zadnja točka jednaka prvoj)
generate_mesh("profiles/hh02.dat")
meshdata = io.gmsh.read_from_msh("mesh.msh", comm)
domain = meshdata.mesh

# granice - outer i airfoil
import numpy as np
tdim = domain.topology.dim
bdry_outer = mesh.locate_entities_boundary(domain, tdim-1, lambda x: x[0]**2 + x[1]**2 > 2)
bdry_airfoil = np.setdiff1d(mesh.exterior_facet_indices(domain.topology), bdry_outer)

# trailing edge paneli
v_edge = mesh.locate_entities_boundary(domain, 0, lambda x: (np.isclose(x[0],1.0)) & (abs(x[1]) < 1))
f_edge = mesh.locate_entities_boundary(domain, 1, lambda x: (np.isclose(x[0],1.0)) & (abs(x[1]) < 1))
x = domain.geometry.x
print("trailing edge vertices: ", v_edge, " ~ ", x[v_edge])

domain.topology.create_connectivity(0, tdim-1)
v_to_f = domain.topology.connectivity(0, tdim-1)
if len(v_edge)==2 :
    v1, v2 = sorted(v_edge, key=lambda v: x[v][1], reverse=True)
    for f in v_to_f.links(v1):
        if f in bdry_airfoil and f != f_edge[0] : f1 = f
    for f in v_to_f.links(v2):
        if f in bdry_airfoil and f != f_edge[0] : f2 = f
elif len(v_edge)==1 :
    f1, f2 = np.intersect1d(bdry_airfoil, v_to_f.links(v_edge[0]))

m1, m2 = mesh.compute_midpoints(domain, tdim-1, np.array([f1, f2]))
if m1[1] < m2[1] :
    f1, f2 = f2, f1
    m1, m2 = m2, m1

print("stražnji gornji panel:", f1, " ~ ", m1)
print("stražnji donji panel:", f2, " ~ ", m2)

# meshtags (za integriranje po dijelovima ruba profila)
bdry_rest = np.setdiff1d(bdry_airfoil, [f1, f2])
marker_upper = np.full_like([f1], 1)
marker_lower = np.full_like([f2], 2)
marker_rest = np.full_like(bdry_rest, 3)
facet_indices = np.hstack([[f1,f2],bdry_rest]).astype(np.int32)
facet_markers = np.hstack([marker_upper, marker_lower,marker_rest]).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
tags = mesh.meshtags(domain, tdim-1, facet_indices[sorted_facets], facet_markers[sorted_facets])

# centar vrtloga v(x)
h = domain.h(1, bdry_airfoil)
idx2 = mesh.locate_entities_boundary(domain, 1, lambda x: (abs(x[0] - 0.5) < max(h)) & (abs(x[1]) < 1))
idx2 = idx2[np.argsort(idx2[:])]
f3 = idx2[-1]
f4 = idx2[0]
m3, m4 = mesh.compute_midpoints(domain, tdim-1, np.array([f3, f4]))

print("srednji gornji panel:", f3, " ~ ", m3)
print("srednji donji panel:", f4, " ~ ", m4)

x0 = [(m3[0] + m4[0])/2, (m3[1] + m4[1])/2, 0.0]
print("centar vrtloga:", x0)
print("udaljenost centra vrtloga od ruba profila:", abs(x0[1]-m3[1]))

# ------------------------------------------------------------------------------------------------------------------------------------------------

# funkcijski prostori
V = fem.functionspace(domain, ("CG", 1)) # - potencijal
V_vec = fem.functionspace(domain, ("CG", 1, (domain.geometry.dim,))) # - brzina

# ufl
import ufl
phi = ufl.TrialFunction(V)
eta = ufl.TestFunction(V)
n = ufl.FacetNormal(domain)
t = ufl.as_vector((n[1], -n[0], n[2]))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

# slaba formulacija
a = ufl.dot(ufl.grad(phi), ufl.grad(eta))*ufl.dx
L = eta*fem.Constant(domain, 0.0)*ufl.dx

# dirichlet na vanjskoj granici
dofs_outer = fem.locate_dofs_topological(V, tdim-1, bdry_outer)
phi_outer = fem.Function(V)
phi_outer.interpolate(lambda x: x[0])
bc_outer = fem.dirichletbc(phi_outer, dofs_outer)

# solver
from dolfinx.fem import petsc
problem = petsc.LinearProblem(a, L, bcs=[bc_outer], petsc_options_prefix="P")

# prvi solve - tok bez cirkulacije
phi_0 = problem.solve()

u = fem.Function(V_vec)
u.name = "velocity"
u.interpolate(fem.Expression(ufl.grad(phi_0), V_vec.element.interpolation_points))
with io.XDMFFile(domain.comm, "results/BaseVelocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

circumference = fem.assemble_scalar(fem.form(1.0*ds(1)+1.0*ds(2)+1.0*ds(3)))
print("opseg profila =", circumference)

# cirkulacija bi trebala biti 0
circulation = fem.assemble_scalar(fem.form(ufl.dot(u,t)*ds(1) + ufl.dot(u,t)*ds(2) + ufl.dot(u,t)*ds(3)))
print("[u0] cirkulacija =", circulation, "(trebalo bi biti ~0)")

# vrtlog oko točke x0 u profilu
x = ufl.SpatialCoordinate(domain)
v = ufl.as_vector([-(x[1]-x0[1]), x[0]-x0[0], 0.0]) / ((x[0]-x0[0])**2 + (x[1]-x0[1])**2) / (2*np.pi)

# slaba formulacija za drugi solve
phi_outer.interpolate(lambda x: np.zeros_like(x[0]))
bc_outer = fem.dirichletbc(phi_outer, dofs_outer)

L = - eta*ufl.dot(v,n)*ds(1) - eta*ufl.dot(v,n)*ds(2) - eta*ufl.dot(v,n)*ds(3)
problem = petsc.LinearProblem(a, L, bcs=[bc_outer], petsc_options_prefix="P")

chi = problem.solve()
w = v + ufl.grad(chi)
    
# provjera - w bi trebao imati "jediničnu" cirkulaciju
circulation = fem.assemble_scalar(fem.form(ufl.dot(w,t)*ds(1) + ufl.dot(w,t)*ds(2) + ufl.dot(w,t)*ds(3)))
print("[w] cirkulacija =", circulation, "(trebalo bi biti ~1)")

# Kuttina cirkulacija
length_upper = fem.assemble_scalar(fem.form(1.0*ds(1)))
length_lower = fem.assemble_scalar(fem.form(1.0*ds(2)))
a_upper = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(phi_0),t)*ds(1))) / length_upper
a_lower = fem.assemble_scalar(fem.form(ufl.dot(ufl.grad(phi_0),t)*ds(2))) / length_lower
b_upper = fem.assemble_scalar(fem.form(ufl.dot(w,t)*ds(1))) / length_upper
b_lower = fem.assemble_scalar(fem.form(ufl.dot(w,t)*ds(2))) / length_lower

Gamma = -(a_upper + a_lower) / (b_upper + b_lower) # valjda "+" jer bi tangente trebali bit suprotno orijentirane
print("Kuttina cirkulacija =", Gamma)

# vrtložni dio toka
u.interpolate(fem.Expression(Gamma*w, V_vec.element.interpolation_points))
with io.XDMFFile(domain.comm, "results/VortexVelocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

# konačno riješenje
u.interpolate(fem.Expression(ufl.grad(phi_0) + Gamma*w, V_vec.element.interpolation_points))
with io.XDMFFile(domain.comm, "results/KuttaVelocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

# cirkulacija "ručno"
circulation = fem.assemble_scalar(fem.form(ufl.dot(u,t)*ds(1) + ufl.dot(u,t)*ds(2) + ufl.dot(u,t)*ds(3)))
print("[u] cirkulacija =", circulation)

