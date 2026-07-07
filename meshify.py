import gmsh

def generate_mesh(filename):

    gmsh.initialize()
    gmsh.model.add("airfoil")

    # pravokutnik
    rectangle = gmsh.model.occ.addRectangle(-5, -5, 0, 10, 10)

    # aeroprofil
    with open(filename) as f:
        coords = [ tuple(map(float, line.split())) for line in f ]
    points = [ gmsh.model.occ.addPoint(x, y, 0.0, 0.05) for x, y in coords ]
    lines = [ gmsh.model.occ.addLine(points[i], points[i+1]) for i in range(len(points)-1) ]
    lines.append(gmsh.model.occ.addLine(points[-1], points[0]))
    loop = gmsh.model.occ.addCurveLoop(lines)
    airfoil = gmsh.model.occ.addPlaneSurface([loop])
    gmsh.model.occ.synchronize()

    # domena = pravokutnik - aeroprofil    
    gmsh.model.occ.cut([(2, rectangle)], [(2, airfoil)])
    gmsh.model.occ.synchronize()

    # "physical groups"
    surfaces = gmsh.model.getEntities(dim=2)
    gmsh.model.addPhysicalGroup(2, [s[1] for s in surfaces], tag=1)
    gmsh.model.setPhysicalName(2, 1, "fluid")

    curves = gmsh.model.getEntities(dim=1)
    gmsh.model.addPhysicalGroup(1, [c[1] for c in curves], tag=2)
    gmsh.model.setPhysicalName(1, 2, "boundary")

    # mesh
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 0.2)
    gmsh.model.mesh.generate(2)
    gmsh.write("mesh.msh")
    gmsh.finalize()
