# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:12:11 2016

@author: jake
"""

from dolfin import *
from dolfin import MPI, mpi_comm_world
from numpy import *

MPI_rank = MPI.rank(mpi_comm_world())


# Directory to write model inputs
#in_dir = "inputs/"
#mesh = Mesh(in_dir + "mesh/mesh.xml")

mesh = UnitSquareMesh(3, 3)
V_cg = FunctionSpace(mesh, "CG", 1)
V_cr = FunctionSpace(mesh, "CR", 1)






g = Function(V_cr)
File('g.xml') >> g
f = Function(V_cg)

#plot(mesh, interactive = True)

#print (MPI_rank, g.vector().array())
#print (MPI_rank, V_cr.tabulate_dof_coordinates().reshape(len(g.vector().array()), 2))

#quit()


v = TestFunction(V_cg)
print (MPI_rank, assemble(g * v * dx).array())
#print (MPI_rank, V_cg.tabulate_dof_coordinates().reshape(len(f.vector().array()), 2))    

quit()
print (MPI_rank, g.vector().array())
print (MPI_rank, V_cr.tabulate_dof_coordinates())

quit()
f = Function(V_cg)
File('f.xml') >> f

print (MPI_rank, f.vector().array().reshape())
"""
print (MPI_rank, mesh.coordinates())
f = Function(V_cg)
f.set_local()

plot(f, interactive = True)"""

#print V_cr.tabulate_dof_coordinates()                      
 

                                      
                                                                                                  
