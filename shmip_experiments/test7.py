from dolfin import *
from dolfin import MPI, mpi_comm_world
import numpy as np 

parameters['ghost_mode'] = 'shared_vertex'

# Process number
MPI_rank = MPI.rank(mpi_comm_world())

mesh = UnitSquareMesh(3,3)
V_cg = FunctionSpace(mesh, 'Lagrange', 1) 
f = Function(V_cg)

prm = NonlinearVariationalSolver.default_parameters()

print dir(prm)

#quit()

#File('f.xml') >> f