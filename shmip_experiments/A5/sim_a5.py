# -*- coding: utf-8 -*-
"""
SHMIP simulation A5. 
"""

from dolfin import *
from constants import *
from channel_model import *
from dolfin import MPI, mpi_comm_world
from scale_functions import *

MPI_rank = MPI.rank(mpi_comm_world())
#input_file = '../inputs/A5/inputs_A5.hdf5'
input_file = 'results_new1/steady.hdf5'
out_dir = 'results_new2'


### Setup the model

model_inputs = {}
model_inputs['input_file'] = input_file
model_inputs['out_dir'] = out_dir
model_inputs['constants'] = pcs

prm = NonlinearVariationalSolver.default_parameters()
prm['newton_solver']['relaxation_parameter'] = 1.0
prm['newton_solver']['relative_tolerance'] = 1e-6
prm['newton_solver']['absolute_tolerance'] = 1e-6
prm['newton_solver']['error_on_nonconvergence'] = False
prm['newton_solver']['maximum_iterations'] = 25


#prm["newton_solver"]["linear_solver"] = "gmres"
#prm["nonlinear_solver"] = "snes"
#prm['linear_solver'] = 'gmres'
#prm['preconditioner'] = 'ilu'
#prm['krylov_solver']['absolute_tolerance'] = 1E-9
#prm['krylov_solver']['relative_tolerance'] = 1E-7
#prm['krylov_solver']['maximum_iterations'] = 1000
#prm['krylov_solver']['gmres']['restart'] = 40
#prm['krylov_solver']['preconditioner']['ilu']['fill_level'] = 0

#model_inputs['newton_params'] = prm



# Create the sheet model
model = ChannelModel(model_inputs)


### Run the simulation

# Seconds per day
spd = pcs['spd']
# End time
T = 500.0 * spd
# Time step
dt = spd / 2.0
# Iteration count
i = 0

while model.t < T:  
  if MPI_rank == 0: 
    current_time = model.t / spd
    print ('%sCurrent time: %s %s' % (fg(1), current_time, attr(0)))
  
  model.step(dt)
  
 # if MPI_rank == 0: 
  print model.S.vector().max()
  
  if i % 1 == 0:
    model.write_pvds(['pfo', 'h'])
    
  if i % 1 == 0:
    model.checkpoint(['h', 'phi', 'S', 'k'])
  
  if MPI_rank == 0: 
    print
    
  i += 1
  
model.write_steady_file(out_dir + '/steady2')
