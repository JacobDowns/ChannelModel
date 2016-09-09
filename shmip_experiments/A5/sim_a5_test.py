# -*- coding: utf-8 -*-
"""
SHMIP simulation A5. 
"""

from dolfin import *
from constants import *
from channel_model import *
from dolfin import MPI, mpi_comm_world
from scale_functions import *
from plot_tools import *

MPI_rank = MPI.rank(mpi_comm_world())
input_file = '../inputs/A5/inputs_A5.hdf5'
out_dir = 'results3'


### Setup the model

model_inputs = {}
model_inputs['input_file'] = input_file
model_inputs['out_dir'] = out_dir
model_inputs['constants'] = pcs

pt = PlotTools('results/out.hdf5')

# Create the sheet model
model = ChannelModel(model_inputs)

spd = pcs['spd']
dt = spd / 4.0

for i in range(pt.num_steps):
  if MPI_rank == 0: 
    current_time = model.t / spd
    print ('%sCurrent time: %s %s' % (fg(1), current_time, attr(0)))
    
  phi = pt.get_phi(i)
  
  #plot(phi)
  model.phi.vector().set_local(phi.vector().array())
  model.phi.vector().apply("insert")
  
  model.update_phi()
  model.step(dt)
  
  model.write_pvds(['pfo', 'h'])
  model.checkpoint(['h', 'phi', 'S', 'k', 'N'])

