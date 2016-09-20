# -*- coding: utf-8 -*-
"""
SHMIP simulation A3. 
"""

from dolfin import *
from plot_rect import *
from pylab import *
import csv

MPI_rank = MPI.rank(mpi_comm_world())
input_file = 'results_abs1/out.hdf5'

#a = zeros((5,5))
a = loadtxt('data.txt', delimiter=",")
xs1 = a[:,0]
Ns1 = a[:,1]
Ns2 = a[:,2]
Ns3 = a[:,3]
qs = a[:,4]
print a





pt = PlotRect(input_file)
Ns = pt.get_int_N(pt.num_steps - 1)

qs1 = pt.get_int_sheet_discharge1(pt.num_steps - 1)


plot(xs1, qs, 'r')
plot(pt.xs, qs1, 'b')
show()
quit()

plot(xs1, Ns1 / 1e6, 'r')
plot(xs1, Ns2 / 1e6, 'r')
plot(xs1, Ns3 / 1e6, 'r')
plot(pt.xs, Ns / 1e6, 'b')

show()

#pt.integrate_y(N)

#plot(N, )