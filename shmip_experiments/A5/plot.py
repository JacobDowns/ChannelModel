# -*- coding: utf-8 -*-
"""
SHMIP simulation A3. 
"""

from dolfin import *
from plot_rect import *
from pylab import *
import csv
from tr_plot import *

MPI_rank = MPI.rank(mpi_comm_world())
input_file = 'results_new2/steady2.hdf5'

#a = zeros((5,5))
a = loadtxt('data.txt', delimiter=",")
xs1 = a[:,0]
Ns1 = a[:,1]
Ns2 = a[:,2] 
Ns3 = a[:,3]
#qs = a[:,4]
#print a





pt = PlotRect(input_file)
#dolfin.plot(pt.get_N(5000), interactive = True)


S = pt.get_S(0)

"""
tr_plot = TRPlot(pt.mesh)
ff = FacetFunctionDouble(pt.mesh)

tr_plot.copy_tr_to_facet(S, ff)

File('out.pvd') << ff

quit()
ff = FacetFunctionDouble(pt.mesh)

ff.array()[:] = S.vector().array()
#dolfin.plot(ff, interactive = True)
File('out.pvd') << ff
print S

coords = pt.V_cr.tabulate_dof_coordinates()

print coords"""

for i in range(100):
  print pt.V_cr.dofmap().tabulate_facet_dofs(i)
  

  
quit()

Ns = pt.get_int_N(pt.num_steps - 2)

#qs1 = pt.get_int_sheet_discharge1(pt.num_steps - 1)


#plot(xs1, qs, 'r')
#plot(pt.xs, qs1, 'b')
#show()
#quit()
xs1 /= 1000.0
xs2 = pt.xs / 1000.0


title('Mean Effective Pressure (averaged over width of domain)')
plot(xs1, Ns1 / 1e6, 'ko-', label = "Mauro's Output")
#plot(xs1, Ns2 / 1e6, 'r')
#plot(xs1, Ns3 / 1e6, 'r')
plot(xs2, Ns / 1e6, 'r', label = "Fenics Output")

xlabel('x (km)')
ylabel('MPa')
xlim([min(xs1), max(xs1)])
legend()

show()

#pt.integrate_y(N)

#plot(N, )