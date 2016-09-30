from dolfin import *
import numpy as np


### Class for plotting trace elements
class TRPlot(object):
  
  def __init__(self, mesh) :
    self.mesh = mesh
    # Facet function for plotting 
    self.ff_plot = FacetFunctionDouble(mesh)
    # tr function space
    self.V_tr =  FunctionSpace(self.mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))
    # Process
    self.MPI_rank = MPI.rank(mpi_comm_world())
    # Compute a map from local facets to global edge indexs
    self.compute_local_facet_to_global_edge_index_map()
    # Compute a map from local edges to global edge indexes
    self.compute_local_edge_to_global_edge_index_map()
    


    
    
  # Copies a tr function to a facet function
  def copy_tr_to_facet(self, tr, ff) :
    # Gather all edge values from each of the local arrays on each process
    tr_vals = Vector()
    tr.vector().gather(tr_vals, np.array(range(self.V_tr.dim()), dtype = 'intc'))
    # Get the edge values corresponding to each local facet
    local_vals = tr_vals[self.local_facet_to_global_edge_index_map]    
    ff.array()[:] = local_vals

  
  # Compute a map from local facets to indexes in a global tr vector
  def compute_local_facet_to_global_edge_index_map(self):
    # Compute the midpoints of local edges in a tr function   
    edge_coords = self.V_tr.tabulate_dof_coordinates()
    edge_coords = edge_coords.reshape((len(edge_coords)/2), 2)
 
    # Gather a global array of midpoints for a tr function
    fx = Function(self.V_tr)
    fy = Function(self.V_tr)
    fx.vector().set_local(edge_coords[:,0])
    fy.vector().set_local(edge_coords[:,1])
    fx.vector().apply("insert")
    fy.vector().apply("insert") 
    vecx = Vector()
    vecy = Vector()    
    fx.vector().gather(vecx, np.array(range(self.V_tr.dim()), dtype = 'intc'))
    fy.vector().gather(vecy, np.array(range(self.V_tr.dim()), dtype = 'intc'))
    
    
    # x coordinate of midpoint
    global_edge_coords_x = vecx.array()
    # y coordinate of midpoint
    global_edge_coords_y = vecy.array()
    
    if self.MPI_rank == 0:
      print "asf"
      np.savetxt('xs.txt', global_edge_coords_x)

    
    print len(global_edge_coords_x)
    print self.ff_plot.array().size    
    
    # Create a dictionary that maps a midpoint tuple to an index in the global tr vector
    midpoint_to_tr_index = {}
    
    for i in range(len(global_edge_coords_x)):
      x = global_edge_coords_x[i]
      y = global_edge_coords_y[i]
      midpoint_to_tr_index[(x,y)] = i
      
    # Create an array that maps local facet indexes to indexes in the global tr vector
    local_facet_to_global_tr = []
    i = 0
    for f in dolfin.facets(self.mesh):
      x = f.midpoint().x()
      y = f.midpoint().y()
      v0, v1 = f.entities(0)
      
      local_facet_to_global_tr.append(midpoint_to_tr_index[(x,y)])
      i += 1
      
    self.local_facet_to_global_edge_index_map = np.array(local_facet_to_global_tr)
    
    
  # Computes a map from local edges to to indexes in a global tr vector
  def compute_local_edge_to_global_edge_index_map(self):
    f = Function(self.V_tr)
    self.local_edge_to_global_edge_index_map = np.zeros(len(f.vector().array()), dtype = 'intc')
    for i in range(len(f.vector().array())):
      self.local_edge_to_global_edge_index_map[i] = self.V_tr.dofmap().local_to_global_index(i)
  
  
  # Plots a tr function
  def plot_tr(self, tr):
    self.copy_tr_to_facet(tr, self.ff_plot)
    plot(self.ff_plot, interactive = True)
