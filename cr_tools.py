from dolfin import *
import numpy as np

class CRTools(object):
  
  def __init__(self, mesh) :
    self.mesh = mesh
    # CR function space
    self.V_cr = V_cr
    # Process
    self.MPI_rank = MPI.rank(mpi_comm_world())
    # Compute a map from local facets to global edge indexs
    self.compute_local_facet_to_global_edge_index_map()
    # Compute a map from local edges to global edge indexes
    self.compute_local_edge_to_global_edge_index_map()
    

    # Facet function for plotting 
    self.ff_plot = FacetFunctionDouble(mesh)
    
    
  # Copies a CR function to a facet function
  def copy_cr_to_facet(self, cr, ff) :
    # Gather all edge values from each of the local arrays on each process
    cr_vals = Vector()
    cr.vector().gather(cr_vals, np.array(range(self.V_cr.dim()), dtype = 'intc'))
    # Get the edge values corresponding to each local facet
    local_vals = cr_vals[self.local_facet_to_global_edge_index_map]    
    ff.array()[:] = local_vals

  
  # Compute a map from local facets to indexes in a global cr vector
  def compute_local_facet_to_global_edge_index_map(self):
    # Compute the midpoints of local edges in a cr function   
    edge_coords = self.V_cr.tabulate_dof_coordinates()
    edge_coords = edge_coords.reshape((len(edge_coords)/2), 2)
 
    # Gather a global array of midpoints for a cr function
    fx = Function(self.V_cr)
    fy = Function(self.V_cr)
    fx.vector().set_local(edge_coords[:,0])
    fy.vector().set_local(edge_coords[:,1])
    fx.vector().apply("insert")
    fy.vector().apply("insert")
    vecx = Vector()
    vecy = Vector()    
    fx.vector().gather(vecx, np.array(range(self.V_cr.dim()), dtype = 'intc'))
    fy.vector().gather(vecy, np.array(range(self.V_cr.dim()), dtype = 'intc'))
    
    # x coordinate of midpoint
    global_edge_coords_x = vecx.array()
    # y coordinate of midpoint
    global_edge_coords_y = vecy.array()
    
    # Create a dictionary that maps a midpoint tuple to an index in the global cr vector
    midpoint_to_cr_index = {}
    
    for i in range(len(global_edge_coords_x)):
      x = global_edge_coords_x[i]
      y = global_edge_coords_y[i]
      midpoint_to_cr_index[(x,y)] = i
      
    # Create an array that maps local facet indexes to indexes in the global cr vector
    local_facet_to_global_cr = []
    i = 0
    for f in dolfin.facets(self.mesh):
      x = f.midpoint().x()
      y = f.midpoint().y()
      v0, v1 = f.entities(0)
      
      local_facet_to_global_cr.append(midpoint_to_cr_index[(x,y)])
      i += 1
      
    self.local_facet_to_global_edge_index_map = np.array(local_facet_to_global_cr)
    
    
  # Computes a map from local edges to to indexes in a global cr vector
  def compute_local_edge_to_global_edge_index_map(self):
    f = Function(self.V_cr)
    self.local_edge_to_global_edge_index_map = np.zeros(len(f.vector().array()), dtype = 'intc')
    for i in range(len(f.vector().array())):
      self.local_edge_to_global_edge_index_map[i] = self.V_cr.dofmap().local_to_global_index(i)
  
  
  # Plots a CR function
  def plot_cr(self, cr):
    self.copy_cr_to_facet(cr, self.ff_plot)
    plot(self.ff_plot, interactive = True)
