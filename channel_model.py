from dolfin import *
from dolfin import MPI, mpi_comm_world
from model import *
from solver import *
import sys as sys

# This is necessary or else assembly of facet integrals will not work in parallel!
parameters['ghost_mode'] = 'shared_facet'

""" Wrapper class for Glads."""

class ChannelModel(Model):

  def __init__(self, model_inputs):
    Model.__init__(self, model_inputs)

    ### Initialize model variables

    ## Function spaces
    # Typical CG FEM space
    self.V_cg = FunctionSpace(self.mesh, "CG", 1)
    # Space of trace elements that have constant values on facets
    self.V_tr = FunctionSpace(self.mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))

    ## Primary model unknowns
    # Hydraulic potential    
    self.phi = Function(self.V_cg)
    # Potential at previous time step
    self.phi_prev = Function(self.V_cg)
    # Sheet height
    self.h = Function(self.V_cg)
    # Channel cross sectional area
    self.S = Function(self.V_tr)

    ## Other fields
    # Bed geometry
    self.B = Function(self.V_cg)
    # Ice thickness
    self.H = Function(self.V_cg)
    # Melt input
    self.m = Function(self.V_cg)
    # Basal sliding speed
    self.u_b = Function(self.V_cg)
    # Hydraulic conductivity
    self.k = Function(self.V_cg)
    # Bump height 
    self.h_r = Function(self.V_cg)
    # Boundary facet function
    self.boundaries = FacetFunction("size_t", self.mesh)
    # Effective pressure
    self.N = Function(self.V_cg)
    # Water pressure
    self.p_w = Function(self.V_cg)
    # Pressure as a fraction of overburden
    self.pfo = Function(self.V_cg)
    # Function for visualizing flux
    self.V_cg_vec = VectorFunctionSpace(self.mesh, "CG", 1)
    self.q_func = Function(self.V_cg_vec)
    # Normal and tangent vectors 
    self.n = FacetNormal(model.mesh)
    self.t = as_vector([n[1], -n[0]])
    # Derivative of phi along channel s
    self.dphi_ds = dot(grad(phi), t)
    # Derivative of p_w along channel s
    self.dpw_ds = dot(grad(p_w), t)

    ## Stuff used to compute fields along channel edges 
    # Sheet height at channel midpoints
    self.h_tr = Function(self.V_tr)
    # Directional derivatives of potential along channel edges
    self.dphi_ds_tr = Function(self.V_tr)
    # Directional derivatives of water pressure along channel edges
    self.dpw_ds_tr = Function(self.V_tr)
    # A tr test function
    self.v_tr = TestFunction(self.V_tr)
    # Vector of edge lengths
    self.edge_lens = assemble(self.v_tr * dS)
    
    
    # Load inputs and initialize input and output files    
    self.init_model()
    
    
    ### Derive some additional fields
    
    # Potential at 0 pressure
    self.phi_m = project(pcs['rho_w'] * pcs['g'] * self.B, self.V_cg)
    # Ice overburden pressure
    self.p_i = project(pcs['rho_i'] * pcs['g'] * self.H, self.V_cg)
    # Potential at overburden pressure
    self.phi_0 = project(self.phi_m + self.p_i, self.V_cg)
    
    # Populate all fields derived from the primary variables
    self.update_phi()
    self.update_h()


    ### Setup boundary conditions
    
    # If there are boundary conditions specified, use them. Otherwise apply
    # default bc of 0 pressure on the margin
    
    self.d_bcs = []    
    if 'd_bcs' in self.model_inputs:
      # Dirichlet boundary conditions
      self.d_bcs = self.model_inputs['d_bcs']    
    elif 'point_bcs' in self.model_inputs:
      # Dictionary entry should be a list of subdomains
      for i in range(len(self.model_inputs['point_bcs'])):
        # Get a subdomain
        sub = self.model_inputs['point_bcs'][i]
        # Apply 0 pressure at the point / points specified by the subdomain
        bc = DirichletBC(self.V_cg, self.phi_m, sub, "pointwise")
        self.d_bcs.append(bc)
    else :
      # By default, a marker of 1 denotes the margin
      self.d_bcs.append(DirichletBC(self.V_cg, self.phi_m, self.boundaries, 1))
      
    #plot(self.boundaries, interactive = True)
      
    ### Newton solver parameters
      
    # If the Newton parameters are specified use them. Otherwise use some
    # defaults
    if 'newton_params' in self.model_inputs :
      self.newton_params = self.model_inputs['newton_params']
    else :
      prm = NonlinearVariationalSolver.default_parameters()
      prm['newton_solver']['relaxation_parameter'] = 1.0
      prm['newton_solver']['relative_tolerance'] = 1e-6
      prm['newton_solver']['absolute_tolerance'] = 1e-6
      prm['newton_solver']['error_on_nonconvergence'] = False
      prm['newton_solver']['maximum_iterations'] = 20
      
      self.newton_params = prm
      
      
    ### Create objects that solve the model equations
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    self.solver = Solver(self)
      
    ### Create output files
    
    # Output directory
    self.h_out = File(self.out_dir + "h.pvd")
    self.S_out = File(self.out_dir + "S.pvd")
    self.phi_out = File(self.out_dir + "phi.pvd")
    self.N_out = File(self.out_dir + "N.pvd")
    self.pfo_out = File(self.out_dir + "pfo.pvd")
    self.m_out = File(self.out_dir + "m.pvd")
    self.u_b_out = File(self.out_dir + "u_b.pvd")
    self.k_out = File(self.out_dir + "k.pvd")
    self.q_out = File(self.out_dir + "q.pvd")
    
    # Facet functions for plotting CR functions in Paraview    
    self.ff_out_S = FacetFunctionDouble(self.mesh) 
    
    
  # Steps phi and h forward by dt
  def step(self, dt):
    # Update model time
    self.solver.step(dt)
    self.t += dt
    
  
  # Look at the input file to check if we're starting or continuing a simulation
  def to_continue(self):
    try :
      # Check if this is a simulation output file. If it is it should have an 
      # attribute called h
      self.input_file.attributes("h")
      return True
    except :
      return False
  
    
  # Initialize the model state for a new simulation
  def start_simulation(self):
    self.load_inputs()

    # Write the initial conditions. These can be used as defaults to initialize
    # the model if a simulation crashes and we want to start it again
    self.output_file.write(self.h, "h_0")
    self.output_file.write(self.m, "m_0")
    self.output_file.write(self.u_b, "u_b_0")
    self.output_file.write(self.phi, "phi_0")
    self.output_file.write(self.mesh, "mesh")
    self.output_file.write(self.B, "B")
    self.output_file.write(self.H, "H")
    self.output_file.write(self.boundaries, "boundaries")
    self.output_file.write(self.S, "S_0")    
    self.output_file.write(self.k, "k_0")
    self.output_file.write(interpolate(Constant(self.pcs['k_c']), self.V_cg), "k_c_0") 
    
  
  # Initialize model using the state at the end of a previous simulation
  def continue_simulation(self):
     self.load_inputs()
     
     # Load the most recent cavity height and channel height values
     num_steps = self.input_file.attributes("h")['count']
     h_last = "h/vector_" + str(num_steps - 1)
     S_last = "S/vector_" + str(num_steps - 1)
     phi_last = "phi/vector_" + str(num_steps - 1)
      
     self.input_file.read(self.h, h_last)
     self.input_file.read(self.S, S_last)
     self.input_file.read(self.phi, phi_last)
      
     # Get the start time for the simulation        
     attr = self.input_file.attributes(h_last)
     self.t = attr['timestamp']
     
     
  # Load all model inputs from the input file if they are not specified
  # in the
  def load_inputs(self):
    try :
      # Bed elevation 
      self.input_file.read(self.B, "B")
      # Ice thickness
      self.input_file.read(self.H, "H")    
      # Boundary facet function
      self.input_file.read(self.boundaries, "boundaries")
      # Melt input
      self.input_file.read(self.m, "m_0")
      # Sliding speed
      self.input_file.read(self.u_b, "u_b_0")
      # Hydraulic conductivity
      self.input_file.read(self.k, "k_0")     
      # Load the initial hydraulic potential
      self.input_file.read(self.phi, "phi_0")
      # Read in the initial cavity height
      self.input_file.read(self.h, "h_0")      
      # Load the intitial channel area 
      #self.input_file.read(self.S, "S_0")
      self.S.assign(interpolate(Constant(1e-10), self.V_tr))
      # Use the default constant bump height
      self.h_r.assign(interpolate(Constant(self.pcs['h_r']), self.V_cg))
      
    except Exception as e:
      # If we can't find one of these model inputs we're done 
      print >> sys.stderr, e
      print >> sys.stderr, "Could not load model inputs."
      sys.exit(1)
      
      
  # Update the effective pressure to reflect current value of phi
  def update_N(self):
    self.N.vector().set_local(self.phi_0.vector().array() - self.phi.vector().array())
    self.N.vector().apply("insert")
    
  
  # Update the water pressure to reflect current value of phi
  def update_pw(self):
    # Compute water pressure
    self.p_w.vector().set_local(self.phi.vector().array() - self.phi_m.vector().array())
    self.p_w.vector().apply("insert")
    # Update derivativews of pw along channel edges
    self.dpw_ds_tr.set_local(assemble((self.dpw_ds * self.v_tr)('+')) / self.edge_lens.array())
    self.dpw_ds_tr.apply("insert")
    # Update pressure as fraction of overburden 
    self.update_pfo()
    
  
  # Update the pressure as a fraction of overburden to reflect the current 
  # value of phi
  def update_pfo(self):
    self.pfo.vector().set_local(self.p_w.vector().array() / self.p_i.vector().array())
    self.pfo.vector().apply("insert")
    
  
  # Updates functions derived from phi
  def update_phi(self):
    # Update derivatives of phi along channel edges
    self.dphi_ds_tr.vector().set_local(assemble((self.dphi_ds * self.v_tr)('+') * dS).array() / self.edge_lens.array())
    self.dphi_ds_tr.vector().apply("insert")    
    # Update phi_prev for backward Euler time stepping
    assign(self.phi_prev, self.phi)
    # Update water pressure
    self.update_pw()
    # Update effective pressure
    self.update_N()
    
    
  # Updates functions derived from h
  def update_h(self):
    self.h_tr.vector().set_local(assemble((self.h * self.v_tr)('+') * dS).array() / self.edge_lens.array())
    self.h_tr.vector().apply("insert")    
    
    
  # Correct the potential so that it is above 0 pressure and below overburden.
  # Return True if underpressure was present?
  def phi_apply_bounds(self):
    
    # Array of values for phi
    phi_vals = self.phi.vector().array()
    # Array of minimum values
    phi_max_vals = self.phi_0.vector().array()
    # Array of maximum values
    phi_min_vals = self.phi_m.vector().array()
    
    # Indexes in the array of phi vals that are overpressure
    indexes_over = phi_vals > phi_max_vals + 1e-3
    # Indexes that are underpressure
    indexes_under = phi_vals < phi_min_vals - 1e-3    
    
    phi_vals[indexes_over] = phi_max_vals[indexes_over]
    phi_vals[indexes_under] = phi_min_vals[indexes_under]
  
    # Update phi    
    self.phi.vector().set_local(phi_vals)
    self.phi.vector().apply("insert")
    
    # If there were over or underpressure values return true
    if indexes_over.any() or indexes_under.any():
      return True
    
    # If pressure was in the correct range return false
    return False
    
  
  # Write fields to pvd files for visualization
  def write_pvds(self, to_write = []):
    to_write = set(to_write)
    if len(to_write) == 0:
      self.h_out << self.h
      self.pfo_out << self.pfo
    else:
      if 'h' in to_write:
        self.h_out << self.h
      if 'phi' in to_write:
        self.h_out << self.h
      if 'N' in to_write:
        self.N_out << self.N
      if 'pfo' in to_write:
        self.pfo_out << self.pfo
      if 'm' in to_write:
        self.m_out << self.m
      if 'u_b' in to_write:
        self.u_b_out << self.u_b
      if 'q' in to_write:
        self.q_func.assign(project(self.solver.q, self.V_cg_vec))
        self.q_out << self.q_func
      if 'k' in to_write:
        self.k_out << self.k
      

  # Write checkpoint files to an hdf5 file
  def checkpoint(self, to_write = [], label = None):
    to_write = set(to_write)

    # Always checkpoint h and S
    self.output_file.write(self.h, "h", self.t)
    self.output_file.write(self.S, "S", self.t)
    self.output_file.write(self.phi, "phi", self.t)

    if 'u_b' in to_write:
      self.output_file.write(self.u_b, "u_b", self.t)
    if 'k' in to_write:
      self.output_file.write(self.k, "k", self.t)
    if 'm' in to_write:
      self.output_file.write(self.m, "m", self.t)
    if 'pfo' in to_write:
      self.output_file.write(self.pfo, "pfo", self.t)
      
    self.output_file.flush()
    
  
  # Sets the melt rate function
  def set_m(self, new_m):
    self.m.assign(new_m)
    
  
  # Sets the sliding speed
  def set_u_b(self, new_u_b):
    self.u_b.assign(new_u_b)
    
    
  # Sets the hydraulic conductivity
  def set_k(self, new_k):
    self.k.assign(project(new_k, self.V_cg))
    
  
  # Write out a steady state file we can use to start new simulations
  def write_steady_file(self, output_file_name):
    output_file = HDF5File(mpi_comm_world(), output_file_name + '.hdf5', 'w')

    ### Write variables
    output_file.write(self.mesh, "mesh")
    output_file.write(self.B, "B")
    output_file.write(self.H, "H")
    output_file.write(self.m, 'm_0')
    output_file.write(self.u_b, 'u_b_0')
    output_file.write(self.h, "h_0")
    output_file.write(self.S, "S_0")
    output_file.write(self.boundaries, "boundaries")
    output_file.write(self.k, "k_0")
    output_file.write(self.phi, "phi_0")
    
    