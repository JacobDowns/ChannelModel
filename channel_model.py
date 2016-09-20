from dolfin import *
from dolfin import MPI, mpi_comm_world
from model import *
from solver import *
import sys as sys

# This is necessary or else assembly of facet integrals will not work in parallel!
#parameters['ghost_mode'] = 'shared_facet'

""" Wrapper class for Glads."""

class ChannelModel(Model):

  def __init__(self, model_inputs):
    Model.__init__(self, model_inputs)

    ### Initialize model variables

    # Function spaces
    self.V_cg = FunctionSpace(self.mesh, "CG", 1)
    
    # Mixed function space
    self.V = self.V_cg * self.V_cg
    # Mixed function combining phi, h
    self.u = Function(self.V)
    # Extract phi, h, and S from u
    (self.phi, self.h) = self.u.split(True)
  
    # Potential at previous time step
    self.phi_prev = Function(self.V_cg)
    # Sheet at previous time step
    self.h_prev = Function(self.V_cg)

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
    if 'd_bcs' in self.model_inputs:
      # Dirichlet boundary conditions
      self.d_bcs = self.model_inputs['d_bcs']    
    else :
      # By default, a marker of 1 denotes the margin
      self.d_bcs = [DirichletBC(self.V.sub(0).sub(0), self.phi_m, self.boundaries, 1)]
      
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
      prm['newton_solver']['maximum_iterations'] = 30
      
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
    
    # Facet functions for plotting CR functions in Paraview    
    self.ff_out_S = FacetFunctionDouble(self.mesh) 
    
    
  # Steps phi and h forward by dt
  def step(self, dt):
    self.solver.step(dt)
    # Update model time
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

      # Read in the initial cavity height
      h_temp = Function(self.V_cg)
      self.input_file.read(h_temp, "h_0")      
      assign(self.h, h_temp)
      
      # Load the initial hydraulic potential
      phi_temp = Function(self.V_cg)
      self.input_file.read(phi_temp, "phi_0")
      assign(self.phi, phi_temp)

      # Load the intitial channel heights
      S_temp = Function(self.V_cr)
      self.input_file.read(S_temp, "S_0")
      assign(self.S, S_temp)
            
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
    self.p_w.vector().set_local(self.phi.vector().array() - self.phi_m.vector().array())
    self.p_w.vector().apply("insert")
    self.update_pfo()
    
  
  # Update the pressure as a fraction of overburden to reflect the current 
  # value of phi
  def update_pfo(self):
    # Compute overburden pressure
    self.pfo.vector().set_local(self.p_w.vector().array() / self.p_i.vector().array())
    self.pfo.vector().apply("insert")
    
  
  # Updates functions derived from phi
  def update_phi(self):
    assign(self.phi_prev, self.phi)
    self.update_pw()
    self.update_N()

    
  # Updates functions derived from h
  def update_h(self):
    assign(self.h_prev, self.h)

    
   # Updates functions derived from S
  def update_S(self):
    assign(self.S_prev, self.S)
    
  
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
      if 'u' in to_write:
        self.u_out << self.phi_solver.u
      if 'm' in to_write:
        self.m_out << self.m
      if 'u_b' in to_write:
        self.u_b_out << self.u_b
      if 'q' in to_write:
        self.q_func.assign(project(self.phi_solver.q, self.V_cg))
        self.q_out << self.q_func
      if 'k' in to_write:
        self.k_out << self.k
      """if 'S' in to_write:
        self.cr_tools.copy_cr_to_facet(self.S, self.ff_out_S)
        self.S_out << self.ff_out_S
      if 'h_cr' in to_write:
        self.cr_tools.copy_cr_to_facet(self.h_cr, self.ff_out_h_cr)
        self.h_cr_out << self.ff_out_h_cr
      if 'N_cr' in to_write:
        self.cr_tools.copy_cr_to_facet(self.N_cr, self.ff_out_N_cr)
        self.N_cr_out << self.ff_out_N_cr"""
      

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
    output_file.write(self.k_cr, "k_c_0")
    output_file.write(self.mask, "mask")
    output_file.write(self.phi, "phi_0")
    
    