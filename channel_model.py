from dolfin import *
from dolfin import MPI, mpi_comm_world
from model import *
from solver import *
import sys as sys

# This is necessary or else assembly of facet integrals will not work in parallel!
parameters['ghost_mode'] = 'shared_facet'

""" Wrapper class for Glads."""

#parameters["form_compiler"]["cpp_optimize"] = True

class ChannelModel(Model):

  def __init__(self, model_inputs):
    Model.__init__(self, model_inputs)

    ### Initialize model

    # Typical CG FEM space
    self.V_cg = FunctionSpace(self.mesh, "CG", 1)
    # DG function space
    self.V_dg = FunctionSpace(self.mesh, "DG", 0)
    # Space of trace elements that have constant values on facets
    self.V_tr = FunctionSpace(self.mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))
    # Vector function space for flux
    self.V_vec = VectorFunctionSpace(self.mesh, "CG", 2)

    ## Primary model unknowns

    # Hydraulic potential
    self.phi = Function(self.V_cg)
    # Sheet height as DG function
    self.h = Function(self.V_dg)
    # Sheet height as a CG function
    self.h_cg = Function(self.V_cg)

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
    # Constraints
    self.phi_min = Function(self.V_cg)
    self.phi_max = Function(self.V_cg)
    # Storage thickness
    self.h_e = Function(self.V_cg)


    # Load inputs and initialize input and output files
    self.init_model()


    ### Bounds for constrained solver

    # If the minimum potential is specified, then use it.
    self.phi_min.assign(self.phi_m)
    if 'phi_min' in self.model_inputs:
      self.phi_min.assign(self.model_inputs['phi_min'])

    # If the maximum potential is specified, then use it.
    self.phi_max.assign(project(self.phi_m + Constant(1.0)*self.p_i, self.V_cg))
    if 'phi_max' in self.model_inputs:
      self.phi_max.assign(self.model_inputs['phi_max'])


    ### Setup boundary conditions

    # If there are boundary conditions specified, use them. Otherwise apply
    # default bc of 0 pressure on the margin
    self.d_bcs = []
    if 'point_bc' in self.model_inputs:
      self.d_bcs = [DirichletBC(self.V_cg, self.phi_m, self.model_inputs['point_bc'], method="pointwise")]
    elif 'd_bcs' in self.model_inputs:
      # Dirichlet boundary conditions
      self.d_bcs = self.model_inputs['d_bcs']
    else :
      # By default, a marker of 1 denotes the margin
      self.d_bcs.append(DirichletBC(self.V_cg, self.phi_m, self.boundaries, 1))


    ### Boundary condition for hyperbolic problem for h
    self.h_bcs = [DirichletBC(self.V_dg, Constant(0.0), self.boundaries, 2, "geometric")]


    ### Newton solver parameters

    # If the Newton parameters are specified use them. Otherwise use some
    # defaults
    if 'newton_params' in self.model_inputs :
      self.newton_params = self.model_inputs['newton_params']
    else :
      prm = NonlinearVariationalSolver.default_parameters()
      prm['newton_solver']['relaxation_parameter'] = 1.0
      prm['newton_solver']['relative_tolerance'] = 1e-12
      prm['newton_solver']['absolute_tolerance'] = 5e-8
      prm['newton_solver']['error_on_nonconvergence'] = False
      prm['newton_solver']['maximum_iterations'] = 30

      self.newton_params = prm

    ### SNES solver aprameters

    if 'snes_params' in self.model_inputs:
      self.snes_params = self.model_inputs['snes_params']
    else :
      self.snes_params = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                           "maximum_iterations": 100,
                           "line_search": "basic",
                           "report": True,
                           "error_on_nonconvergence": False,
                           "relative_tolerance" : 1e-11,
                           "absolute_tolerance" : 1e-7}}



    ### Flag to turn off channels
    self.use_channels = False


    ### Create object that solves the model equations

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
    self.q_n_out = File(self.out_dir + "q_n.pvd")
    self.q_mag_out = File(self.out_dir + "q_mag.pvd")
    self.h_e_out = File(self.out_dir + "h_e.pvd")


  # Steps phi and h forward by dt
  def step(self, dt, constrain = False):
    # Update model time
    self.solver.step(dt, constrain)
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
    self.output_file.write(self.k, "k_0")
    self.output_file.write(interpolate(Constant(self.pcs['k_c']), self.V_cg), "k_c_0")


  # Initialize model using the state at the end of a previous simulation
  def continue_simulation(self):
     self.load_inputs()

     # Load the most recent cavity height and channel height values
     num_steps = self.input_file.attributes("h")['count']
     h_last = "h/vector_" + str(num_steps - 1)
     phi_last = "phi/vector_" + str(num_steps - 1)

     self.input_file.read(self.h, h_last)
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
      self.input_file.read(self.h, "h_0")
      self.h_cg.assign(self.h)
      # Use the default constant bump height
      self.h_r.assign(interpolate(Constant(self.pcs['h_r']), self.V_cg))

      ### Derive variables we'll need later

      # Potential at 0 pressure
      self.phi_m = project(pcs['rho_w'] * pcs['g'] * self.B, self.V_cg)
      # Ice overburden pressure
      self.p_i = project(pcs['rho_i'] * pcs['g'] * self.H, self.V_cg)
      # Potential at overburden pressure
      self.phi_0 = project(self.phi_m + self.p_i, self.V_cg)


      ### Load the initial potential function, if there is one
      has_phi_0 = False
      try :
        self.input_file.attributes("phi_0")
        has_phi_0 = True
      except :
        pass

      if has_phi_0:
        self.input_file.read(self.phi, "phi_0")
      else :
        # No initial potential specified, use overburden potential
        self.phi.assign(self.phi_0)


      ### Load phi_min function if there is one
      has_phi_min = False
      try :
         self.input_file.attributes("phi_min")
      except :
        pass

      if has_phi_min:
        self.input_file.read(self.phi_min, "phi_min")
      else :
        self.phi_min.assign(self.phi_m)


      ### Load phi_max function if there is one
      has_phi_max = False
      try :
         self.input_file.attributes("phi_max")
         has_phi_max = True
      except :
        pass

      if has_phi_max:
        print "here"
        self.input_file.read(self.phi_max, "phi_max")
      else :
        self.phi_max.assign(self.phi_0)


      # Update phi
      self.update_phi()

    except Exception as e:
      # If we can't find one of these model inputs we're done
      print >> sys.stderr, e
      print >> sys.stderr, "Could not load model inputs."
      sys.exit(1)


  # Update the effective pressure to reflect current value of phi
  def update_N(self):
    self.N.assign(project(self.phi_0 - self.phi, self.V_cg))


  # Update the water pressure to reflect current value of phi
  def update_pw(self):
    # Compute water pressure
    self.p_w.assign(project(self.phi - self.phi_m, self.V_cg))
    # Update pressure as fraction of overburden
    self.update_pfo()
    # Update storage
    self.update_h_e()


  # Update the pressure as a fraction of overburden to reflect the current
  # value of phi
  def update_pfo(self):
    self.pfo.assign(project(self.p_w / self.p_i, self.V_cg))


  # Update water storage to reflect current pressure
  def update_h_e(self):
    e_v = self.pcs['e_v']
    rho_w = self.pcs['rho_w']
    g = self.pcs['g']
    self.h_e.assign(project(Constant(e_v / (rho_w * g)) * self.p_w, self.V_cg))


  # Updates functions derived from phi
  def update_phi(self):
    # Update water pressure
    self.update_pw()
    # Update effective pressure
    self.update_N()


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
        self.q_out << project(self.solver.q, self.V_vec)
      if 'q_n' in to_write:
        n = as_vector([-1.0, 0.0])
        self.q_n_out << project(dot(self.solver.q, n), self.V_cg)
      if 'q_mag' in to_write:
        self.q_mag_out << project(sqrt(dot(self.solver.q, self.solver.q)), self.V_cg)
      if 'k' in to_write:
        self.k_out << self.k
      if 'h_e' in to_write:
        self.h_e_out << self.h_e


  # Write checkpoint files to an hdf5 file
  def checkpoint(self, to_write = [], label = None):
    to_write = set(to_write)

    # Always checkpoint h
    self.output_file.write(self.h, "h", self.t)

    if 'N' in to_write:
      self.output_file.write(self.phi, "N", self.t)
    if 'u_b' in to_write:
      self.output_file.write(self.u_b, "u_b", self.t)
    if 'k' in to_write:
      self.output_file.write(self.k, "k", self.t)
    if 'm' in to_write:
      self.output_file.write(self.m, "m", self.t)
    if 'pfo' in to_write:
      self.output_file.write(self.pfo, "pfo", self.t)
    if 'q' in to_write:
      self.output_file.write(project(self.solver.q, self.V_vec), "q", self.t)
    if 'q_mag' in to_write:
      q_mag = project(sqrt(dot(self.solver.q, self.solver.q)), self.V_cg)
      self.output_file.write(q_mag, "q", self.t)
    if 'q_n' in to_write:
      n = as_vector([-1.0, 0.0])
      q_n = project(dot(self.solver.q, n), self.V_cg)
      self.output_file.write(q_n, "q_n", self.t)
    if 'h_e' in to_write:
      self.output_file.write(self.h_e, "h_e", self.t)

    self.output_file.flush()


  # Sets the melt rate function
  def set_m(self, new_m):
    self.m.assign(new_m)


  # Sets sheet height function
  def set_h(self, new_h):
    self.h.assign(project(new_h,  self.V_cg))


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
    output_file.write(self.boundaries, "boundaries")
    output_file.write(self.k, "k_0")
    output_file.write(self.phi, "phi_0")
    output_file.write(self.N, "N_0")
    q_mag = project(sqrt(dot(self.solver.q, self.solver.q)), self.V_cg)
    output_file.write(q_mag, 'q_0')
    n = as_vector([-1.0, 0.0])
    q_n = project(dot(self.solver.q, n), self.V_cg)
    output_file.write(q_n, 'q_n')
