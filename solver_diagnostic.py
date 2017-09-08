from dolfin import *
from dolfin import MPI, mpi_comm_world
import ufl
import numpy as np
from petsc4py import PETSc

parameters["form_compiler"]["precision"] = 100

# Some kind of hack necessary to get conditionals to work
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

""" Solves PDE for phi and ODEs for h and S."""

class Solver(object):

  def __init__(self, model):

    # MPI rank
    self.MPI_rank = MPI.rank(mpi_comm_world())
    # CG function space
    V_cg = model.V_cg
    V_tr = model.V_tr
    self.V_cg2 = FunctionSpace(model.mesh, "CG", 3)

    ### Get a bunch of fields from the model

    # Hydraulic potential
    phi = model.phi
    # Potential 2 time steps ago
    phi1 = Function(V_cg)
    phi1.assign(phi)
    # Sheet height
    h = model.h
    # Channel cross sectional area
    S = model.S
    # Get melt rate
    m = model.m
    # Basal sliding speed
    u_b = model.u_b
    # Potential
    phi = model.phi
    # Potential at 0 pressure
    phi_m = model.phi_m
    # Potential at overburden pressure
    phi_0 = model.phi_0
    # Constraints
    phi_min = model.phi_min
    phi_max = model.phi_max
    # Bump height
    h_r = model.h_r
    # Conductivity
    k = model.k
    # tr version of Q for output
    self.Q_tr = Function(V_tr)
    # tr version of Pi for output
    self.Pi_tr = Function(V_tr)
    # tr version of dpw_ds for output
    self.dpw_ds_tr = Function(V_tr)
    # tr version of dphi_ds for output
    self.dphi_ds_tr = Function(V_tr)
    # tr version of q_c for output
    self.q_c_tr = Function(V_tr)
    # tr version of f for output
    self.f_tr = Function(V_tr)
    # Alternative guess for newton solver
    self.zero_guess = Function(V_cg)
    self.zero_guess.assign(phi_0)



    ### Define constants

    # Density of ice
    rho_i = model.pcs['rho_i']
    # Density of water
    rho_w = model.pcs['rho_w']
    # Rate factor
    A = model.pcs['A']
    # Channel conductivity
    k_c = model.pcs['k_c']
    # Distance between bumps
    l_r = model.pcs['l_r']
    # Sheet width under channel
    l_c = model.pcs['l_c']
    # Latent heat
    L = model.pcs['L']
    # Void storage ratio
    e_v = model.pcs['e_v']
    # Gravitational acceleration
    g = model.pcs['g']
    # Specific heat capacity of ice
    c_w = model.pcs['c_w']
    # Pressure melting coefficient
    c_t = model.pcs['c_t']
    # Exponents
    alpha = model.pcs['alpha']
    delta = model.pcs['delta']
    # Regularization parameter
    phi_reg = Constant(1e-20)


    ### Define variational forms for each unknown phi, h, and S

    # Test functions
    theta_cg = TestFunction(V_cg)
    theta_tr = TestFunction(V_tr)
    # Used for midpoint quadratures
    self.dS1 = dS(metadata={'quadrature_degree': 1})

    self.storage = False
    if not e_v == 0:
      self.storage = True

    # Expression for effective pressure
    N = phi_0 - phi
    # Flux vector
    q = -k*(h**alpha)*(dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # Opening term
    w = conditional(gt(h_r - h, 0.0), u_b*(h_r - h)/Constant(l_r), 0.0)
    #w = u_b*(h_r - h)/Constant(l_r)
    # Closing term
    v = Constant(A) * h * N**3
    # Normal and tangent vectors
    n = FacetNormal(model.mesh)
    s = as_vector([n[1], -n[0]])
    # Derivative of phi along channel
    dphi_ds = dot(grad(phi), s)
    # Discharge through channels
    Q = -Constant(k_c) * S**alpha * abs(dphi_ds + phi_reg)**delta * dphi_ds
    # Approximate discharge of sheet in direction of channel
    q_c = -k * (h**alpha) * abs(dphi_ds + phi_reg)**delta * dphi_ds
    # Energy dissipation
    Xi = abs(Q * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)
    # Derivative of water pressure along channels
    dpw_ds = dot(grad(phi - phi_m), s)
    # Switch to turn refreezing on or of
    f1 = conditional(gt(S,0.0),1.0,0.0)
    f2 = conditional(gt(q_c * dpw_ds, 0.0), 1.0, 0.0)
    f = (f1 + f2) / (f1 + f2 + Constant(1e-16))
    # Sensible heat change
    Pi = Constant(0.0)
    if model.use_pi:
      Pi = Constant(-c_t*c_w*rho_w) * (Q + f*Constant(l_c)*q_c) * dpw_ds
    # Channel creep closure rate
    v_c = Constant(A) * S * N**3
    # Another channel source term
    w_c = ((Xi - Pi) / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
    # Define a time step constant
    dt = Constant(1.0)
    # Storage constant
    C = Constant(e_v/(rho_w * g))


    ### First, the variational form for hydraulic potential PDE

    # Sheet and channel components of variational form
    U1 = (-dot(grad(theta_cg), q) + (w - v - m)*theta_cg)*dx
    U2 = (-dot(grad(theta_cg), s)*Q + (w_c - v_c)*theta_cg)('+')*dS

    if self.storage :
      F1_phi = (C*(phi - phi1)/dt)*theta_cg*dx
      F1_phi += U1
      if model.use_channels:
        F1_phi += U2
    else :
      F1_phi = U1
      if model.use_channels:
        F1_phi += U2

    d1_phi = TrialFunction(V_cg)
    J1_phi = derivative(F1_phi, phi, d1_phi)


    ### Setup constrained solver

    phi_problem1 = NonlinearVariationalProblem(F1_phi, phi, model.d_bcs, J1_phi)
    phi_problem1.set_bounds(phi_min, phi_max)
    phi_solver1 = NonlinearVariationalSolver(phi_problem1)
    phi_solver1.parameters.update(model.snes_params)


    ### Secondly, set up the sheet height and channel area ODEs

    # Bump height vector
    h_r_n = h_r.vector().array()
    # Channel edgle lens
    self.edge_lens = assemble(theta_tr('+') * dS + theta_tr * ds).array()
    # Local mask corresponding to the local array of a tr function that is 1
    # on interior edges and 0 on exterior edges
    local_mask = (assemble(theta_tr('+') * dS).array() > 1e-15).astype(float)
    # h as petsc4py vec (metaphysically linked to h)
    self.h_v = as_backend_type(h.vector()).vec()
    # S as petsc4py vec (metaphysically linked to S)
    self.S_v = as_backend_type(S.vector()).vec()
    # MPI communicator
    comm = model.mesh.mpi_comm().tompi4py()
    # Array of zeros the same langth as the local cg vector
    self.zs_cg = np.zeros(len(h.vector().array()))
    # Array of zeros the same langth as the local tr vector
    self.zs_tr = np.zeros(len(S.vector().array()))


    # Object that encapsulates h ODE for PETSc
    class H_ODE(object):

      def __init__(self):
        #  Part of  opening term that doesn't depend on h
        self.v_o_0 = None
        # Part of closure term that doesn't depend on h
        self.v_c_0 = None

      def rhs(self, ts, t, h, dhdt):
        h_n = h.getArray()
        # Sheet opening term
        v_o_n = self.v_o_0 * (h_r_n - h_n)
        # Ensure that the opening term is non-negative
        v_o_n[v_o_n < 0.0] = 0.0
        # Sheet closure term
        v_c_n = self.v_c_0 * h_n
        # Set right hand side
        dhdt.setArray(v_o_n - v_c_n)


    # Create PETSc time stepping solver for h
    h_ode = H_ODE()
    h_ode_solver = PETSc.TS().create(comm=comm)
    h_ode_solver.setType(h_ode_solver.Type.RK)
    h_ode_solver.setRHSFunction(h_ode.rhs)
    h_ode_solver.setTime(0.0)
    h_ode_solver.setInitialTimeStep(0.0, 1.0)
    h_ode_solver.setTolerances(atol=1e-10, rtol=1e-13)
    h_ode_solver.setMaxSteps(200)
    h_ode_solver.setExactFinalTime(h_ode_solver.ExactFinalTimeOption.MATCHSTEP)


    # Object that encapsulates S ODE for PETSc
    class S_ODE(object):

      def __init__(self):
        # Derivative of phi along edges
        self.dphi_ds_n = None
        # Derivative of p_W along channels
        self.dpw_ds_n = None
        # N**3 at channel midpoint
        self.N_cubed_n = None
        # Sheet sflux under channel
        self.q_c_n = None

      def rhs(self, ts, t, S, dSdt):
        S_n = S.getArray()
        # Ensure that the channel area is positive
        S_n[S_n < 0.0] = 0.0
        # Flux
        Q_n = -k_c * S_n**alpha * np.abs(self.dphi_ds_n + 1e-20)**delta * self.dphi_ds_n
        # Dissipation melting due to turbulent flux
        Xi_n = np.abs(Q_n * self.dphi_ds_n) + np.abs(l_c * self.q_c_n * self.dphi_ds_n)
        # Sensible heat change
        f_n = np.logical_or(S_n > 0.0, self.q_c_n * self.dpw_ds_n > 0.0)
        #f_n = 0.0
        Pi_n = model.use_pi * (-c_t * c_w * rho_w) * (Q_n + f_n*l_c*self.q_c_n) * self.dpw_ds_n
        # Creep closure
        v_c_n = A * S_n * self.N_cubed_n
        # Total opening rate
        v_o_n = (Xi_n - Pi_n) / (rho_i * L)
        # Calculate rate of channel size change. Local mask makes sure exterior channel edges are zero.
        dSdt.setArray(local_mask * (v_o_n - v_c_n))


    # Create PETSc time stepping solver for S
    S_ode = S_ODE()
    S_ode_solver = PETSc.TS().create(comm=comm)
    S_ode_solver.setType(S_ode_solver.Type.RK)
    S_ode_solver.setRHSFunction(S_ode.rhs)
    S_ode_solver.setTime(0.0)
    S_ode_solver.setInitialTimeStep(0.0, 1.0)
    S_ode_solver.setTolerances(atol=2e-10, rtol=1e-13)
    S_ode_solver.setMaxSteps(10000)
    S_ode_solver.setExactFinalTime(S_ode_solver.ExactFinalTimeOption.MATCHSTEP)


    ### Assign local variables

    self.model = model
    self.phi = phi
    self.phi1 = phi1
    self.h = h
    self.S = S
    self.q = q
    self.dpw_ds = dpw_ds
    self.dphi_ds = dphi_ds
    self.F1_phi = F1_phi
    self.J1_phi = J1_phi
    self.dt = dt
    self.q_c = q_c
    self.theta_tr = theta_tr
    self.N = N
    self.u_b = u_b
    self.Q = Q
    self.Pi = Pi
    self.f = f
    self.h_ode_solver = h_ode_solver
    self.S_ode_solver = S_ode_solver
    self.S_ode = S_ode
    self.h_ode = h_ode
    self.phi_solver1 = phi_solver1
    # Effective pressure as function
    self.N_func = model.N
    self.n = n
    self.dh_dt = w - v
    self.R = div(q) + w - v - m
    self.k = k
    self.alpha = alpha
    self.delta = delta

    self.ds = Measure('ds', domain = model.mesh, subdomain_data = model.boundaries)


  # Step PDE for phi forward by dt
  def step_phi(self, dt, constrain = False):
    # Assign time step
    self.dt.assign(dt)
    success = False

    try :
      if not constrain :
        # Solve for potential
        solve(self.F1_phi == 0, self.phi, self.model.d_bcs, J = self.J1_phi, solver_parameters = self.model.newton_params)
      else :
        (i, converged) = self.phi_solver1.solve()
    except :
      # Take two half steps
      if not constrain :
        self.model.newton_params['newton_solver']['error_on_nonconvergence'] = False
        self.model.newton_params['newton_solver']['maximum_iterations'] = 50
        self.model.newton_params['newton_solver']['relaxation_parameter'] = 0.8
        solve(self.F1_phi == 0, self.phi, self.model.d_bcs, J = self.J1_phi, solver_parameters = self.model.newton_params)
        self.model.newton_params['newton_solver']['maximum_iterations'] = 30
        self.model.newton_params['newton_solver']['relaxation_parameter'] = 0.95
        self.model.newton_params['newton_solver']['error_on_nonconvergence'] = True
      else :
        self.model.snes_params['snes_solver']['error_on_nonconvergence'] = False
        self.phi_solver1.parameters.update(self.model.snes_params)
        (i, converged) = self.phi_solver1.solve()
        self.model.snes_params['snes_solver']['error_on_nonconvergence'] = True
        self.phi_solver1.parameters.update(self.model.snes_params)




    dhdt = assemble(self.dh_dt*dx)
    #R1 = assemble()
    imo = assemble(self.model.m*dx) - assemble(dot(self.q, self.n)*self.ds(1) + dot(self.q, self.n)*self.ds(0))
    R = assemble(self.F1_phi)

    #print assemble(div(self.q)*dx)
    q2t = project(self.q, VectorFunctionSpace(self.model.mesh, 'CG', 2))
    #print assemble((div(q2t) + self.dh_dt - self.model.m)*dx)


    print assemble(dot(self.q, self.n)*ds)
    print assemble(div(q2t)*dx)
    quit()
    print assemble(dot(q2, self.n)*ds )
    #print assemble(dot(q2, self.n)*ds)

    print assemble(div(q2)*dx)
    quit()
    print assemble(self.R*dx)
    f = f.vector().array()
    print
    print R.sum()
    quit()
    #vec = as_vector([-1.0, 0.0])
    #thing = project(dot(vec, self.q), self.model.V_cg)
    #File('thing.pvd') << self.width_integrate(thing)

    Rsum = R.sum()

    Ro = Function(self.model.V_cg)
    Ro.vector().set_local(R.array())
    Ro.vector().apply("insert")
    File('R.pvd') << Ro


    if self.MPI_rank == 0:
        print "RSum: " + str(Rsum)
        print dhdt
        print imo
    print

    # Update phi1
    self.phi1.assign(self.phi)
    # Update fields derived from phi
    self.model.update_phi()


  # Step odes for h and S forward by dt
  def step_ode(self, dt):

    ### S ODE

    if self.model.use_channels:
      # Update fields for use in channel ODE
      self.S_ode.q_c_n = assemble((self.q_c *  self.theta_tr)('+') * self.dS1).array() / self.edge_lens
      self.S_ode.N_cubed_n = assemble((self.N**3 *  self.theta_tr)('+') * self.dS1).array() / self.edge_lens
      self.S_ode.dphi_ds_n = assemble((self.dphi_ds * self.theta_tr)('+') * dS).array() / self.edge_lens
      self.S_ode.dpw_ds_n = assemble((self.dpw_ds * self.theta_tr)('+') * dS).array() / self.edge_lens

      # Step S forward
      self.S_ode_solver.setTime(0.0)
      self.S_ode_solver.setMaxTime(dt)
      self.S_ode_solver.solve(self.S_v)

      if self.MPI_rank == 0:
        print('steps %d (%d rejected)'
              % (self.S_ode_solver.getStepNumber(), self.S_ode_solver.getStepRejections()))

      # Make sure it's non-negative
      self.S.vector().set_local(np.maximum(self.S.vector().array(), self.zs_tr))
      # Apply changes to vector
      self.S.vector().apply("insert")


    ### h ODE

    # Precompute parts of the rhs for h that don't depend on h, but do depend on other unknowns
    self.h_ode.v_o_0 = self.u_b.vector().array() / self.model.pcs['l_r']
    self.h_ode.v_c_0 = self.model.pcs['A'] * self.N_func.vector().array()**3

    # Step h forward
    self.h_ode_solver.setTime(0.0)
    self.h_ode_solver.setMaxTime(dt)
    self.h_ode_solver.solve(self.h_v)

    if self.MPI_rank == 0:
      print('steps %d (%d rejected)'
            % (self.h_ode_solver.getStepNumber(), self.h_ode_solver.getStepRejections()))

    # Make sure it's non-negative
    self.h.vector().set_local(np.maximum(self.h_v.getArray(), self.zs_cg))
    # Apply changes to vector
    self.h.vector().apply("insert")


  # Steps the potential forward by dt
  def step(self, dt, constrain = False):
    # Note : don't change the order of these
    self.step_phi(dt, constrain)
    self.step_ode(dt)


  # Get Q as a tr function
  def get_Q_tr(self):
    self.Q_tr.vector().set_local(assemble((self.Q * self.theta_tr)('+')*dS).array() / self.edge_lens)
    self.Q_tr.vector().apply("insert")
    return self.Q_tr


  # Get Pi as a tr function
  def get_Pi_tr(self):
    self.Pi_tr.vector().set_local(assemble((self.Pi * self.theta_tr)('+')*dS).array() / self.edge_lens)
    self.Pi_tr.vector().apply("insert")
    return self.Pi_tr


  # Get dpw_ds as a tr function
  def get_dpw_ds_tr(self):
    self.dpw_ds_tr.vector().set_local(assemble((self.dpw_ds * self.theta_tr)('+')*dS).array() / self.edge_lens)
    self.dpw_ds_tr.vector().apply("insert")
    return self.dpw_ds_tr


  # Get dpw_ds as a tr function
  def get_dphi_ds_tr(self):
    self.dphi_ds_tr.vector().set_local(assemble((self.dphi_ds * self.theta_tr)('+')*dS).array() / self.edge_lens)
    self.dphi_ds_tr.vector().apply("insert")
    return self.dphi_ds_tr

  # Get q_c as a tr fucntion
  def get_q_c_tr(self):
    self.q_c_tr.vector().set_local(assemble((self.q_c * self.theta_tr)('+')*dS).array() / self.edge_lens)
    self.q_c_tr.vector().apply("insert")
    return self.q_c_tr


  # Get f as a tr fucntion
  def get_f_tr(self):
    self.f_tr.vector().set_local(assemble((self.f * self.theta_tr)('+')*dS).array() / self.edge_lens)
    self.f_tr.vector().apply("insert")
    return self.f_tr