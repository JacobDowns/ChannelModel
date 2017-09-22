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
    V_dg = model.V_dg
    V_tr = model.V_tr
    V_vec = model.V_vec

    ### Get a bunch of fields from the model

    # Hydraulic potential
    phi = model.phi
    # Potential last time step
    phi1 = Function(V_cg)
    phi1.assign(phi)
    # Sheet height
    h = model.h
    h_cg = model.h_cg
    # h last step
    h1 = Function(V_dg)
    h1.assign(h)
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
    phi_reg = Constant(1e-16)


    ### Define variational forms for each unknown phi, h, and S

    # Test functions
    theta_cg = TestFunction(V_cg)
    theta_tr = TestFunction(V_tr)
    # Used for midpoint quadratures
    self.dS1 = dS(metadata={'quadrature_degree': 1})
    self.ds1 = Measure('ds', domain = model.mesh, subdomain_data = model.boundaries)


    self.storage = False
    if not e_v == 0:
      self.storage = True

    #phi = phi #+ Constant(rho_w * g)*h

    # Expression for effective pressure
    N = phi_0 - phi
    # Flux vector

    q = -k*(h**alpha)*(dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # Opening term
    w = conditional(gt(h_r - h, 0.0), u_b*(h_r - h)/Constant(l_r), 0.0)
    # Closing term
    v = Constant(A) * h * N**3
    # Normal and tangent vectors
    n = FacetNormal(model.mesh)
    # Define a time step constant
    dt = Constant(1.0)
    # Storage constant
    C = Constant(e_v/(rho_w * g))


    ### Variational form for hydraulic potential PDE

    # Sheet and channel components of variational form
    U1 = (-dot(grad(theta_cg), q) + (w - v - m)*theta_cg)*dx

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



    ### Variational form for conservation equation

    # DG test function
    theta_dg = TestFunction(V_dg)
    # Vector part of the flux (really, just everything but h**alpha)
    q_vec = -k * (dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # q_vec as a function
    q_vec_func = Function(V_vec)
    # Continuous flux for DG
    q_h = h**alpha * q_vec_func
    # Derivative of flux wrt h
    def Max(a, b): return (a+b+sqrt((a-b + Constant(1e-16))**2))/Constant(2.)
    dq_du = Constant(alpha) * Max(h('+'), h('-'))**(alpha - 1.0) * q_vec_func('+')
    W = abs(dot(dq_du, n('+')))
    # LF numerical flux
    q_star = avg(q_h) + (W / Constant(2.0))*n('+')*(h('+')-h('-'))
    # Outward cell Normal
    n_hat = (theta_dg('+')*n('+') + theta_dg('-')*n('-'))
    # Time derivative of h
    dt_h = Constant(1.0)
    dh_dt = (h - h1)/dt_h
    # Swtich to prevent any minor inflow
    outflow_switch = Function(V_tr)
    outflow_switch.interpolate(Constant(1.0))

    F_h =  (dh_dt*theta_dg - dot(q, grad(theta_dg)))*dx
    F_h -= m*theta_dg*dx
    F_h += dot(n_hat, q_star)*dS
    F_h += outflow_switch*dot(q_h, n)*theta_dg*ds

    self.inflow = outflow_switch*dot(q_h, n)*theta_tr*ds

    d_h = TrialFunction(V_dg)
    J_h = derivative(F_h, h, d_h)


    ### Setup solvers

    phi_problem1 = NonlinearVariationalProblem(F1_phi, phi, model.d_bcs, J1_phi)
    phi_problem1.set_bounds(phi_min, phi_max)
    phi_solver1 = NonlinearVariationalSolver(phi_problem1)
    phi_solver1.parameters.update(model.snes_params)


    h_min = interpolate(Constant(0.0), V_dg)
    h_max = interpolate(Constant(100.0), V_dg)
    h_problem = NonlinearVariationalProblem(F_h, h, model.h_bcs, J_h)
    h_problem.set_bounds(h_min, h_max)
    h_solver = NonlinearVariationalSolver(h_problem)
    h_solver.parameters.update(model.snes_params)


    ### Assign local variables

    self.model = model
    self.dt_h = dt_h
    self.phi = phi
    self.phi1 = phi1
    self.h = h
    self.h1 = h1
    self.q = q
    self.q_vec = q_vec
    self.q_vec_func = q_vec_func
    self.F1_phi = F1_phi
    self.J1_phi = J1_phi
    self.dt = dt
    self.theta_tr = theta_tr
    self.N = N
    self.u_b = u_b
    self.phi_solver1 = phi_solver1
    self.h_solver = h_solver
    #self.h_solver = h_solver
    # Effective pressure as function
    self.N_func = model.N
    self.n = n
    self.theta_tr = theta_tr
    self.outflow_switch = outflow_switch
    self.phi_min = phi_min
    self.phi_max = phi_max



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
        """
        print "sab"
        File('checkis/B.pvd') << self.model.B
        File('checkis/H.pvd') << self.model.H
        File('checkis/m.pvd') << self.model.m
        File('checkis/u_b.pvd') << self.model.u_b
        File('checkis/k.pvd') << self.model.k
        File('checkis/phi_m.pvd') << self.model.phi_m
        File('checkis/phi_0.pvd') << self.model.phi_0
        quit()"""

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

    if not constrain:
      print "sadf"
      self.constrain_phi()

    # Update fields derived from phi
    self.model.update_phi()
    self.q_vec_func.assign(project(self.q_vec, self.model.V_vec))
    File('test2/q_vec_func.pvd') << self.q_vec_func


  def constrain_phi(self):
    phi_vals = self.phi.vector().array()
    indexes_over = phi_vals > self.phi_max.vector().array()
    indexes_under = phi_vals < self.phi_min.vector().array()
    phi_vals[indexes_over] = self.phi_max.vector().array()[indexes_over]
    phi_vals[indexes_under] = self.phi_min.vector().array()[indexes_under]
    self.phi.vector().set_local(phi_vals)
    self.phi.vector().apply("insert")


  def constrain_h(self):
    h_vals = self.h.vector().array()
    indexes_over = h_vals > 20.0
    h_vals[indexes_over] = 20.0
    self.h.vector().set_local(h_vals)
    self.h.vector().apply("insert")


  # Step odes for h and S forward by dt
  def step_ode(self, dt):
    self.dt_h.assign(dt)
    ### Step forward, ODE for h
    self.outflow_switch.interpolate(Constant(1.0))
    inflow_indexes = assemble(self.inflow).array() < 0.0
    switch_vals = np.ones(len(self.outflow_switch.vector().array()))
    switch_vals[inflow_indexes] = 0.0
    self.outflow_switch.vector().set_local(switch_vals)
    self.outflow_switch.vector().apply("insert")
    print assemble(self.inflow).array().min()

    (i, converged) = self.h_solver.solve()
    self.constrain_h()
    self.h1.assign(self.h)


  # Steps the potential forward by dt
  def step(self, dt, constrain = False):
    # Note : don't change the order of these
    self.step_phi(dt, constrain)
    self.step_ode(dt)
