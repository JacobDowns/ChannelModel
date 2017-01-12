from dolfin import *
from dolfin import MPI, mpi_comm_world
import ufl
import numpy as np
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

""" Solves phi with h and S fixed."""

class Solver(object):
  
  def __init__(self, model):
    
    # MPI rank
    self.MPI_rank = MPI.rank(mpi_comm_world())    
    
    ### Get function spaces
    V_cg = model.V_cg
    V_tr = model.V_tr
    
    
    ### Get a bunch of fields from the model 
    
    # Hydraulic potential 
    phi = model.phi
    # Potential at previous time step
    phi_prev = model.phi_prev
    # Sheet height
    h = model.h
    # Sheet height at previous time step
    h_prev = Function(V_cg)
    h_prev.assign(h)
    # Channel cross sectional area
    S = model.S
    # Channel cross sectional area at previous time step
    S_prev = Function(V_tr)
    S_prev.assign(S)
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
    phi_reg = Constant(1e-15)
    
  
    ### Define variational forms for each unknown phi, h, and S
  
    # Test functions 
    theta_cg = TestFunction(V_cg)
    theta_tr = TestFunction(V_tr)
    
    
    ### Expressions used in the variational forms
    
    # Expression for effective pressure
    N = phi_0 - phi
    # Flux vector
    q = -k * h**alpha * (dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # Opening term 
    #w = conditional(gt(h_r - h, 0.0), u_b * (h_r - h) / Constant(l_r), 0.0)
    w = u_b * (h_r - h) / Constant(l_r)    
    # Closing term
    v = Constant(A) * h * N**3    
    # Normal and tangent vectors 
    n = FacetNormal(model.mesh)
    t = as_vector([n[1], -n[0]])
    # Derivative of phi along channel 
    dphi_ds = dot(grad(phi), t)
    # Discharge through channels
    Q = -Constant(k_c) * abs(S)**alpha * abs(dphi_ds + phi_reg)**delta * dphi_ds
    # Approximate discharge of sheet in direction of channel
    q_c = -k * h**alpha * abs(dphi_ds + phi_reg)**delta * dphi_ds
    # Energy dissipation 
    Xi = abs(Q * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)
    # Derivative of water pressure along channels
    dpw_ds = dot(grad(phi - phi_m), t)
    # Switch to turn refreezing on or of
    f = conditional(gt(S,0.0),1.0,0.0)
    # Sensible heat change
    Pi = -Constant(c_t * c_w * rho_w) * (Q + f * Constant(l_c) * q_c) * dpw_ds
    #Pi = Constant(0.0)
    # Channel creep closure rate
    v_c = Constant(A) * S * N**3
    # Another channel source term
    w_c = ((Xi - Pi) / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
    # Define a time step constant
    dt = Constant(1.0)
  
  
    ### First the variational form for hydraulic potential PDE
  
    F_phi = Constant(e_v / (rho_w * g)) * (phi - phi_prev) * theta_cg * dx
    F_phi += dt * (-dot(grad(theta_cg), q) + (w - v - m) * theta_cg) * dx 
    F_phi += dt * (-dot(grad(theta_cg), t) * Q + (w_c - v_c) * theta_cg)('+') * dS
    #F_phi = (-dot(grad(theta_cg), q) + (w - v - m) * theta_cg) * dx 
    
    d_phi = TrialFunction(V_cg)
    J_phi = derivative(F_phi, phi, d_phi) 
    
    # Setup the nonlinear problem for the hydraulic potential
    phi_problem = NonlinearVariationalProblem(F_phi, phi, model.d_bcs, J = J_phi)
    phi_problem.set_bounds(model.phi_m, model.phi_0)
    
    snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "lu",
                                          "maximum_iterations": 500,
                                          "report": True,
                                          "error_on_nonconvergence": False, 
                                          "relative_tolerance" : 1e-7,
                                          "absolute_tolerance" : 1e-6}}
    
    phi_solver = NonlinearVariationalSolver(phi_problem)
    phi_solver.parameters.update(snes_solver_parameters)


    ### Variational form for the sheet height ODE
    F_h = ((h - h_prev) - (w - v) * dt) * theta_cg * dx
    
    d_h = TrialFunction(V_cg)
    J_h = derivative(F_h, h, d_h) 

    
    ### Variational form for the channel cross sectional area ODE
    F_S = (((S - S_prev) - (((Xi - Pi) / Constant(rho_i * L)) - v_c) * dt) * theta_tr)('+') * dS
    # Add a term so the Jacobian is non-singular
    F_S += S * theta_tr * ds
    
    d_S = TrialFunction(V_tr)
    J_S = derivative(F_S, S, d_S) 
    
    """
    # Setup the nonlinear problem for S
    S_problem = NonlinearVariationalProblem(F_S, S, J = J_S)
    S_problem.set_bounds(interpolate(Constant(0.0), V_tr), interpolate(Constant(500.0), V_tr))
    
    
    S_solver = NonlinearVariationalSolver(S_problem)
    S_solver.parameters.update(snes_solver_parameters)"""
    
    
    ### Assign local variables
    
    self.phi = phi
    self.h = h
    self.S = S
    self.q = q
    self.F_phi = F_phi
    self.F_h = F_h
    self.F_S = F_S
    self.J_phi = J_phi
    self.J_h = J_h
    self.J_S = J_S
    self.model = model
    self.dt = dt
    self.phi_solver = phi_solver
    self.S_prev = S_prev
    #self.S_solver = S_solver
    
   
  ### Step PDE for phi forward by dt. Constrain using SNES solver. 
  def step_phi_constrained(self, dt):
    self.dt.assign(dt)
    # Solve for potential
    (i, converged) = self.phi_solver.solve()
    # Update phi
    self.model.update_phi()  
    
    
  ### Step PDE for phi forward by dt
  def step_phi(self, dt):
    self.dt.assign(dt)
    # Solve for potential
    solve(self.F_phi == 0, self.phi, self.model.d_bcs, J = self.J_phi, solver_parameters = self.model.newton_params)
    # Update phi
    self.model.update_phi()   
    
    
  ### Step ODE for h forward by dt
  def step_h(self, dt):
    self.dt.assign(dt)
    # Solve ODE
    solve(self.F_h == 0, self.h, J = self.J_h, solver_parameters = self.model.newton_params)
    
    # Local h vals
    h_vals = self.h.vector().array()
    indexes_under = h_vals < 0.0
    # Set values to positive
    h_vals[indexes_under] = 0.0
    # Update global vector
    self.h.vector().set_local(h_vals)
    self.h.vector().apply("insert")
    
    # Update h
    self.model.update_h()
    
    
  ### Step PDE for S forward by dt
  def step_S(self, dt):    
    
    prm = NonlinearVariationalSolver.default_parameters()
    prm['newton_solver']['relaxation_parameter'] = 1.0
    prm['newton_solver']['relative_tolerance'] = 1e-6
    prm['newton_solver']['absolute_tolerance'] = 1e-6
    prm['newton_solver']['error_on_nonconvergence'] = False
    prm['newton_solver']['maximum_iterations'] = 3
      
    S_dt = 60.0 * 1.0
    
    if dt < S_dt:
      dts = [dt]
    else :
      dts = np.repeat(S_dt, dt / S_dt)
      if not (dt % S_dt) == 0:
        dts = np.append(dts, dt % S_dt)
        
    local_time = self.model.t
    
    for dt in dts:
      if self.MPI_rank == 0:
        print '%sLocal time: %s %s' + str(local_time) 
        
      self.dt.assign(dt)
      # Solve ODE
      solve(self.F_S == 0, self.S, J = self.J_S, solver_parameters = prm)    
      
      # Local S vals
      S_vals = self.S.vector().array()
      indexes_under = S_vals < 0.0
      # Set values to positive
      S_vals[indexes_under] = 0.0
      # Update global vector
      self.S.vector().set_local(S_vals)
      self.S.vector().apply("insert")
      
      # Update phi
      self.S_prev.assign(self.S)
      
      local_time += dt


  # Steps the potential forward by dt
  def step(self, dt):
    # Note : order matters!
    self.step_phi(dt)
    
    #File('pfo1.pvd') << self.model.pfo
    #quit()
    # Step s forward by dt
    self.step_S(dt)
    # Step h forward by dt
    #self.step_h(dt)
    
    File('S3.xml') << self.model.S
  