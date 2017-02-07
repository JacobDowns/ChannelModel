from dolfin import *
from dolfin import MPI, mpi_comm_world
import ufl
import numpy as np
from scipy.integrate import ode
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True
parameters["form_compiler"]["precision"] = 100

""" Solves PDE for phi and ODEs for h and S."""

class Solver(object):
  
  def __init__(self, model):
    
    # MPI rank
    self.MPI_rank = MPI.rank(mpi_comm_world())    
    # CG function space
    V_cg = model.V_cg
    V_tr = model.V_tr
    
    ### Get a bunch of fields from the model 
    
    # Hydraulic potential 
    phi = model.phi
    # Potential 1 time step ago
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
    # Bump height
    h_r = model.h_r
    # Conductivity
    k = model.k
    # Initial model time
    t0 = model.t
    # Effective pressure as a function
    N_func = model.N
    
    
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
    
    dS1 = dS(metadata={'quadrature_degree': 2})
    
    ### Expressions used in the variational forms
    
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
    f = (f1 + f2) / (f1 + f2 + Constant(1e-20))      
    # Sensible heat change
    Pi = Constant(0.0)
    if model.use_pi:
      Pi = -Constant(c_t*c_w*rho_w) * (Q + f*Constant(l_c)*q_c) * dpw_ds
    # Channel creep closure rate
    v_c = Constant(A) * S * N**3
    # Another channel source term
    w_c = ((Xi - Pi) / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
    # Define a time step constant
    dt = Constant(1.0)
  
  
    ### First, the variational form for hydraulic potential PDE
        
    U1 = dt * (-dot(grad(theta_cg), q) + (w - v - m)*theta_cg)*dx 
    U2 = dt * (-dot(grad(theta_cg), s)*Q + (w_c - v_c)*theta_cg)('+')*dS1
    C = Constant(e_v/(rho_w * g))
    
    ## First order BDF variational form (backward Euler)    
    F1_phi = C*(phi - phi1)*theta_cg*dx
    F1_phi += U1 + U2
    
    d1_phi = TrialFunction(V_cg)
    J1_phi = derivative(F1_phi, phi, d1_phi)
    
    """
    # Second order BDF variational form
    F2_phi = C*Constant(3.0)*(phi - Constant(4.0/3.0)*phi1 + Constant(1.0/3.0)*phi2)*theta_cg*dx
    F2_phi += Constant(2.0)*(U1 + U2)
    
    d2_phi = TrialFunction(V_cg)
    J2_phi = derivative(F2_phi, phi, d2_phi)"""


    ### Secondly, set up the sheet height and channel area ODEs

    # A form for the rate of change of S
    dS_dt = (((Xi - Pi) / Constant(rho_i * L)) - v_c)

    # Initial sheet height
    h0 = model.h.vector().array()
    # Initial channel areas
    S0 = model.S.vector().array()
    # Length of h vector
    h_len = len(h0) 
    # Bump height vector
    h_r_n = h_r.vector().array()
    # Channel edgle lens
    edge_lens = assemble(theta_tr('+') * dS1 + theta_tr * ds).array()
    # Local mask corresponding to the local array of a tr function that is 1
    # on interior edges and 0 on exterior edges
    local_mask = (assemble(theta_tr('+') * dS1).array() > 1e-15).astype(float)
    # Average q_c over edges
    self.q_c_n = None
    # Average of N**3 over edges
    self.N_cubed_n = None
    # Derivative of phi over edges
    self.dphi_ds_n = None
    # Derivative of pw over edges
    self.dpw_ds_n = None
    # Local array of u_b values
    #self.ub_n
    # Local array of N values
    #self.N_n
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
    
            
    # Alternative right hand side for the sheet height ODE
    def h_rhs(t, h_n) :
      # Ensure that the sheet height is positive
      h_n[h_n < 0.0] = 0.0
      # Sheet opening term
      w_n = u_b.vector().array() * (h_r_n - h_n) / l_r
      # Ensure that the opening term is non-negative
      w_n[w_n < 0.0] = 0.0
      # Sheet closure term
      v_n = A * h_n * N_func.vector().array()**3
      # Return the time rate of change of the sheet
      return (w_n - v_n)
    

    # Right hand side for the channel area ODE
    def S_rhs(t, S_n):
      # Ensure that the channel area is positive
      S_n[S_n < 0.0] = 0.0
      # Along channel flux
      Q_n = -k_c * S_n**alpha * np.abs(self.dphi_ds_n + 1e-20)**delta * self.dphi_ds_n
      # Dissipation melting due to turbulent flux
      Xi_n = np.abs(Q_n * self.dphi_ds_n) + np.abs(l_c * self.q_c_n * self.dphi_ds_n)  
      # Sensible heat change
      f_n = np.logical_or(S_n > 0.0, self.q_c_n * self.dpw_ds_n > 0.0) 
      Pi_n = self.model.use_pi * (-c_t * c_w * rho_w) * (Q_n + f_n*l_c*self.q_c_n) * self.dpw_ds_n 
      # Creep closure
      v_c_n = A * S_n * self.N_cubed_n
      # Total opening rate
      v_o_n = (Xi_n - Pi_n) / (rho_i * L)
      # Calculate rate of channel size change. Loacal mask makes sure exterior channel edges are zero.
      dsdt = local_mask * (v_o_n - v_c_n)
      return dsdt
      
      
    # Right hand side for the channel area ODE (for comparison)
    def S_rhs1(t, S_n):
      S_n[S_n < 0.0] = 0.0
      S.vector().set_local(S_n)
      S.vector().apply("insert")
      dsdt = assemble((dS_dt *  theta_tr)('+') * dS1).array() / edge_lens
      return dsdt
      
      
    # Combined right hand side for h and S
    def rhs(t, Y):         
      Ys = np.split(Y, [h_len])
      h_n = Ys[0]
      S_n = Ys[1]
      
      dhdt = h_rhs(t, h_n)
      dsdt = S_rhs(t, S_n)
      
      return np.hstack((dhdt, dsdt))
      
    
    # ODE solver initial condition
    Y0 = np.hstack((h0, S0))
    # Array of zeros the same langth as the local tr vector
    zs = np.zeros(len(Y0))
    # Set up ODE solver
    ode_solver = ode(rhs).set_integrator('dopri5', max_step = 30.0, atol = 7e-7, rtol = 1e-8, nsteps = 1000)
    ode_solver.set_initial_value(Y0, t0)
      
    
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
    #self.F2_phi = F2_phi
    #self.J2_phi = J2_phi
    self.dt = dt
    self.ode_solver = ode_solver
    self.h_len = h_len
    self.zs = zs
    self.q_c = q_c
    self.edge_lens = edge_lens
    self.theta_tr = theta_tr
    self.N = N
    self.u_b = u_b
    self.Q = Q
    self.Pi = Pi
    self.f = f
    self.dS1 = dS1
    
    
  # Step PDE for phi forward by dt
  def step_phi(self, dt):
    # Assign time step
    self.dt.assign(dt)
    # Solve for potential
    solve(self.F1_phi == 0, self.phi, self.model.d_bcs, J = self.J1_phi, solver_parameters = self.model.newton_params)
    # Update phi
    self.phi1.assign(self.phi)
    self.model.update_phi()  
      
    
    
  # Step combined ode for h and S forward by dt
  def step_ode(self, dt):
    
    # Update fields for use in sheet ODE
    #self.ub_n = self.u_b.vector().array()
    #self.N_n = self.model.N.vector().array()
    # Update fields for use in channel ODE
    self.q_c_n = assemble((self.q_c *  self.theta_tr)('+') * self.dS1).array() / self.edge_lens
    self.N_cubed_n = assemble((self.N**3 *  self.theta_tr)('+') * self.dS1).array() / self.edge_lens
    self.dphi_ds_n = assemble((self.dphi_ds * self.theta_tr)('+') * self.dS1).array() / self.edge_lens
    self.dpw_ds_n = assemble((self.dpw_ds * self.theta_tr)('+') * self.dS1).array() / self.edge_lens
    
    # Step h and S forward
    self.ode_solver.integrate(self.model.t + dt)
    self.ode_solver.y[:] = np.maximum(self.zs, self.ode_solver.y)

    # Retrieve values from the ODE solver    
    Y = np.split(self.ode_solver.y, [self.h_len])
    self.model.h.vector().set_local(Y[0])
    self.model.h.vector().apply("insert")
    self.model.S.vector().set_local(Y[1])
    self.model.S.vector().apply("insert")

    
  # Steps the potential forward by dt
  def step(self, dt):
    # Note : don't change the order of these
    self.step_phi(dt)
    self.step_ode(dt)
    
    
  # Get Q as a tr function
  def get_Q_tr(self):
    self.Q_tr.vector().set_local(assemble((self.Q * self.theta_tr)('+')*self.dS1).array() / self.edge_lens)
    self.Q_tr.vector().apply("insert")
    return self.Q_tr
    
    
  # Get Pi as a tr function
  def get_Pi_tr(self):
    self.Pi_tr.vector().set_local(assemble((self.Pi * self.theta_tr)('+')*self.dS1).array() / self.edge_lens)
    self.Pi_tr.vector().apply("insert")
    return self.Pi_tr
    
    
  # Get dpw_ds as a tr function
  def get_dpw_ds_tr(self):
    self.dpw_ds_tr.vector().set_local(assemble((self.dpw_ds * self.theta_tr)('+')*self.dS1).array() / self.edge_lens)
    self.dpw_ds_tr.vector().apply("insert")
    return self.dpw_ds_tr
    
    
  # Get dpw_ds as a tr function
  def get_dphi_ds_tr(self):
    self.dphi_ds_tr.vector().set_local(assemble((self.dphi_ds * self.theta_tr)('+')*self.dS1).array() / self.edge_lens)
    self.dphi_ds_tr.vector().apply("insert")
    return self.dphi_ds_tr
  
  # Get q_c as a tr fucntion
  def get_q_c_tr(self):
    self.q_c_tr.vector().set_local(assemble((self.q_c * self.theta_tr)('+')*self.dS1).array() / self.edge_lens)
    self.q_c_tr.vector().apply("insert")
    return self.q_c_tr
    
    
  # Get f as a tr fucntion
  def get_f_tr(self):
    self.f_tr.vector().set_local(assemble((self.f * self.theta_tr)('+')*self.dS1).array() / self.edge_lens)
    self.f_tr.vector().apply("insert")
    return self.f_tr
    
  