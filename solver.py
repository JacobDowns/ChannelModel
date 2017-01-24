from dolfin import *
from dolfin import MPI, mpi_comm_world
import ufl
import numpy as np
from scipy.integrate import ode
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

""" Solves PDE for phi and ODEs for h and S."""

class Solver(object):
  
  def __init__(self, model):
    
    # MPI rank
    self.MPI_rank = MPI.rank(mpi_comm_world())    
    # CG function space
    V_cg = model.V_cg
    
    ### Get a bunch of fields from the model 
    
    # Hydraulic potential 
    phi = model.phi
    # Potential at previous time step
    phi_prev = model.phi_prev
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
    # phi and pw edge derivatives
    dphi_ds_tr = model.dphi_ds_tr
    dpw_ds_tr = model.dpw_ds_tr
    # h, k, and N on edge derivatives
    k_tr = model.k_tr
    N_tr = model.N_tr
    h_tr = model.h_tr
    # Local mask corresponding to 2500 the local array of a tr function that is 1
    # on interior edges and 0 on exterior edges
    local_mask = model.local_mask
    # Array of zeros the same langth as the local tr vector
    zs = np.zeros(len(S.vector().array()))
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
    phi_reg = Constant(1e-16)
    
  
    ### Define variational forms for each unknown phi, h, and S
  
    # Test functions 
    theta_cg = TestFunction(V_cg)
    theta_tr = model.v_tr
    
    ### Expressions used in the variational forms
    
    # Expression for effective pressure
    N = phi_0 - phi
    # Flux vector
    q = -k * h**alpha * (dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # Opening term 
    w = conditional(gt(h_r - h, 0.0), u_b * (h_r - h) / Constant(l_r), 0.0)
    #w = u_b * (h_r - h) / Constant(l_r)    
    # Closing term
    v = Constant(A) * h * N**3    
    # Normal and tangent vectors 
    n = FacetNormal(model.mesh)
    t = as_vector([n[1], -n[0]])
    # Derivative of phi along channel 
    dphi_ds = dot(grad(phi), t)
    # Discharge through channels
    Q = -Constant(k_c) * S**alpha * abs(dphi_ds + phi_reg)**delta * dphi_ds
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
    # Channel creep closure rate
    v_c = Constant(A) * S * N**3
    # Another channel source term
    w_c = ((Xi - Pi) / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
    # Define a time step constant
    dt = Constant(1.0)
  
  
    ### First, the variational form for hydraulic potential PDE
  
    F_phi = Constant(e_v / (rho_w * g)) * (phi - phi_prev) * theta_cg * dx
    F_phi += dt * (-dot(grad(theta_cg), q) + (w - v - m) * theta_cg) * dx 
    F_phi += dt * (-dot(grad(theta_cg), t) * Q + (w_c - v_c) * theta_cg)('+') * dS
    
    d_phi = TrialFunction(V_cg)
    J_phi = derivative(F_phi, phi, d_phi)


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
    edge_lens = model.edge_lens
    
            
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
      # Get effective pressures, sheet thickness on edges.
      N_n = N_tr.vector().array()
      # Get midpoint values of sheet thickness
      h_n = h_tr.vector().array()
      # Array form of the derivative of the potential 
      phi_s = dphi_ds_tr.vector().array()  
      # Along channel flux
      Q_n = -k_c * S_n**alpha * np.abs(phi_s + 1e-16)**delta * phi_s
      # Flux of sheet under channel
      q_n = -k_tr.vector().array() * h_n**alpha * np.abs(phi_s + 1e-16)**delta * phi_s
      # Dissipation melting due to turbulent flux
      Xi_n = np.abs(Q_n * phi_s) + np.abs(l_c * q_n * phi_s)
      # Switch to turn refreezing on or off
      f_n = S_n > 0.0
      # Sensible heat change
      Pi_n = (-c_t * c_w * rho_w) * (Q_n + f_n * l_c * q_n) * dpw_ds_tr.vector().array()
      # Creep closure
      v_c_n = A * S_n * N_n**3
      # Total opening rate
      v_o_n = (Xi_n -Pi_n) / (rho_i * L)
      # Calculate rate of channel size change
      dsdt = local_mask * (v_o_n - v_c_n)
      return dsdt
      
      
    # Right hand side for the channel area ODE (for comparison)
    def S_rhs1(t, S_n):
      print t
      S_n[S_n < 0.0] = 0.0
      S.vector().set_local(S_n)
      S.vector().apply("insert")
      dsdt = assemble((dS_dt *  theta_tr)('+') * dS).array() / edge_lens.array()
      return dsdt
      
      
    # Combined right hand side for h and S
    def rhs(t, Y):        
      #if self.MPI_rank == 0:      
      #  print (self.MPI_rank, t)
      
      Ys = np.split(Y, [h_len])
      h_n = Ys[0]
      S_n = Ys[1]
      
      dhdt = h_rhs(t, h_n)
      dsdt = S_rhs(t, S_n)
      
      return np.hstack((dhdt, dsdt))
      
    
    # ODE solver initial condition
    Y0 = np.hstack((h0, S0))
    # Set up ODE solver
    ode_solver = ode(rhs).set_integrator('vode', method = 'adams', max_step = 20.0, atol = 1e-6, rtol = 1e-8)
    ode_solver.set_initial_value(Y0, t0)
      
    
    ### Assign local variables
    
    self.phi = phi
    self.h = h
    self.S = S
    self.q = q
    self.F_phi = F_phi
    self.J_phi = J_phi
    self.model = model
    self.dt = dt
    self.ode_solver = ode_solver
    self.h_len = h_len
    self.zs = zs
    
    
  # Step PDE for phi forward by dt
  def step_phi(self, dt):
    self.dt.assign(dt)
    # Solve for potential
    solve(self.F_phi == 0, self.phi, self.model.d_bcs, J = self.J_phi, solver_parameters = self.model.newton_params)
    # Update phi
    self.model.update_phi()   
    
    
  # Step combined ode for h and S forward by dt
  def step_ode(self, dt):
    
    # Step h and S forward
    self.ode_solver.integrate(self.model.t + dt)

    # Retrieve values from the ODE solver    
    Y = np.split(self.ode_solver.y, [self.h_len])
    self.model.h.vector().set_local(Y[0])
    self.model.h.vector().apply("insert")
    self.model.S.vector().set_local(np.maximum(Y[1], self.zs))
    self.model.S.vector().apply("insert")
    
    # Update h
    self.model.update_h()

    
  # Steps the potential forward by dt
  def step(self, dt):
    # Note : don't change the order of these
    self.step_phi(dt)
    self.step_ode(dt)
  