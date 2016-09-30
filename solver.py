from dolfin import *
import ufl
ufl.algorithms.apply_derivatives.CONDITIONAL_WORKAROUND = True

""" Solves phi with h and S fixed."""

class Solver(object):
  
  def __init__(self, model):
    
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
    h_prev = model.h_prev
    # Channel cross sectional area
    S = model.S
    # Channel cross sectional area at previous time step
    S_prev = model.S_prev
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
    Q = -Constant(k_c) * (S_prev + phi_reg)**alpha * abs(dphi_ds + phi_reg)**delta * dphi_ds
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
  
  
    ### First the variational form for hydraulic potential PDE
  
    F_phi = Constant(e_v / (rho_w * g)) * (phi - phi_prev) * theta_cg * dx
    F_phi += dt * (-dot(grad(theta_cg), q) + (w - v - m) * theta_cg) * dx 
    F_phi += dt * (-dot(grad(theta_cg), t) * Q + (w_c - v_c) * theta_cg)('+') * dS
    
    d_phi = TrialFunction(V_cg)
    J_phi = derivative(F_phi, phi, d_phi) 


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
    
    
    ### Assign local variables
    
    self.phi = phi
    self.h = h
    self.S = S
    self.F_phi = F_phi
    self.F_h = F_h
    self.F_S = F_S
    self.J_phi = J_phi
    self.J_h = J_h
    self.J_S = J_S
    self.model = model
    self.dt = dt
    
   
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
    # Update h
    self.model.update_h()
    
    
  ### Step PDE for S forward by dt
  def step_S(self, dt):
    self.dt.assign(dt)
    # Solve ODE
    solve(self.F_S == 0, self.S, J = self.J_S, solver_parameters = self.model.newton_params)
    # Update S
    self.model.update_S()


  # Steps the potential forward by dt
  def step(self, dt):
    # Note : order matters!
    self.step_phi(dt)
    # Step s forward by dt
    self.step_S(dt)
    # Step h forward by dt
    self.step_h(dt)
  
    


  