from dolfin import *

""" Solves phi with h and S fixed."""

class Solver(object):
  
  def __init__(self, model):
    
    ### Get function space
    V_cg = model.V_cg
    
    ### Get a bunch of fields from the model 
    
    # Hydraulic potential 
    phi = model.phi
    # Potential at previous time step
    phi_prev = model.phi_prev
    # Sheet height
    h = model.h
    # Sheet height at previous time step
    h_prev = model.h_prev
    # Get melt rate
    m = model.m
    # Basal sliding speed
    u_b = model.u_b
    # Potential
    phi = model.phi
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
    # Gravitational acceleration
    g = model.pcs['g']
    # Exponents
    alpha = model.pcs['alpha']
    delta = model.pcs['delta']
    # Regularization parameter
    phi_reg = Constant(1e-15)
    
  
    ### Define variational forms for each unknown phi, h, and S
  
    # Test functions 
    theta = TestFunction(V_cg)
    
    
    ### Expressions used in the variational forms
    
    # Expression for effective pressure
    N = phi_0 - phi
    # Flux vector
    q = -k * h**alpha * (dot(grad(phi), grad(phi)) + phi_reg)**(delta / 2.0) * grad(phi)
    # Opening term 
    w = conditional(gt(h_r - h, 0.0), u_b * (h_r - h) / Constant(l_r), 0.0)
    # Closing term
    v = Constant(A) * h * N**3    
    # Time step 
    dt = Constant(1.0)
  
  
    ### First the variational form for hydraulic potential PDE
  
    F1 = Constant(e_v / (rho_w * g)) * (phi - phi_prev) * theta1 * dx
    F1 += dt * (-dot(grad(theta1), q) + (w - v - m) * theta1) * dx 

    ### Variational form for the sheet height ODE
    F2 = ((h - h_prev) - (w - v) * dt) * theta2 * dx
    
    ### Variational form for the sheet height ODE
    F3 = (((S - S_prev) - ((Xi - Pi) / Constant(rho_i * L)) * dt) * theta3)('+') * dS
    
    ### Combined variational form
    F = F1 + F2 + F3

    du = TrialFunction(V)
    J = derivative(F, u, du) 
    
    
    ### Assign local variables

    self.F = F
    self.J = J
    self.model = model
    self.dt = dt


  # Steps the potential forward by dt
  def step(self, dt):
    self.dt.assign(dt)
    # Solve for potential
    solve(self.F == 0, self.model.phi, self.model.d_bcs, J = self.J, solver_parameters = self.model.newton_params)
    # Derive values from the new potential 
    self.model.update_phi()
    self.update_h()
    self.update_S()

  