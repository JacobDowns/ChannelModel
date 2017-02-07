from dolfin import *
import numpy as np
from tr_plot import *
parameters["form_compiler"]["precision"] = 100

class Solver(object):
  
  def __init__(self, model):
    
    # MPI rank
    self.MPI_rank = MPI.rank(mpi_comm_world())    
    # CG function space
    V_cg = model.V_cg
    V_tr = model.V_tr
    # Plots TR functions
    pt = TRPlot(model.mesh)
    
    ### Get a bunch of fields from the model 
    
    # Hydraulic potential 
    phi = Function(V_cg)
    phi.assign(project(0.9 * model.phi_0, V_cg))
    # Sheet height
    h = model.h
    h.assign(project(Expression("(x[0] + 1.0) / 600.0", degree = 1), V_cg))
    # Channel cross sectional area
    S = Function(V_tr)
    S.interpolate(Constant(1.1))
    # Potential at 0 pressure
    phi_m = model.phi_m
    # Potential at overburden pressure
    phi_0 = model.phi_0
    # Bump height
    h_r = model.h_r
    # Conductivity
    k = model.k
    # Local mask corresponding to the local array of a tr function that is 1
    # on interior edges and 0 on exterior edges
    local_mask = model.local_mask
    # Effective pressure as a function
    N_func = model.N
    # Edge lengths
    edge_lens = model.edge_lens
    
    
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
    phi_reg = Constant(1e-30)
    
  
    ### Define variational forms for each unknown phi, h, and S
  
    # Test functions 
    v_cg = TestFunction(V_cg)
    v_tr = TestFunction(V_tr)
    
    ### Expressions used in the variational forms
    
    # Expression for effective pressure
    N = phi_0 - phi
    # Normal and tangent vectors 
    n = FacetNormal(model.mesh)
    t = as_vector([n[1], -n[0]])
    # Derivative of phi along channel 
    dphi_ds = dot(grad(phi), t)
    # Discharge through channels
    Q = -Constant(k_c) * S**alpha * abs(dphi_ds + phi_reg)**delta * dphi_ds
    # Approximate discharge of sheet in direction of channel
    q_c = -k*(h**alpha)*abs(dphi_ds + phi_reg)**delta * dphi_ds
    # Energy dissipation 
    Xi = abs(Q * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)
    # Derivative of water pressure along channels
    dpw_ds = dot(grad(phi - phi_m), t)
    # Switch to turn refreezing on or of
    f1 = conditional(gt(S,0.0),1.0,0.0)
    f2 = conditional(gt(q_c * dpw_ds, 0.0), 1.0, 0.0)
    f = (f1 + f2) / (f1 + f2 + phi_reg)    
    f = Constant(1.0)
    # Sensible heat change
    Pi = -Constant(c_t*c_w*rho_w)*(Q + f*Constant(l_c)*q_c)*dpw_ds
    #Pi = -Constant(c_t*c_w*rho_w)*(Constant(l_c)*q_c)*dpw_ds    
    # Channel creep closure rate
    v_c = Constant(A)*S*(N**3)
    # Another channel source term
    w_c = ((Xi - Pi) / Constant(L)) * Constant((1. / rho_i) - (1. / rho_w))
  
  
    #ff = FacetFunctionDouble(model.mesh)
    A = Function(V_tr)
    A.vector()[:] = np.random.randint(2, size = V_tr.dim())
    
    
    dpw_ds
    
    print S.vector().array()
    quit()
    
    
    ### Manually compute a bunch of stuff
    
    # Stand in for a test function
    theta = Function(V_cg)
    
    n = FacetNormal(model.mesh)
    t = as_vector([n[1], -n[0]])
    
    # S vector array
    Sn = S.vector().array()
    # Len of channel edges
    edge_lens = assemble(v_tr('+')*dS + v_tr*ds).array()
    # Derivative of phi over edges
    dphi_dsn = assemble((dot(grad(phi),t)*v_tr)('+')*dS).array() / edge_lens
    # Derivative of pw over edges
    dpw_dsn = assemble((dot(grad(phi - phi_m),t)*v_tr)('+')*dS).array() / edge_lens
    # Flux on edges
    Qn = -k_c * Sn**alpha * np.abs(dphi_dsn + 1e-30)**delta * dphi_dsn
    # q_c on edges
    #h_alphan = (assemble((h * v_tr)('+')*dS).array() / edge_lens)**alpha
    k0 = k.vector().array()[0]
    
    
    assembled_vec = np.zeros(V_cg.dim())
    # Iterate 
    for i in range(V_cg.dim()):
      print i
      
      # Determine what edges connect to this vertex dof
      theta.vector()[i] = 1.0    
      
      l_edges = np.where(np.abs(assemble((v_tr*theta)('+')*dS).array()) > 1e-15)[0]
      
      q_c_int = -k0 * assemble((h**alpha * theta * v_tr)('+')*dS).array()[l_edges]
      q_c_int *= np.abs(dphi_dsn[l_edges] + 1e-30)**delta * dphi_dsn[l_edges]
      
      Xi_int = np.abs(q_c_int * l_c * dphi_dsn[l_edges])
      Xi_int += np.abs(Qn[l_edges] * dphi_dsn[l_edges]) * edge_lens[l_edges] * 0.5
      
      Pi_int = -c_t * c_w * rho_w * l_c * q_c_int * dpw_dsn[l_edges]
      Pi_int += -c_t * c_w * rho_w * Qn[l_edges] * dpw_dsn[l_edges] * edge_lens[l_edges] * 0.5
      
      w_c_int = ((Xi_int - Pi_int) / L) * ((1./rho_i) - (1./rho_w))
      
      v_c_int = A * assemble((N**3 * theta * v_tr)('+')*dS).array()[l_edges] * Sn[l_edges]
      
      one_int = assemble((-dot(grad(theta), t) * v_tr)('+')*dS).array()[l_edges] * Qn[l_edges]
      
      theta.vector()[i] = 0.0
      
      
      assembled_vec[i] = sum(one_int + w_c_int - v_c_int)
      #assembled_vec[i] = sum(one_int)

    print (assembled_vec - assemble((-dot(grad(v_cg), t)*Q + (w_c - v_c)*v_cg)('+')*dS).array()).max()
    quit()
       
       
    assembled_vec = np.zeros(V_cg.dim())  

    dg_ds = dot(grad(g), t)   
    
    # Iterate through each vertex dof and compute the edge integrals
    for i in [0]: #range(V_cg.dim()):
      print i
    
      
      # Derivative of test function along local edges
      g.vector()[i] = 1.0  
      dtheta_ds = assemble((dg_ds*v_tr)('+')*dS).array()
      # Integral of h times a test function centered at vertex dof i over edges
      #h_alpha_ns = assemble((h**alpha*g*v_tr)('+')*dS).array()
      
      print assemble((Q*g)('+')*dS)
      g.vector()[i] = 0.0
      
      l_edges = np.where(np.abs(dtheta_ds) > 5e-16)
      l_dtheta_ns = dtheta_ds[l_edges]
      
      # Flux along local edges
      l_Q_ns = Q_ns[l_edges]
      # Local edge lens
      l_edge_lens = edge_lens.array()[l_edges]
      # Local phi derivative on edges
      l_dphi_ns = dphi_ns[l_edges]
      #
      #l_h_alpha_ns = h_alpha_ns[l_edges]
      
      # Integrals of q_c over local channel edges
      #l_q_c_ns = -k*l_h_alpha_ns*np.abs(l_dphi_ns + 1e-30)**delta * l_dphi_ns
      
      #abs(Q * dphi_ds) + abs(Constant(l_c) * q_c * dphi_ds)
      # Integratl of Xi over local channel edges
      #l_Xi_ns = np.absolute(l_Q_ns)*l_edge_lens
      #l_Xi_ns += np.abs(l_c*l_q_c_ns*l_dphi_ns)  
      
      #print l_Q_ns*
      assembled_vec[i] = sum(l_Q_ns)
    
      #assembled_vec[i] = sum(-l_dtheta_ns * l_Q_ns)
      #assembled_vec[i] = sum(-k*l_h_alpha_ns*np.abs(l_dphi_ns + 1e-30)**delta * l_dphi_ns)
      print l_edges
      
      #q_c = -k*(h**alpha)*abs(dphi_ds + phi_reg)**delta * dphi_ds
    
    print assembled_vec
    print assemble(theta_cg('+')*dS).array()    
    #print assemble((Q*theta_cg)('+')*dS).array()

    
    
  

  
    
    

  