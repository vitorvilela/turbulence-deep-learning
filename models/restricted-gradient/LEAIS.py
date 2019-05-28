# Large Eddy Artificial Intelligence Simulation (LEAIS) - Python 3.4
# Author: Vitor Vilela
# Created at: 28/09/2017
# Last modification at: 01/02/2018



# TODO
# Pyplot profiles (u, v) automatically against Ghia
# Create cnn inputs and outputs
# Study scales to set dns mesh properly

# Run a case until 40s with a diffusive model, than change to AI model to verify dependence from the transient regime


import numpy as np
import pyamg
from scipy.misc import imsave
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.externals.joblib import dump 
from sklearn.externals.joblib import load


# Coarse Mesh for Filtered Variables
COARSE_MESH = 32

# Fine Mesh for Detailed Variables
FINE_MESH = 128

# Number of sides
SIDES = 4

# Number of variables in the transport equation 
VARIABLES = 3

# Estimator / Corrector
STEPS = 2

# Characteristic Length
L = 1.

# Characteristic Velocity
VEL = 1.

# Reynolds
Re = 1000

# Properties
NU = VEL*L/Re

# Spatial Step
DELTA_FINE = L/FINE_MESH
DELTA_COARSE = L/COARSE_MESH

# Time Step
DT = DELTA_FINE*DELTA_FINE
#DT = DELTA_FINE

# Pressure solver tolerance
EPS = 1.e-8

# Simulation time
ST = 60.

# Print at
PRINT_AT = ST/10

# AI at
EXPORT_AI = True
#PRINT_AI_AT = ST/50
PRINT_AI_AT = 1000*DT

# Probe at
#PROBE_AT = ST/100
PROBE_AT = 1000*DT

# Info at
#INFO_AT = ST/100
INFO_AT = 1000*DT

# Mesh tag
COARSE = 0
FINE = 1

# Coarse simulation with fine simulation just for direct comparison
RUN_FILTERED = False

# Run AI model with FINE_MESH
RUN_MODEL = False

# Print velocity fields for comparison 
PRINT_COMPARE = True


# Faces Dictionary (GHOST LEFT, LEFT, GHOST RIGHT, RIGHT, GHOST BOTTOM, BOTTOM, GHOST TOP, TOP)
F = []
F.append({'GL': 0, 'L': 1, 'GR': COARSE_MESH+1, 'R': COARSE_MESH, 'GB': 0, 'B': 1, 'GT': COARSE_MESH+1, 'T': COARSE_MESH})
F.append({'GL': 0, 'L': 1, 'GR': FINE_MESH+1, 'R': FINE_MESH, 'GB': 0, 'B': 1, 'GT': FINE_MESH+1, 'T': FINE_MESH})
  
# Estimator / Corrector Dictionary
EC = {'E': 0, 'C':1}


# Variables
# Fine Velocity
u, v = [np.zeros((STEPS, FINE_MESH+2, FINE_MESH+2), dtype=np.float64) for _ in range(2)]

# Fine Pressure
p = np.zeros((FINE_MESH+2, FINE_MESH+2), dtype=np.float64) 

# Fine Cross-Velocity
uu, vv, uv = [np.zeros((FINE_MESH+2, FINE_MESH+2), dtype=np.float64) for _ in range(3)]

# Fine Stress-Tensor
sxx, syy, sxy = [np.zeros((FINE_MESH+2, FINE_MESH+2), dtype=np.float64) for _ in range(3)]

# Velocity Gradients
dudx, dudy, dvdx, dvdy = [np.zeros((FINE_MESH+2, FINE_MESH+2), dtype=np.float64) for _ in range(4)]


# Variables
# Filtered Velocity
u_hat, v_hat = [np.zeros((STEPS, COARSE_MESH+2, COARSE_MESH+2), dtype=np.float64) for _ in range(2)]

# Filtered Pressure
p_hat = np.zeros((COARSE_MESH+2, COARSE_MESH+2), dtype=np.float64) 

# Filtered Stress-Tensor
sxx_hat, syy_hat, sxy_hat = [np.zeros((COARSE_MESH+2, COARSE_MESH+2), dtype=np.float64) for _ in range(3)]


# Outputs
# Coarse Velocity, Filtered 
U, V = [np.zeros((COARSE_MESH+2, COARSE_MESH+2), dtype=np.float64) for _ in range(2)]

# Coarse Cross-Velocity, Filtered
UU, VV, UV = [np.zeros((COARSE_MESH+2, COARSE_MESH+2), dtype=np.float64) for _ in range(3)]

# Coarse Cross-Velocity, Filtered
DUDX, DUDY, DVDX, DVDY = [np.zeros((COARSE_MESH+2, COARSE_MESH+2), dtype=np.float64) for _ in range(4)]


# Load ML models
#TUU_model = load('TUU_model.sav') 
#TVV_model = load('TVV_model.sav') 
#TUV_model = load('TUV_model.sav') 
#scaler = load('scaler.sav')



# Boundary Condition for Cavity
def flow_boundary(variable, rfn):
   
  if rfn == FINE: 
   
    if variable == 'velocity': 
      
      # ...left
      v[EC['C'],F[rfn]['GL'],:] = -v[EC['C'],F[rfn]['L'],:] 
	    
      # ...right
      u[EC['C'],F[rfn]['GR'],:] = 0.
      v[EC['C'],F[rfn]['GR'],:] = -v[EC['C'],F[rfn]['R'],:]
	    
      # ...bottom
      u[EC['C'],:,F[rfn]['GB']] = -u[EC['C'],:,F[rfn]['B']]
	  
      # ...top
      u[EC['C'],:,F[rfn]['GT']] = 2.*VEL - u[EC['C'],:,F[rfn]['T']]
      v[EC['C'],:,F[rfn]['GT']] = 0.
  
    elif variable == 'velocity-estimation': 
		
      # ...right
      u[EC['E'],F[rfn]['GR'],:] = u[EC['E'],F[rfn]['R'],:]
	      
      # ...top
      v[EC['E'],:,F[rfn]['GT']] = v[EC['E'],:,F[rfn]['T']]
    
    elif variable == 'pressure':
  
      # ...left
      p[F[rfn]['GL'],:] = p[F[rfn]['L'],:]
  
      # ...right
      p[F[rfn]['GR'],:] = p[F[rfn]['R'],:] 
    
      # ...bottom
      p[:,F[rfn]['GB']] = p[:,F[rfn]['B']]
  
      # ...top
      p[:,F[rfn]['GT']] = p[:,F[rfn]['T']]
    
    
    elif variable == 'stress': 
    
      # ...left
      sxx[F[rfn]['GL'],:] = sxx[F[rfn]['L'],:]
      syy[F[rfn]['GL'],:] = syy[F[rfn]['L'],:]
    
      # ...right
      sxx[F[rfn]['GR'],:] = sxx[F[rfn]['R'],:]
      syy[F[rfn]['GR'],:] = syy[F[rfn]['R'],:]
    
      # ...bottom
      sxx[:,F[rfn]['GB']] = sxx[:,F[rfn]['B']]
      syy[:,F[rfn]['GB']] = syy[:,F[rfn]['B']]
    
      # ...top
      sxx[:,F[rfn]['GT']] = sxx[:,F[rfn]['T']]
      syy[:,F[rfn]['GT']] = syy[:,F[rfn]['T']]
    
    
  elif rfn == COARSE:  
    
    if variable == 'velocity': 
      
      # ...left
      v_hat[EC['C'],F[rfn]['GL'],:] = -v_hat[EC['C'],F[rfn]['L'],:] 
	    
      # ...right
      u_hat[EC['C'],F[rfn]['GR'],:] = 0.
      v_hat[EC['C'],F[rfn]['GR'],:] = -v_hat[EC['C'],F[rfn]['R'],:]
	    
      # ...bottom
      u_hat[EC['C'],:,F[rfn]['GB']] = -u_hat[EC['C'],:,F[rfn]['B']]
	  
      # ...top
      u_hat[EC['C'],:,F[rfn]['GT']] = 2.*VEL - u_hat[EC['C'],:,F[rfn]['T']]
      v_hat[EC['C'],:,F[rfn]['GT']] = 0.
  
    elif variable == 'velocity-estimation': 
		
      # ...right
      u_hat[EC['E'],F[rfn]['GR'],:] = u_hat[EC['E'],F[rfn]['R'],:]
	      
      # ...top
      v_hat[EC['E'],:,F[rfn]['GT']] = v_hat[EC['E'],:,F[rfn]['T']]
    
    elif variable == 'pressure':
  
      # ...left
      p_hat[F[rfn]['GL'],:] = p_hat[F[rfn]['L'],:]
  
      # ...right
      p_hat[F[rfn]['GR'],:] = p_hat[F[rfn]['R'],:] 
    
      # ...bottom
      p_hat[:,F[rfn]['GB']] = p_hat[:,F[rfn]['B']]
  
      # ...top
      p_hat[:,F[rfn]['GT']] = p_hat[:,F[rfn]['T']]
    
    
    elif variable == 'stress': 
    
      # ...left
      sxx_hat[F[rfn]['GL'],:] = sxx_hat[F[rfn]['L'],:]
      syy_hat[F[rfn]['GL'],:] = syy_hat[F[rfn]['L'],:]
    
      # ...right
      sxx_hat[F[rfn]['GR'],:] = sxx_hat[F[rfn]['R'],:]
      syy_hat[F[rfn]['GR'],:] = syy_hat[F[rfn]['R'],:]
    
      # ...bottom
      sxx_hat[:,F[rfn]['GB']] = sxx_hat[:,F[rfn]['B']]
      syy_hat[:,F[rfn]['GB']] = syy_hat[:,F[rfn]['B']]
    
      # ...top
      sxx_hat[:,F[rfn]['GT']] = sxx_hat[:,F[rfn]['T']]
      syy_hat[:,F[rfn]['GT']] = syy_hat[:,F[rfn]['T']]
   
   
   
 
def flow_stress():
    
  for i in range(F[FINE]['L'], F[FINE]['GR']+1):
    for j in range(F[FINE]['B'], F[FINE]['GT']+1):
      
      # Normal stresses - 
      # It ensures just computing for domain cells
      if i < F[FINE]['R']+1 and j < F[FINE]['T']+1:
        sxx[i,j] = -0.25*pow(u[EC['C'],i,j]+u[EC['C'],i+1,j],2.) + 2.*NU*(u[EC['C'],i+1,j]-u[EC['C'],i,j])/DELTA_FINE
        syy[i,j] = -0.25*pow(v[EC['C'],i,j]+v[EC['C'],i,j+1],2.) + 2.*NU*(v[EC['C'],i,j+1]-v[EC['C'],i,j])/DELTA_FINE
  
      # Shear stress
      sxy[i,j] = -0.25*(u[EC['C'],i,j]+u[EC['C'],i,j-1])*(v[EC['C'],i,j]+v[EC['C'],i-1,j]) + NU*(u[EC['C'],i,j]-u[EC['C'],i,j-1]+v[EC['C'],i,j]-v[EC['C'],i-1,j])/DELTA_FINE
  


# Atualizado 02/02/2018
# Compute velocity gradient fields for fine mesh
def flow_velocityGradient():    
  for i in range(F[FINE]['L'], F[FINE]['R']):
    for j in range(F[FINE]['B'], F[FINE]['T']):  
      dudx = (u[EC['C'],i+1,j]-u[EC['C'],i-1,j])/(2*DELTA_FINE)
      dudy = (u[EC['C'],i,j+1]-u[EC['C'],i,j-1])/(2*DELTA_FINE)
      dvdx = (v[EC['C'],i+1,j]-v[EC['C'],i-1,j])/(2*DELTA_FINE)
      dvdy = (v[EC['C'],i,j+1]-v[EC['C'],i,j-1])/(2*DELTA_FINE)
      
      
  
 
# Function to model the Turbulence tensor 
def flow_ml_stress():
   
  
  for i in range(F[FINE]['L'], F[FINE]['R']+2):
    for j in range(F[FINE]['B'], F[FINE]['T']+2):
            
      ml_input = np.array( [[ i/FINE_MESH, j/FINE_MESH, u[EC['C'],i,j], v[EC['C'],i,j], u[EC['C'],i,j]*v[EC['C'],i,j] ]] )
            
      tuu_pred = TUU_model.predict(scaler.transform(ml_input))
      tvv_pred = TVV_model.predict(scaler.transform(ml_input))
      tuv_pred = TUV_model.predict(scaler.transform(ml_input))
            
      # Normal stresses
      if i < F[FINE]['R']+1 and j < F[FINE]['T']+1:
        sxx[i,j] = -np.exp(tuu_pred) - 0.25*pow(u[EC['C'],i,j]+u[EC['C'],i+1,j],2.) + 2.*NU*(u[EC['C'],i+1,j]-u[EC['C'],i,j])/DELTA_FINE      
        syy[i,j] = -np.exp(tvv_pred) - 0.25*pow(v[EC['C'],i,j]+v[EC['C'],i,j+1],2.) + 2.*NU*(v[EC['C'],i,j+1]-v[EC['C'],i,j])/DELTA_FINE
                
      # Shear stress
      sxy[i,j] = -np.exp(tuv_pred) - 0.25*(u[EC['C'],i,j]+u[EC['C'],i,j-1])*(v[EC['C'],i,j]+v[EC['C'],i-1,j]) + NU*(u[EC['C'],i,j]-u[EC['C'],i,j-1]+v[EC['C'],i,j]-v[EC['C'],i-1,j])/DELTA_FINE
      
      

  
  
  
# Function to model the cross-velocity tensor directly
#def flow_ml_stress():
   
  
  #for i in range(F[FINE]['L'], F[FINE]['R']+2):
    #for j in range(F[FINE]['B'], F[FINE]['T']+2):
      
      #ml_input = np.array( [[ i/FINE_MESH, j/FINE_MESH, u[EC['C'],i,j], v[EC['C'],i,j], u[EC['C'],i,j]*v[EC['C'],i,j] ]] )
      ##ml_input = np.array( [[ u[EC['C'],i,j], v[EC['C'],i,j], u[EC['C'],i,j]*v[EC['C'],i,j] ]] )
      
      #if np.fabs(u[EC['C'],i,j]*v[EC['C'],i,j]) > 1.e-4:
        #u_pred = UU_model.predict(scaler.transform(ml_input))
        #v_pred = VV_model.predict(scaler.transform(ml_input))
        #uv_pred = UV_model.predict(scaler.transform(ml_input))
      #else:
        #u_pred = 0.
        #v_pred = 0.
        #uv_pred = 0.
      
      ## Normal stresses
      #if i < F[FINE]['R']+1 and j < F[FINE]['T']+1:
        #sxx[i,j] = -u_pred + 2.*NU*(u[EC['C'],i+1,j]-u[EC['C'],i,j])/DELTA_FINE      
        #syy[i,j] = -v_pred + 2.*NU*(v[EC['C'],i,j+1]-v[EC['C'],i,j])/DELTA_FINE
                
      ## Shear stress
      #sxy[i,j] = -uv_pred + NU*(u[EC['C'],i,j]-u[EC['C'],i,j-1]+v[EC['C'],i,j]-v[EC['C'],i-1,j])/DELTA_FINE
      
      





def flow_estimator(rfn):
  
  if rfn == FINE:
  
    for i in range(F[rfn]['L'], F[rfn]['R']+1):
      for j in range(F[rfn]['B'], F[rfn]['T']+1):
        u[EC['E'],i,j] = u[EC['C'],i,j] + DT*(sxx[i,j]-sxx[i-1,j]+sxy[i,j+1]-sxy[i,j])/DELTA_FINE
        v[EC['E'],i,j] = v[EC['C'],i,j] + DT*(syy[i,j]-syy[i,j-1]+sxy[i+1,j]-sxy[i,j])/DELTA_FINE
      
  elif rfn == COARSE:

    for i in range(F[rfn]['L'], F[rfn]['R']+1):
      for j in range(F[rfn]['B'], F[rfn]['T']+1):
        u_hat[EC['E'],i,j] = u_hat[EC['C'],i,j] + DT*(sxx_hat[i,j]-sxx_hat[i-1,j]+sxy_hat[i,j+1]-sxy_hat[i,j])/DELTA_COARSE
        v_hat[EC['E'],i,j] = v_hat[EC['C'],i,j] + DT*(syy_hat[i,j]-syy_hat[i,j-1]+sxy_hat[i+1,j]-sxy_hat[i,j])/DELTA_COARSE
      
      
      
      
      
      
def solver_pressure(t):
  
  A = pyamg.gallery.poisson((FINE_MESH, FINE_MESH), format='csr')  
  b = np.zeros((FINE_MESH, FINE_MESH), dtype=np.float64) 
  x = np.zeros((FINE_MESH, FINE_MESH), dtype=np.float64) 
  
  max_div = 0.
  
  for i in range(F[FINE]['L'], F[FINE]['R']+1):
    for j in range(F[FINE]['B'], F[FINE]['T']+1):
      div = pow(DELTA_FINE,2.)*(u[EC['E'],i+1,j]-u[EC['E'],i,j]+v[EC['E'],i,j+1]-v[EC['E'],i,j])
      b[i-1,j-1] = -div/DT
      if np.fabs(div) > max_div: 
        max_div = np.fabs(div)
  
  if t % INFO_AT < DT:
    print('{} {:1.1e}'.format('                                      MAX_DIV:', max_div) )
  
  rhs = b.ravel()
      
  ml = pyamg.ruge_stuben_solver(A)      # construct the multigrid hierarchy
  x = ml.solve(b, tol=EPS)              # solve Ax=b to a tolerance of EPS
  return np.linalg.norm(rhs-A*x)	# compute and return the norm of residual vector    
  
  
  
 
def flow_corrector(rfn):
  
  if rfn == FINE:
  
    for i in range(F[rfn]['L'], F[rfn]['R']+1):
      for j in range(F[rfn]['B'], F[rfn]['T']+1):
        u[EC['C'],i,j] = u[EC['E'],i,j] + DT*(p[i,j]-p[i-1,j])/DELTA_FINE
        v[EC['C'],i,j] = v[EC['E'],i,j] + DT*(p[i,j]-p[i,j-1])/DELTA_FINE
  
  elif rfn == COARSE:

    for i in range(F[rfn]['L'], F[rfn]['R']+1):
      for j in range(F[rfn]['B'], F[rfn]['T']+1):
        u_hat[EC['C'],i,j] = u_hat[EC['E'],i,j] + DT*(p_hat[i,j]-p_hat[i-1,j])/DELTA_COARSE
        v_hat[EC['C'],i,j] = v_hat[EC['E'],i,j] + DT*(p_hat[i,j]-p_hat[i,j-1])/DELTA_COARSE





# uv at cell's left-bottom corner
def flow_crossVelocity():
  
  for i in range(F[FINE]['L'], F[FINE]['R']+1):
    for j in range(F[FINE]['B'], F[FINE]['T']+1):
      uu[i,j] = u[EC['C'],i,j]*u[EC['C'],i,j]
      vv[i,j] = v[EC['C'],i,j]*v[EC['C'],i,j]
      uv[i,j] = 0.25*(u[EC['C'],i,j]+u[EC['C'],i,j-1])*(v[EC['C'],i,j]+v[EC['C'],i-1,j])



def filtered_velocity():
  
  MR = int(FINE_MESH/COARSE_MESH)
        
  for I in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for J in range(F[COARSE]['B'], F[COARSE]['T']+1): 
      U[I,J] = 0.
      V[I,J] = 0.
      for i in range(1, MR):
        for j in range(1, MR):
          U[I,J] += (1./pow(MR, 2.))*(u[EC['C'], (I-1)*MR+i, (J-1)*MR+j])
          V[I,J] += (1./pow(MR, 2.))*(v[EC['C'], (I-1)*MR+i, (J-1)*MR+j])


# UV at cell's left-bottom corner
def filtered_crossVelocity():
  
  MR = int(FINE_MESH/COARSE_MESH)
      
  for I in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for J in range(F[COARSE]['B'], F[COARSE]['T']+1):
      UU[I,J] = 0.
      VV[I,J] = 0.
      UV[I,J] = 0.          
      for i in range(1, MR):
        for j in range(1, MR):  
          UU[I,J] += (1./pow(MR, 2.))*(uu[(I-1)*MR+i, (J-1)*MR+j])
          VV[I,J] += (1./pow(MR, 2.))*(vv[(I-1)*MR+i, (J-1)*MR+j])
          UV[I,J] += (1./pow(MR, 2.))*(uv[(I-1)*MR+i, (J-1)*MR+j])


# Atualizado 02/02/2018
# Interpolate velocity gradient fields from fine to coarse mesh in order to feed AI model
def filtered_velocityGradient():
  
  MR = int(FINE_MESH/COARSE_MESH)
        
  for I in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for J in range(F[COARSE]['B'], F[COARSE]['T']+1): 
      DUDX[I,J] = 0.
      DUDY[I,J] = 0.
      DVDX[I,J] = 0.
      DVDY[I,J] = 0.
      for i in range(1, MR):
        for j in range(1, MR):
          DUDX[I,J] += (1./pow(MR, 2.))*(dudx[(I-1)*MR+i, (J-1)*MR+j])
          DUDY[I,J] += (1./pow(MR, 2.))*(dudy[(I-1)*MR+i, (J-1)*MR+j])
          DVDX[I,J] += (1./pow(MR, 2.))*(dvdx[(I-1)*MR+i, (J-1)*MR+j])
          DVDY[I,J] += (1./pow(MR, 2.))*(dvdy[(I-1)*MR+i, (J-1)*MR+j])
     
  
  
def flow_filteredStress():
  
  # Normal stresses
  for i in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for j in range(F[COARSE]['B'], F[COARSE]['T']+1):
      sxx_hat[i,j] = -UU[i,j] + 2.*NU*(u_hat[EC['C'],i+1,j]-u_hat[EC['C'],i,j])/DELTA_COARSE
      syy_hat[i,j] = -VV[i,j] + 2.*NU*(v_hat[EC['C'],i,j+1]-v_hat[EC['C'],i,j])/DELTA_COARSE
  
  # Shear stress
  for i in range(F[COARSE]['L'], F[COARSE]['R']+2):
    for j in range(F[COARSE]['B'], F[COARSE]['T']+2):
      sxy_hat[i,j] = -UV[i,j] + NU*(u_hat[EC['C'],i,j]-u_hat[EC['C'],i,j-1]+v_hat[EC['C'],i,j]-v_hat[EC['C'],i-1,j])/DELTA_COARSE
  
  
  
def solver_filteredPressure(t):
  
  A = pyamg.gallery.poisson((COARSE_MESH, COARSE_MESH), format='csr')  
  b = np.zeros((COARSE_MESH, COARSE_MESH), dtype=np.float64) 
  x = np.zeros((COARSE_MESH, COARSE_MESH), dtype=np.float64) 
  
  max_div = 0.
  
  for i in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for j in range(F[COARSE]['B'], F[COARSE]['T']+1):
      div = pow(DELTA_COARSE,2.)*(u_hat[EC['E'],i+1,j]-u_hat[EC['E'],i,j]+v_hat[EC['E'],i,j+1]-v_hat[EC['E'],i,j])
      b[i-1,j-1] = -div/DT
      if np.fabs(div) > max_div: 
        max_div = np.fabs(div)
            
  if t % INFO_AT < DT:
    print('{} {:1.1e}'.format('                                      FILTERED MAX_DIV:', max_div) )    
  
  rhs = b.ravel()
      
  ml = pyamg.ruge_stuben_solver(A)      # construct the multigrid hierarchy
  x = ml.solve(b, tol=EPS)              # solve Ax=b to a tolerance of EPS
  return np.linalg.norm(rhs-A*x)	# compute and return the norm of residual vector    
    



def flow_probe(t):
  
  i = int(FINE_MESH/2)
  j = int(FINE_MESH/2)
  print('{} {:1.1e} {} {:1.3e} {} {:1.3e}'.format('TIME', t, '     u', u[EC['C'],i,j], '     v', v[EC['C'],i,j])) 

  if RUN_FILTERED == True:
    i = int(COARSE_MESH/2)
    j = int(COARSE_MESH/2)
    print('{} {:1.1e} {} {:1.3e} {} {:1.3e}'.format('TIME', t, '     u_hat', u_hat[EC['C'],i,j], '     v_hat', v_hat[EC['C'],i,j])) 
  
  
  
def flow_print(t):
  
  file_name = '{}{:d}{}'.format('u-',int(t),'s.png')   
  imsave(file_name, np.flipud( np.swapaxes( u[EC['C'], F[FINE]['L']:F[FINE]['R']+1, F[FINE]['B']:F[FINE]['T']+1], 0, 1 ) ) )
  file_name = '{}{:d}{}'.format('v-',int(t),'s.png')
  imsave(file_name, np.flipud( np.swapaxes( v[EC['C'], F[FINE]['L']:F[FINE]['R']+1, F[FINE]['B']:F[FINE]['T']+1], 0, 1 ) ) )

  if RUN_FILTERED == True:
    file_name = '{}{:d}{}'.format('u_hat-',int(t),'s.png')   
    imsave(file_name, np.flipud( np.swapaxes( u_hat[EC['C'], F[COARSE]['L']:F[COARSE]['R']+1, F[COARSE]['B']:F[COARSE]['T']+1], 0, 1 ) ) )
    file_name = '{}{:d}{}'.format('v_hat-',int(t),'s.png')
    imsave(file_name, np.flipud( np.swapaxes( v_hat[EC['C'], F[COARSE]['L']:F[COARSE]['R']+1, F[COARSE]['B']:F[COARSE]['T']+1], 0, 1 ) ) )
 




def ai_print(t, ai_file):
  
  for i in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for j in range(F[COARSE]['B'], F[COARSE]['T']+1): 
      line = '{:.8f}{}{:d}{}{:d}{}{:.16f}{}{:.16f}{}{:.16f}{}{:.16f}{}{:.16f}{}{:.16f}{}{:.16f}{}{:.16f}{}{:.16f}{}'.format(t, ',', i, ',', j, ',', U[i,j], ',', V[i,j], ',', UU[i,j], ',', VV[i,j], ',', UV[i,j], ',', DUDX[i,j], ',', DUDY[i,j], ',', DVDX[i,j], ',', DVDY[i,j], '\n')
      ai_file.write(line)  
  
  
  
  

def field_print(field_file):
  for i in range(F[COARSE]['L'], F[COARSE]['R']+1):
    for j in range(F[COARSE]['B'], F[COARSE]['T']+1):
      field_file.write('{:d}{}{:d}{}{:.16f}{}{:.16f}{}'.format(i, ',', j, ',', u[EC['C'],i,j], ',', v[EC['C'],i,j], '\n'))
  
   

t = 0.
rsd = rsd_hat = 0.


if EXPORT_AI:
  ai_file = open('feed.csv', 'w')
  line = '{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format('t', ',', 'i', ',', 'j', ',', 'U', ',', 'V', ',', 'UU', ',', 'VV', ',', 'UV', ',', 'DUDX', ',', 'DUDY', ',', 'DVDX', ',', 'DVDY', '\n')
  ai_file.write(line)  


# Flow Simulation
while t <= ST:

  t += DT

  flow_boundary('velocity', FINE)
    
  
  # AI
  if EXPORT_AI:    
    filtered_velocity()  
    flow_crossVelocity()    
    filtered_crossVelocity()
    flow_velocityGradient()  
    filtered_velocityGradient()
    


  # FILTERED  
  if RUN_FILTERED:
    flow_boundary('velocity', COARSE)
    flow_filteredStress()
    flow_boundary('stress', COARSE)
    flow_estimator(COARSE)
    flow_boundary('velocity-estimation', COARSE)
    rsd_hat = solver_filteredPressure(t)    
    flow_boundary('pressure', COARSE)
    flow_corrector(COARSE)


  # FINE
  if RUN_MODEL:
    flow_ml_stress()
  else:
    flow_stress()
  flow_boundary('stress', FINE)
  flow_estimator(FINE)
  flow_boundary('velocity-estimation', FINE)
  rsd = solver_pressure(t)    
  flow_boundary('pressure', FINE)
  flow_corrector(FINE)
  
  

  
  
  
  if t % INFO_AT < DT:
    print('{} {:1.1e} {} {:1.1e}'.format('TIME:', t, '                        RESIDUE:', rsd) )
    if RUN_FILTERED == True:
      print('{} {:1.1e}'.format('                                      FILTERED RESIDUE:', rsd_hat) )
    
  if t % PROBE_AT < DT:
    flow_probe(t)
    
  # The current scipy version started to normalize all images so that min(data) become black and max(data) become white  
  if t % PRINT_AT < DT:      
    flow_print(t)
    
    if PRINT_COMPARE:
      field_filename = '{}{:d}{}'.format('field-', int(t), '.csv')
      field_file = open(field_filename, 'w')
      field_print(field_file)
      field_file.close()
  
    if EXPORT_AI and t % PRINT_AI_AT < DT:
      ai_print(t, ai_file)
    
    
    


if EXPORT_AI:
  ai_file.close()
