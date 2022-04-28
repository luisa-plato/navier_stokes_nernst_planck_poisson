#An implementation of the algorithm A1 introduced in the paper
#'Convergent finite element discretizations of the Navier--Stokes--Nernst--Planck--Poisson system'
#by Andreas Prohl and Markus Schmuck

#The set of equation we aim to solve are
#       u_t - (u * grad, u) - div(grad(u)) + gard(p) = - (p - n) * grad(phi)
#       div(u) = 0
#		p_t - div(grad(p) + p * grad(phi) - u * p) = 0
#		n_t - div(grad(n) - n * grad(phi) - u * n) = 0
#		- div(grad(phi)) = p - n

#equipped with the initial conditions
# p(0) = p_0, n(0) = n_0, u(0) = u_0
#the initial condition for phi are then already fixed by solving -div(grad(phi_0)) = p_0 - n_0
#p_0, n_0 have to be between zero and one and have to be equal in the L^1-norm over the domain.

#Additionally we assume zero Neumann boundary conditions for p, n and phi and no-slip conditions for u, that is u = 0 on the boundary of the domain.


from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

set_log_level(50)
np.set_printoptions(precision=3)
tol = 1E-14

#Create the mesh
nx = ny = 16
meshsize = 1/nx
mesh = RectangleMesh(Point(0,0), Point(1,1), nx, ny)

#upper bound for error in fixed point solver
theta = 0.000001

#Define function space ---- Solution space V contains u and p,n and phi lie in Y
V = VectorFunctionSpace(mesh, "P", 2)
Y = FunctionSpace(mesh, "P", 1)

#Set the final time and the time-step size
T = 1.0
num_steps = 100
dt = T / num_steps

#Define the analytic solutions
n_plus_e = Expression('t*cos(pi*x[0])', degree = 6, tol = tol, t = 0, pi = np.pi)
n_minus_e = Expression('t*sin(pi*x[1])', degree = 6, tol = tol, t = 0, pi = np.pi)
phi_e = Expression('t * pow(pi,-2) * (cos(pi * x[0]) - sin(pi * x[1]))', degree = 6, t = 0.0, pi = np.pi)
u_e = Expression(('-t*cos(pi*x[0])*sin(pi*x[1])','t*sin(pi*x[0])*cos(pi*x[1])'), degree = 6, t = 0, pi = np.pi)
p_e = Expression('-0.25 * (cos(2 * pi * x[0]) + cos(2 * pi * x[1]))', degree = 6, pi = np.pi)

#Define initial values for p, n and u
u_i = interpolate(Constant((0,0)),V)
p_i = interpolate(p_e, Y)
n_plus_i = interpolate(Constant(0),Y)
n_minus_i = interpolate(Constant(0),Y)
phi_i = interpolate(Constant(0),Y)

#Define boundary
boundary  = 'near(x[0], 0) || near(x[0], 1) || near(x[1],0) || near(x[1],1)'

#Define Dirichlet boundary conditions for the velocity field u based on the analytic solution
bc_u  = DirichletBC(V, u_e, boundary)
bc_p = DirichletBC(Y, p_e, boundary)
bc_plus = DirichletBC(Y, n_plus_e, boundary)
bc_minus = DirichletBC(Y, n_minus_e, boundary)
bc_phi = DirichletBC(Y, phi_e, boundary)

#Define trial- and testfunctions
u = TrialFunction(V)
n_plus = TrialFunction(Y)
n_minus = TrialFunction(Y)
phi =TrialFunction(Y)
p = TrialFunction(Y)

v = TestFunction(V)
q_plus = TestFunction(Y)
q_minus = TestFunction(Y)
g = TestFunction(Y)
q = TestFunction(Y)

#Define tentative functions for fixed point solver
u_ = Function(V)
p_ = Function(Y)
n_plus_ = Function(Y)
n_minus_ = Function(Y)
phi_ = Function(Y)

#Define tentavie velocity and pressure for the incremental pressure correction
u_ipc = Function(V)
p_ipc = Function(Y)

#Define linearized variational forms

#Define variational for the positive charge p using the tentative potential phi_ and velocity u_
a_plus = n_plus * q_plus * dx\
	+ dt * dot(grad(n_plus),grad(q_plus)) * dx\
	+ dt * n_plus * dot(grad(phi_), grad(q_plus)) * dx\
    - dt * n_plus * dot(u_, grad(q_plus)) * dx
L_plus = n_plus_i * q_plus * dx

#Define the variational form for the negative charge n using the tentative potential phi_ and velocity u_
a_minus = n_minus * q_minus *dx\
	+ dt * dot(grad(n_minus),grad(q_minus)) * dx\
	- dt * n_minus * dot(grad(phi_), grad(q_minus)) * dx\
    - dt * n_minus * dot(u_, grad(q_minus)) * dx
L_minus = n_minus_i * q_minus * dx

#Define the variational form for phi using the tentative charge densities
a_phi = dot(grad(phi),grad(g)) * dx
L_phi = (n_plus_ - n_minus_) * g * dx
L_phi_0 = (n_plus_i - n_minus_i) * g * dx

# Define expressions used in variational forms for the ICP for the velocity field u and the pressure p
U   = 0.5*(u_i + u)
n   = FacetNormal(mesh)
f   = - (n_plus_ - n_minus_) * grad(phi_)
k   = Constant(dt)

# Define strain-rate tensor
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = dot((u - u_i) / k, v)*dx + \
     dot(dot(u, nabla_grad(u_i)), v)*dx \
   + inner(sigma(U, p_i), grad(v))*dx \
   + dot(p_i * n, v) * ds - dot(nabla_grad(U) * n, v) * ds\
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_i), nabla_grad(q))*dx - (1/k)*div(u_ipc)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_ipc, v)*dx - k*dot(nabla_grad(p_ipc - p_i), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
bc_u.apply(A1)
bc_p.apply(A2)

#Create VTK file for saving solution
vtkfile_u = File('./fp_solver_nsnpp/velocity.pvd')
vtkfile_plus = File('./fp_solver_nsnpp/positive.pvd')
vtkfile_minus = File('./fp_solver_nsnpp/negative.pvd')
vtkfile_phi = File('./fp_solver_nsnpp/phi.pvd')

#Time-stepping
u = Function(V)
p = Function(Y)
n_plus = Function(Y)
n_minus = Function(Y)
phi = Function(Y)

#calculate initial value for phi
solve(a_phi == L_phi_0, phi, bc_phi)
phi_i.assign(phi)

#Saving the initial data
t = 0

vtkfile_u << (u_i, t)
vtkfile_plus << (n_plus_i, t)
vtkfile_minus << (n_minus_i, t)
vtkfile_phi << (phi_i, t)


for i in tqdm(range(num_steps)):

    # Update current time
    t += dt

    #Update time in exact solutions conditions
    u_e.t = t
    n_plus_e.t = t
    n_minus_e.t = t
    phi_e.t = t
    #the exact solution for the pressure is time indepent and thus needs no update

    #Set tentative solution to solution at previous time step
    u_.assign(u_i)
    p_.assign(p_i)
    n_plus_.assign(n_plus_i)
    n_minus_.assign(n_minus_i)
    phi_.assign(phi_i)

    #Save streamline plot of the velocity field
    plot(u_i)
    file_name = './fp_solver_nsnpp/plots/velocity_' + str(t) + '.png'
    plt.savefig(file_name)
    plt.close()
    #and of the exact solution
    u_e_projected = project(u_e, V)
    plot(u_e_projected)
    file_name = './fp_solver_nsnpp/plots/exact_velocity_' + str(t) + '.png'
    plt.savefig(file_name)
    plt.close()

    #Compute the solution for the electric potential with the tentative charges 
    solve(a_phi == L_phi, phi, bc_phi)

    #Step 1: Tentative velocity step
    b1 = assemble(L1)
    bc_u.apply(b1)
    solve(A1, u.vector(), b1)
    u_ipc.assign(u)

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    bc_p.apply(b2)
    solve(A2, p.vector(), b2)
    p_ipc.assign(p)

    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u.vector(), b3)
    u_ipc.assign(u)

    #Compute solution for the charges
    solve(a_plus == L_plus, n_plus, bc_plus)
    solve(a_minus == L_minus, n_minus, bc_minus)

    #Calculte the difference of the solution to the tentative estimate for n and p
    error_plus = norm(n_plus.vector() - n_plus_.vector(), 'linf')
    error_minus = norm(n_minus.vector() - n_minus_.vector(), 'linf')

    #Calculte the difference of the solution to the tentative estimate for phi
    error_phi = errornorm(phi, phi_, 'H10', mesh=mesh)

    #Calculate the difference of the solution to the tentative estimate for u
    error_u = errornorm(u, u_, 'L2', mesh=mesh)

    #Calculte the error
    error = error_u + error_phi + error_plus + error_minus
    #print('The error is: ', error)

    #begin for loop for the fixed point iteration
    for j in range(1000):
        #while the difference of the tentative solution to the solution at the previous time step is too big, we calculate the solution p, n, phi and u again
        if error < theta:
            #print('The fixed point solver took', j, 'iterations.')
            break

        if j == 101:
            print('This is taking a looong time!')

        #Update tentative solutions
        u_.assign(u_ipc)
        p_.assign(p_ipc)
        n_plus_.assign(n_plus)
        n_minus_.assign(n_minus)
        phi_.assign(phi)

        #Compute the solution for the electric potential with the tentative charges 
        solve(a_phi == L_phi, phi, bc_phi)

        #Step 1: Tentative velocity step
        b1 = assemble(L1)
        bc_u.apply(b1)
        solve(A1, u.vector(), b1)
        u_ipc.assign(u)

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        bc_p.apply(b2)
        solve(A2, p.vector(), b2)
        p_ipc.assign(p)

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, u.vector(), b3)
        u_ipc.assign(u)

        #Compute solution for the charges
        solve(a_plus == L_plus, n_plus, bc_plus)
        solve(a_minus == L_minus, n_minus, bc_minus)

        #Calculte the difference of the solution to the tentative estimate for n and p
        error_plus = norm(n_plus.vector() - n_plus_.vector(), 'linf')
        error_minus = norm(n_minus.vector() - n_minus_.vector(), 'linf')

        #Calculte the difference of the solution to the tentative estimate for phi
        error_phi = errornorm(phi, phi_, 'H10', mesh=mesh)

        #Calculate the difference of the solution to the tentative estimate for u
        error_u = errornorm(u, u_, 'L2', mesh=mesh)

        #Calculte the error
        error = error_u + error_phi + error_plus + error_minus

    #the tentative solution is close enough and becomes the new solution and becomes the new previous solution
    u_i.assign(u)
    p_i.assign(p)
    n_plus_i.assign(n_plus)
    n_minus_i.assign(n_minus)
    phi_i.assign(phi)

    # Save to file
    vtkfile_u << (u_i, t)
    vtkfile_plus << (n_plus_i,t)
    vtkfile_minus << (n_minus_i,t)
    vtkfile_phi << (phi_i, t)

#Calculate the error norms
diff_u = errornorm(u_e, u_i, 'L2')
diff_plus = errornorm(n_plus_e, n_plus_i, 'L2')
diff_minus = errornorm(n_minus_e, n_minus_i, 'L2')
diff_phi = errornorm(phi_e, phi_i, 'L2')

#Write the errors to a csv-file for later use
data = {'diff_u': [diff_u],
    'diff_minus': [diff_minus],
    'diff_plus': [diff_plus],
    'diff_phi': [diff_phi]
}

df = pd.DataFrame(data, index=['1/16'])
df.to_csv('errors.txt', mode='a', header = False)