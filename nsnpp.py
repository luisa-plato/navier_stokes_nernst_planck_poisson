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
nx = ny = 64
meshsize = 1/nx
mesh = RectangleMesh(Point(0,0), Point(1,1), nx, ny)

#upper bound for error in fixed point solver
#theta = meshsize**3
theta = 0.000001
regul = 0.0001


#Define the MINI element for the velocity u
P1 = FiniteElement("Lagrange", "triangle", 1)
B = FiniteElement("Bubble", "triangle", 3)
MINI = VectorElement(NodalEnrichedElement(P1, B))

#Define function space ---- Solution space V contains u and p,n and phi lie in Y
V = FunctionSpace(mesh, MINI)
Y = FunctionSpace(mesh, "P", 1)

#Set the final time and the time-step size
T = 0.1
num_steps = 100
dt = T / num_steps

#Define the analytic solutions
p_e = Expression('t*cos(pi*x[0])', degree = 1, tol = tol, t = 0, pi = np.pi)
n_e = Expression('t*sin(pi*x[1])', degree = 1, tol = tol, t = 0, pi = np.pi)
phi_e = Expression('t * pow(pi,-2) * (cos(pi * x[0]) - sin(pi * x[1]))', degree = 1, t = 0.0, pi = np.pi)
u_e = Expression(('-t*cos(pi*x[0])*sin(pi*x[1])','t*sin(pi*x[0])*cos(pi*x[1])'), degree = 1, t = 0, pi = np.pi)

#Define initial values for p, n and u
u_i = project(Constant((0,0)),V)
p_i = interpolate(Constant(0),Y)
n_i = interpolate(Constant(0),Y)
phi_i = interpolate(Constant(0),Y)

#Define boundary
boundary  = 'near(x[0], 0) || near(x[0], 1) || near(x[1],0) || near(x[1],1)'

#Define Dirichlet boundary conditions for the velocity field u based on the analytic solution
bc_u  = DirichletBC(V, u_e, boundary)
bc_p = DirichletBC(Y, p_e, boundary)
bc_n = DirichletBC(Y, n_e, boundary)
bc_phi = DirichletBC(Y, phi_e, boundary)

#Define trial- and testfunctions
u = TrialFunction(V)
p = TrialFunction(Y)
n = TrialFunction(Y)
phi =TrialFunction(Y)

v = TestFunction(V)
q_p = TestFunction(Y)
q_n = TestFunction(Y)
g = TestFunction(Y)

#Define tentative functions for fixed point solver
u_ = Function(V)
p_ = Function(Y)
n_ = Function(Y)
phi_ = Function(Y)

#Define linearized variational forms

#Define variational for the positive charge p using the tentative potential phi_ and velocity u_
a_p = p * q_p * dx\
	+ dt * dot(grad(p),grad(q_p)) * dx\
	+ dt * p * dot(grad(phi_), grad(q_p)) * dx\
    - dt * p * dot(u_, grad(q_p)) * dx
L_p = p_i * q_p * dx

#Define the variational form for the negative charge n using the tentative potential phi_ and velocity u_
a_n = n * q_n *dx\
	+ dt * dot(grad(n),grad(q_n)) * dx\
	- dt * n * dot(grad(phi_), grad(q_n)) * dx\
    - dt * n * dot(u_, grad(q_n)) * dx
L_n = n_i * q_n * dx

#Define the variational form for phi using the tentative charge densities
a_phi = dot(grad(phi),grad(g)) * dx
L_phi = (p_ - n_) * g * dx
L_phi_0 = (p_i - n_i) * g * dx

#Define the variational form for the velocity field u 
a_u = dot(u, v) * dx\
    + dt * inner(grad(u), grad(v)) * dx\
    + dt * dot(dot(u_i, nabla_grad(u)), v) * dx\
    + 0.5 * dt * div(u_) * dot(u, v) * dx
#+ regul * inner(grad(u), grad(v)) * dx
L_u = - dt * (p_ - n_) * dot(grad(phi_), v) * dx\
    + dot(u_i, v) * dx
#+ regul * inner(grad(u_i), grad(v)) * dx


#Create VTK file for saving solution
vtkfile_u = File('./fp_solver_nsnpp/velocity.pvd')
vtkfile_plus = File('./fp_solver_nsnpp/positive.pvd')
vtkfile_minus = File('./fp_solver_nsnpp/negative.pvd')
vtkfile_phi = File('./fp_solver_nsnpp/phi.pvd')

#Time-stepping
u = Function(V)
p = Function(Y)
n = Function(Y)
phi = Function(Y)

#calculate initial value for phi
solve(a_phi == L_phi_0, phi, bc_phi)
phi_i.assign(phi)

#E = project(grad(phi_i), W)
#plot(E)
#plt.show()

#Saving the initial data
t = 0

vtkfile_u << (u_i, t)
vtkfile_plus << (p_i, t)
vtkfile_minus << (n_i, t)
vtkfile_phi << (phi_i, t)


for i in tqdm(range(num_steps)):

    # Update current time
    t += dt

    #Update time in exact solutions conditions
    u_e.t = t
    p_e.t = t
    n_e.t = t
    phi_e.t = t

    #Set tentative solution to solution at previous time step
    u_.assign(u_i)
    p_.assign(p_i)
    n_.assign(n_i)
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

    #Compute the solution for the velocity field 
    solve(a_u == L_u, u, bc_u)

    #Compute solution for the charges
    solve(a_p == L_p, p, bc_p)
    solve(a_n == L_n, n, bc_n)

    #Calculte the difference of the solution to the tentative estimate for n and p
    error_p = norm(p.vector() - p_.vector(), 'linf')
    error_n = norm(n.vector() - n_.vector(), 'linf')

    #Calculte the difference of the solution to the tentative estimate for phi
    error_phi = errornorm(phi, phi_, 'H10', mesh=mesh)

    #Calculate the difference of the solution to the tentative estimate for u
    error_u = errornorm(u, u_, 'L2', mesh=mesh)

    #Calculte the error
    error = error_u + error_phi + error_p + error_n
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
        u_.assign(u)
        p_.assign(p)
        n_.assign(n)
        phi_.assign(phi)

        #Compute the solution for the electric potential with the tentative charges 
        solve(a_phi == L_phi, phi, bc_phi)

        #Compute the solution for the velocity field 
        solve(a_u == L_u, u, bc_u)

        #Compute solution for the charges
        solve(a_p == L_p, p, bc_p)
        solve(a_n == L_n, n, bc_n)

        #Calculte the difference of the solution to the tentative estimate for n and p
        error_p = norm(p.vector() - p_.vector(), 'linf')
        error_n = norm(n.vector() - n_.vector(), 'linf')

        #Calculte the difference of the solution to the tentative estimate for phi
        error_phi = errornorm(phi, phi_, 'H10', mesh=mesh)

        #Calculate the difference of the solution to the tentative estimate for u
        error_u = errornorm(u, u_, 'L2', mesh=mesh)

        #Calculte the error
        error = error_u + error_phi + error_p + error_n

    #the tentative solution is close enough and becomes the new solution and becomes the new previous solution
    u_i.assign(u)
    p_i.assign(p)
    n_i.assign(n)
    phi_i.assign(phi)

    # Save to file
    vtkfile_u << (u_i, t)
    vtkfile_plus << (p_i,t)
    vtkfile_minus << (n_i,t)
    vtkfile_phi << (phi_i, t)

#Calculate the error norms
p_e_projected = interpolate(p_e, Y)
n_e_projected = interpolate(n_e, Y)
phi_e_projected = interpolate(phi_e, Y)
u_e_projected = interpolate(u_e, V)

diff_u = errornorm(u_i, u_e_projected, 'L2')
diff_p = errornorm(p_i, p_e_projected, 'L2')
diff_n = errornorm(n_i, n_e_projected, 'L2')
diff_phi = errornorm(phi_i, phi_e_projected, 'L2')

#Write the errors to a csv-file for later use
data = {'diff_u': [diff_u],
    'diff_n': [diff_n],
    'diff_p': [diff_p],
    'diff_phi': [diff_phi]
}

df = pd.DataFrame(data, index=['1/64'])
df.to_csv('errors.txt', mode='a', header = False)