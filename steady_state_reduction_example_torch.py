#!/usr/bin/env python3
#
# SSR_code_torch.py | Version 1.0 
#
# PyTorch version of the steady state reduction code for generalized Lotka-Volterra systems
# Original code by Eric Jones and Jean Carlson
#
###############################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import os
from torchdiffeq import odeint

torch.set_printoptions(precision=5)

###############################################################################
##### CONTAINER CLASSES THAT CHARACTERIZE GLV AND SSR SYSTEMS
###############################################################################

class Params:
    """ This container class holds the growth rates (rho), interaction
    parameters (K), and antibiotic efficacies (eps)  associated with a gLV
    system. If no input file is provided, the default parameters from Stein et
    al. are used. rho and eps are Nx1, K is NxN. """
    def __init__(s, my_params=None, filename='stein_parameters.csv'):
        # use your own gLV parameters, if desired
        if my_params:
            s.labels, s.rho, s.K = my_params
        else:
            # Get the directory where this module is located
            module_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Construct absolute paths to the CSV files
            params_path = os.path.join(module_dir, filename)
            ic_path = os.path.join(module_dir, 'stein_ic.csv')
            
            # import "messy" variables and initial conditions from .csv files
            with open(params_path, 'r') as f:
                var_data = [line.strip().split(",") for line in f][1:]
            with open(ic_path, 'r') as f:
                ic_data = [line.strip().split(",") for line in f]

            # turn "messy" data and ICs into variables
            s.labels, s.rho, s.K, s.eps = parse_data(var_data)
            s.ics = parse_ics(ic_data)
            
            # Convert numpy arrays to torch tensors
            s.rho = torch.tensor(s.rho, dtype=torch.float32)
            s.K = torch.tensor(s.K, dtype=torch.float32)
            s.eps = torch.tensor(s.eps, dtype=torch.float32)
            s.ics = torch.tensor(s.ics, dtype=torch.float32)

    def integrand(s, t, Y, u_params=None):
        """ Integrand for the N-dimensional generalized Lotka-Volterra
        equations, using the parameters contained in s. Handles both single states
        and batched states (batch_dim, state_dim). """
        # Handle both single state and batched states
        if Y.dim() == 1:
            # Single state: (state_dim,)
            return (torch.diag(s.rho) @ Y + torch.diag(s.K @ Y) @ Y
                    + s.u(t, u_params) * torch.diag(s.eps) @ Y)
        else:
            # Batched states: (batch_dim, state_dim)
            # Reshape parameters for broadcasting
            rho = s.rho.unsqueeze(0)  # (1, state_dim)
            eps = s.eps.unsqueeze(0)  # (1, state_dim)
            
            # Compute interactions for each batch
            K_Y = torch.matmul(s.K, Y.transpose(0, 1)).transpose(0, 1)  # (batch_dim, state_dim)
            
            # Compute the full dynamics
            return (rho * Y + K_Y * Y + s.u(t, u_params) * eps * Y)

    def u(s, t, u_params):
        """ Returns the concentration of antibiotic currently active """
        try: concentration, duration = u_params
        except TypeError: return torch.tensor(0.0)

        if t < duration:
            return torch.tensor(concentration)
        else: return torch.tensor(0.0)

    def project_to_2D(s, traj, ssa, ssb):
        """Projects a high-dimensional trajectory traj into a 2D system,
        defined by the origin and steady states ssa and ssb, and returns a
        2-dimensional trajectory, following Eq. S18 of the supplement."""
        new_traj = []
        for elem in traj:
            uu = torch.dot(ssa, ssa)
            vv = torch.dot(ssb, ssb)
            xu = torch.dot(elem, ssa)
            xv = torch.dot(elem, ssb)
            uv = torch.dot(ssa, ssb)
            new_traj.append([(xu*vv - xv*uv)/(uu*vv - uv**2),
                             (uu*xv - xu*uv)/(uu*vv - uv**2)])
        new_traj = torch.tensor(new_traj)
        return new_traj

class ssrParams:
    """ This container class holds the parameters for the 2D SSR-reduced
    system, based on the parameters that are passed in p, the steady state ssa,
    and the steady state ssb. The high-dimensional parameters are rho and K;
    the 2D parameters are mu and M """
    def __init__(s, p, ssa, ssb):
        s.mu, s.M = get_SSR_params(p, ssa, ssb)
        s.ssa = ssa
        s.ssb = ssb

    def integrand(s, t, Y):
        """ Integrand for the 2-dimensional generalized Lotka-Volterra
        equations, using the parameters contained in s. Handles both single states
        and batched states (batch_dim, state_dim). """
        # Handle both single state and batched states
        if Y.dim() == 1:
            # Single state: (state_dim,)
            return torch.diag(s.mu) @ Y + torch.diag(s.M @ Y) @ Y
        else:
            # Batched states: (batch_dim, state_dim)
            # Reshape parameters for broadcasting
            mu = s.mu.unsqueeze(0)  # (1, state_dim)
            
            # Compute interactions for each batch
            M_Y = torch.matmul(s.M, Y.transpose(0, 1)).transpose(0, 1)  # (batch_dim, state_dim)
            
            # Compute the full dynamics
            return mu * Y + M_Y * Y

    def get_11_ss(s):
        """ Returns the coexistent steady state (x_a^*, x_b^*) of the 2D gLV
        equations (assuming the equations are not nondimensionalized) """
        xa = - ((-s.M[1][1]*s.mu[0] + s.M[0][1]*s.mu[1]) /
                (s.M[0][1]*s.M[1][0] - s.M[0][0]*s.M[1][1]))
        xb = - ((s.M[1][0]*s.mu[0] - s.M[0][0]*s.mu[1]) /
                (s.M[0][1]*s.M[1][0] - s.M[0][0]*s.M[1][1]))
        return torch.tensor([xa, xb])

###############################################################################
##### HELPER FUNCTIONS
###############################################################################

def parse_data(var_data):
    """ Transforms raw interaction data from the stein_parameters.csv file into
    parameters: labels is the names of each population; mu is the growth rates
    of each population; M[i][j] is the effect of population j on population i;
    eps is the antibiotic susceptibilities of each population"""
    # extract microbe labels, to be placed in legend
    labels = [label.replace("_"," ") for label in var_data[-1] if label.strip()]
    # extract M, mu, and eps from var_data
    str_inter = [elem[1:(1+len(labels))] for elem in var_data][:-1]
    str_gro = [elem[len(labels)+1] for elem in var_data][:-1]
    str_sus = [elem[len(labels)+2] for elem in var_data][:-1]
    float_inter = [[float(value) for value in row] for row in str_inter]
    float_gro = [float(value) for value in str_gro]
    float_sus = [float(value) for value in str_sus]
    # convert to numpy arrays
    M = np.array(float_inter)
    mu = np.array(float_gro)
    eps = np.array(float_sus)
    return labels, mu, M, eps

def parse_ics(ic_data):
    """ Transforms raw initial condition data from the stein_ic.csv file into
    a list of initial conditions (there are 9 experimentally measured initial
    conditions). """
    ic_list_str = [[elem[i] for elem in ic_data][5:-2] for i in \
                    range(1,np.shape(ic_data)[1]) if float(ic_data[3][i])==0]
    ic_list_float = [[float(value) for value in row] for row in ic_list_str]
    ics = np.array(ic_list_float)
    return ics

def solve(p, ic, t_end, interventions={}):
    """ Solves the gLV equations using the parameters given in 'param_list',
    for the scenario specified by 'interventions'. This function also includes
    FMT implementation. u_params = [concentration of dose, duration of dose];
    cd_inoculation = time of CD exposure; transplant_params = [transplant
    composition, transplant size, time of transplantation] """

    # separate 'interventions' parameters into antibiotics (u_params), CD
    # inoculation, and FMT terms
    try: u_params = interventions['u_params']
    except KeyError: u_params = None
    try: cd_inoculation = interventions['CD']
    except KeyError: cd_inoculation = None
    try: transplant_params = interventions['transplant']
    except KeyError: transplant_params = None

    # Convert initial condition to tensor if it's not already
    if not isinstance(ic, torch.Tensor):
        ic = torch.tensor(ic, dtype=torch.float32)

    # integrate with no transplant or CD inoculation
    if (not cd_inoculation) and (not transplant_params):
        t = torch.linspace(0, t_end, 101)
        if not u_params:
            y = odeint(lambda t, y: p.integrand(t, y), ic, t)
        else:
            y = odeint(lambda t, y: p.integrand(t, y, u_params), ic, t)
        return t, y

    # integrate with arbitrary transplant
    if transplant_params:
        t_type, t_size, t_time = transplant_params
        if t_time == 0: t_time = 1e-6
        t01 = torch.linspace(0, t_time, 101)
        t12 = torch.linspace(t_time, t_end, 101)
        y01 = odeint(lambda t, y: p.integrand(t, y, u_params), ic, t01)
        # apply transplant:
        new_ic = y01[-1] + torch.tensor([t_size*x for x in t_type], dtype=torch.float32)
        y12 = odeint(lambda t, y: p.integrand(t, y, u_params), new_ic, t12)

    # integrate with CD inoculation
    if cd_inoculation:
        t01 = torch.linspace(0, cd_inoculation, 101)
        t12 = torch.linspace(cd_inoculation, t_end, 101)
        y01 = odeint(lambda t, y: p.integrand(t, y, u_params), ic, t01)
        # inoculate w/ CD:
        cd_index = p.labels.index("Clostridium difficile")
        cd_transplant = torch.zeros(len(y01[0]))
        cd_transplant[cd_index] = 1e-10
        new_ic = y01[-1] + cd_transplant
        y12 = odeint(lambda t, y: p.integrand(t, y, u_params), new_ic, t12)

    return torch.cat((t01,t12)), torch.cat((y01,y12))

def get_all_stein_steady_states(p):
    """ Numerically generates all five steady states of the Stein model that
    are reachable from any of the nine experimentally measured initial
    conditions. Steady states are stored in the dictionary "ss_list", with keys
    'A' - 'E'.  Here we obtain each steady state by starting at initial
    conditions 0 or 4, exposing or not exposing the initial condition to a
    small amount of CD, and applying or not applying 1 pulse of antibiotics to
    the system.  For details of how these steady states were "found", see Fig 4
    of Jones and Carlson, PLOS Comp. Bio. 2018.  """

    # 'SS attained': (IC num, if CD exposure, if RX applied)
    ss_conditions = {'A': (0, True, False), 'B': (0, False, False),
                     'C': (4, False, False), 'D': (4, True, True),
                     'E': (4, False, True)}
    ss_list = {}
    for ss in ss_conditions:
        ic_num, if_CD, if_RX = ss_conditions[ss]
        ic = p.ics[ic_num]

        if (not if_CD) and (not if_RX): interventions = {}
        if (if_CD) and (not if_RX): interventions = {'CD': 10}
        if (not if_CD) and (if_RX): interventions = {'u_params': (1, 1)}
        if (if_CD) and (if_RX): interventions = {'u_params': (1, 1), 'CD': 5}

        # solve the gLV ODE for the scenario characterized by 'interventions'
        t, y = solve(p, ic, 10000, interventions)
        ss_list[ss] = torch.maximum(y[-1], torch.tensor(0.0))

    return ss_list

def get_SSR_params(p, ssa, ssb):
    """ Given parameters p.rho and p.K, and steady states ssa and ssb, return
    the SSR-generated parameters s.mu and s.M, according to Eqs. 3, A16, and
    A17 of the paper. All parameters are written in terms of the scaled
    variables z_a and z_b as shown in Fig. 2, and as described in Eqs. S20-S22
    of the supplement.  """

    # from Eq 3:
    mu_a = torch.dot(torch.diag(ssa) @ ssa, p.rho)/(torch.norm(ssa)**2)
    mu_b = torch.dot(torch.diag(ssb) @ ssb, p.rho)/(torch.norm(ssb)**2)

    # note these are lacking a factor of norm(ssb) or norm(ssa) as given in
    # Eq 3 since we are working in scaled variables (cf. Eqs S20-22)
    M_aa = ( torch.dot((torch.diag(ssa) @ ssa).T, p.K @ ssa)
             / (torch.norm(ssa)**2) )
    M_bb = ( torch.dot((torch.diag(ssb) @ ssb).T, p.K @ ssb)
             / (torch.norm(ssb)**2) )

    # from Eqs A18 and A19 (complicated cross terms):
    # (ya and yb are as used in the appendix)
    ya = ssa/torch.norm(ssa)
    yb = ssb/torch.norm(ssb)
    numerator = (
        sum([sum([p.K[i][j]*(ya[i]*yb[j] + yb[i]*ya[j])
                  * sum([ya[i]*yb[k]**2 - yb[i]*ya[k]*yb[k]
                         for k in range(len(ssa))])
                  for j in range(len(ssa))])
             for i in range(len(ssa))]) )
    denom = (
        sum([ya[i]**2 for i in range(len(ssa))])
        * sum([yb[i]**2 for i in range(len(ssa))])
        - sum([ya[i]*yb[i] for i in range(len(ssa))])**2 )
    # multiply by norm(ssb) because we are working in scaled variables
    M_ab = numerator*torch.norm(ssb)/denom

    ya = ssa/torch.norm(ssa)
    yb = ssb/torch.norm(ssb)
    numerator = (
        sum([sum([p.K[i][j]*(ya[i]*yb[j] + yb[i]*ya[j])
                  * sum([yb[i]*ya[k]**2 - ya[i]*ya[k]*yb[k]
                         for k in range(len(ssa))])
                  for j in range(len(ssa))])
             for i in range(len(ssa))]) )
    denom = (
        sum([ya[i]**2 for i in range(len(ssa))])
        * sum([yb[i]**2 for i in range(len(ssa))])
        - sum([ya[i]*yb[i] for i in range(len(ssa))])**2 )
    # multiply by norm(ssa) because we are working in scaled variables
    M_ba = numerator*torch.norm(ssa)/denom

    mu = torch.tensor([mu_a, mu_b])
    M = torch.tensor([[M_aa, M_ab], [M_ba, M_bb]])
    return mu, M

def get_separatrix_taylor_coeffs(s, order, dir_choice=1):
    """ Return a dictionary of Taylor coefficients for the unstable or stable
    manifolds of the semistable coexisting fixed point (x_a^*, x_b^*), up to
    order 'order'. Here I let (u, v) = (x_a^*, x_b^*) for notational
    convenience.  dir_choice = 0 returns the unstable manifold coefficients,
    dir_choice = 1 returns the stable manifold coefficients (i.e. dir_choice
    = 1 returns the separatrix). These coefficients are described in Eq 6 of
    the main text. """
    u, v = s.get_11_ss()
    coeffs = torch.zeros(order)
    for i in range(order):
        if i == 0:
            coeffs[i] = v
            continue
        if i == 1:
            a = s.M[0][1]*u
            b = s.M[0][0]*u - s.M[1][1]*v
            c = -s.M[1][0]*v
            if dir_choice == 0:
                lin_val = (-b + torch.sqrt(b**2 - 4*a*c))/(2*a)
            else:
                lin_val = (-b - torch.sqrt(b**2 - 4*a*c))/(2*a)
            coeffs[i] = lin_val
            continue
        if i == 2:
            # Eq 37 of supplement. In my terms:
            # c_m/m! * alpha = c_m-1/(m-1)! * beta
            alpha = i*u*s.M[0][0] + (i+1)*u*s.M[0][1]*coeffs[1] - s.M[1][1]*v
            beta = ( s.M[1][0] + s.M[1][1]*coeffs[1] - (i-1)*s.M[0][0] -
                     (i-1)*s.M[0][1]*coeffs[1] )
            i_coeff = ( math.factorial(i) *
                        (coeffs[i-1]/math.factorial(i-1)*beta) ) / alpha
            coeffs[i] = i_coeff
            continue
        # Eq 38 of supplement. In my terms:
        # c_m/m! * alpha = c_m-1/(m-1)! * beta + sum_i=2^m-1 gamma[i]
        alpha = i*u*s.M[0][0] + (i+1)*u*s.M[0][1]*coeffs[1] - s.M[1][1]*v
        beta = ( s.M[1][0] + s.M[1][1]*coeffs[1] - (i-1)*s.M[0][0] -
                 (i-1)*s.M[0][1]*coeffs[1] )
        gamma = torch.sum(torch.tensor([ (coeffs[j]/(math.factorial(j) * math.factorial(i - j))
                          * (s.M[1][1]*coeffs[i-j]
                             - (i-j)*s.M[0][1]*coeffs[i-j]
                             - u*s.M[0][1]*coeffs[i-j+1]))
                        for j in range(2, i)]))

        i_coeff = ( i / alpha * coeffs[i-1]*beta
                    + math.factorial(i) / alpha * gamma)
        coeffs[i] = i_coeff
    return coeffs

def get_lower_bound_of_separatrix(p, s, x_min, x_max, y_min, y_max,
                                  num_sections):
    """ Part of an iterative algorithm that numerically computes the
    separatrix, but uses adaptive sampling to only compute points nearby the
    separatrix (instead of far away from it). The range of points that you are
    investigating is [x_min, x_max, y_min, y_max]. This program assumes that
    at x_min the separatrix is larger than y_min, and at x_max it is smaller
    than y_max. This region will be cut along the x-axis into num_sections
    number of sections, and the y-axis will be cut so that the subregions are
    all squares. This function returns a list of points (that are ordered in
    the x-value) that delineate the lower bound of the separatrix, as well as
    the spatial step-size delta that was used to generate the subregions. """

    delta = (x_max - x_min)/num_sections # spatial variation
    N = num_sections + 1 # number of points needed to make num_sections
    xs = torch.linspace(x_min, x_max, N)
    num_y_points = int(((y_max - y_min)/delta) + 1)
    ys = torch.linspace(y_min, y_max, num_y_points)

    ssa = s.ssa
    ssb = s.ssb

    eps = 1e-3
    outcome_dict = {}
    for x in xs:
        for y in ys:
            ic = x*ssa + y*ssb
            t_out, y_out = solve(p, ic, 1000)
            ss = y_out[-1]

            if torch.norm(ss - ssa) < eps:
                outcome_dict[x.item(), y.item()] = 'a'
            elif torch.norm(ss - ssb) < eps:
                outcome_dict[x.item(), y.item()] = 'b'
            elif x == 0 and y == 0:
                # edge case
                outcome_dict[x.item(), y.item()] = 'a'
            else:
                outcome_dict[x.item(), y.item()] = '?'

    lower_bound_list = []
    for x in xs:
        flag = False
        for y in ys.flip(0):
            if flag is False:
                flag = outcome_dict[x.item(), y.item()]
            if flag == outcome_dict[x.item(), y.item()]:
                continue
            if flag != outcome_dict[x.item(), y.item()]:
                lower_bound_list.append([x.item(), y.item()])
                flag = outcome_dict[x.item(), y.item()]

    return lower_bound_list, delta

###############################################################################
##### FUNCTIONS THAT PRODUCE PLOTS
###############################################################################

def plot_original_and_SSR_trajectories(p, s, ax=None):
    """ Plots a SSR trajectory (characterized by SSR params 's') with initial
    condition ic_2d. Also plots the in-plane projection of the high-dimensional
    trajectory that starts at the corresponding initial condition in the
    high-dimensional space, as in Fig 2 of the main text. """
    if not ax:
        ax = plt.gca()

    for i,ic in enumerate([[.5, .5], [.1, .1], [.8, .2], [.9, .1]]):
        ic_2d = torch.tensor(ic, dtype=torch.float32)
        ic_high = ic_2d[0]*s.ssa + ic_2d[1]*s.ssb

        t_2d, traj_2d = solve(s, ic_2d, 50)
        t_high, traj_high = solve(p, ic_high, 50)
        # project high-dimensional trajectory into the plane spanned by ssa and ssb
        traj_high_proj = p.project_to_2D(traj_high, s.ssa, s.ssb)

        ax.plot(traj_2d[0,0].item(), traj_2d[0,1].item(), 'k.', zorder=3, ms=18)
        if i == 0:
            ax.plot(traj_2d[:,0].numpy(), traj_2d[:,1].numpy(), color='blue', label='2D trajectory')
            ax.plot(traj_high_proj[:,0].numpy(), traj_high_proj[:,1].numpy(), color='orange',
                    label='high-dimensional trajectory')
        else:
            ax.plot(traj_2d[:,0].numpy(), traj_2d[:,1].numpy(), color='blue')
            ax.plot(traj_high_proj[:,0].numpy(), traj_high_proj[:,1].numpy(), color='orange')

    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])

    plt.savefig('SSR_demo_1.pdf')
    print('... saved figure to SSR_demo_1.pdf')

    return ax

def plot_2D_separatrix(p, s, ax=None):
    """ Plots the 2D separatrix, as given in Eq 6 of the main text """
    if not ax:
        ax = plt.gca()

    taylor_coeffs = get_separatrix_taylor_coeffs(s, order=100, dir_choice=1)
    # (u, v) is the coexistent fixed point (x_a^*, x_b^*) of the 2D system
    u, v = s.get_11_ss()
    xs = torch.linspace(0, 1, 100)
    ys = torch.tensor([sum( [float(taylor_coeffs[j])*(xx-u)**j/math.factorial(j)
                for j in range(len(taylor_coeffs))] ) for xx in xs])

    ax.plot(xs.numpy(), ys.numpy(), color='k', lw=4, label='2D separatrix')
    plt.savefig('SSR_demo_2.pdf')
    print('... saved figure to SSR_demo_2.pdf')

    return ax

def plot_ND_separatrix(p, s, ax=None, sep_filename='11D_separatrix_1e-2.data',
                       color='b', y_max=2, label='separatrix', delta=0.01,
                       save_plot=False):
    """ Plots the in-plane 11D separatrix, which shows which steady state a
    point on the plane will tend towards. Sampling of points is done with a
    bisection-like method. """

    goal_delta = delta

    if not ax:
        ax = plt.gca()

    prev_lower_bound_list, delta = (
        get_lower_bound_of_separatrix(p, s, x_min=0, x_max=1, y_min=0,
                                      y_max=y_max, num_sections=2) )
    prev_delta = delta

    # to save a calculated separatrix to the file, set load_data = True
    # to change the resolution of the calculated separatrix change delta
    load_data = True 

    if load_data:
        with open(sep_filename, 'rb') as f:
            separatrix_lower_bound = pickle.load(f)
    else:
        print('RESOLUTION, NUMBER OF BISECTIONS')
        while delta > goal_delta:
            cumulative_lower_bound_list = []
            for i in range(len(prev_lower_bound_list)-1):
                x_min = prev_lower_bound_list[i][0]
                x_max = prev_lower_bound_list[i+1][0]
                y_min = prev_lower_bound_list[i][1]
                y_max = prev_lower_bound_list[i+1][1] + prev_delta
                lower_bound_list, delta = (
                    get_lower_bound_of_separatrix(p, s, x_min=x_min, x_max=x_max,
                                        y_min=y_min, y_max=y_max, num_sections=2) )
                try:
                    if lower_bound_list[0] == cumulative_lower_bound_list[-1]:
                        cumulative_lower_bound_list.extend(lower_bound_list[1:])
                except IndexError:
                    cumulative_lower_bound_list.extend(lower_bound_list)
            prev_lower_bound_list = cumulative_lower_bound_list
            prev_delta = delta
            print(delta, len(cumulative_lower_bound_list))

            separatrix_lower_bound = torch.tensor(cumulative_lower_bound_list)

        separatrix_lower_bound = torch.tensor(cumulative_lower_bound_list)
        with open(sep_filename, 'wb') as f:
            pickle.dump(separatrix_lower_bound, f)
        print('(reduce computation time by setting load_data = True')

    ax.plot(separatrix_lower_bound[:,0].numpy(), separatrix_lower_bound[:,1].numpy(),
            color=color, label=label)
    if label:
        ax.legend()

    ax.legend(fontsize=12)
    if save_plot:
        plt.savefig('SSR_demo_3.pdf')
        print('... saved figure to SSR_demo_3.pdf')

    return ax

def get_dynamics_function():
    """Returns a function that evaluates p.integrand at t=0.
    
    Returns:
        callable: Function that takes state y and returns dy/dt at t=0
    """
    p = Params()
    def dynamics(y):
        return p.integrand(0.0, y)
    return dynamics

def get_ssr_dynamics_function():
    """Returns a function that evaluates the SSR dynamics at t=0.
    
    Returns:
        callable: Function that takes state y and returns dy/dt at t=0 for the SSR system
    """
    p = Params()
    ssr_steady_states = get_all_stein_steady_states(p)
    s = ssrParams(p, ssr_steady_states['E'], ssr_steady_states['C'])
    
    def dynamics(y):
        return s.integrand(0.0, y)
    return dynamics

def get_attractors():
    """Returns a tensor containing all steady states (attractors) for the system.
    
    Returns:
        torch.Tensor: Tensor of shape (n_attractors, state_dim) containing all steady states
    """
    p = Params()
    ssr_steady_states = get_all_stein_steady_states(p)
    # Convert dictionary of numpy arrays to list of tensors
    tensor_steady_states = [
        torch.tensor(v, dtype=torch.float32) 
        for v in ssr_steady_states.values()
    ]
    # Stack into single tensor
    return torch.stack(tensor_steady_states)

def get_attractors_ssr():
    """Returns a tensor containing the steady states for the SSR system.
    
    Returns:
        torch.Tensor: Tensor of shape (2, 2) containing the SSR steady states [[1,0], [0,1]]
    """
    return torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)




###############################################################################
##### MAIN FUNCTION
###############################################################################

if __name__ == '__main__':
    func = get_dynamics_function()
    func(torch.ones((10,11)))
    func = get_ssr_dynamics_function()
    func(torch.ones(10,2))
    # # load gLV parameters (p.rho, p.K, p.eps) from Stein et al
    # p = Params()
    #
    # # generate every reachable steady state (labeled A-E) of the Stein model
    # ssr_steady_states = get_all_stein_steady_states(p)
    # # print each steady state:
    # #[print(x, ssr_steady_states[x]) for x in ssr_steady_states]
    #
    # # generate SSR params (s.mu, s.M) using steady states E and C
    # s = ssrParams(p, ssr_steady_states['E'], ssr_steady_states['C'])
    #
    # # plot corresponding high-dimensional and SSR-reduced trajectories
    # ax = plot_original_and_SSR_trajectories(p, s)
    # # plot the 2D separatrix (analytically generated)
    # # ax = plot_2D_separatrix(p, s, ax)
    # # plot the ND separatrix (numerically generated)
    # # NOTE: due to the algorithm for computing this separatrix, ensure that the
    # # more unstable steady state (i.e. the steady state with a smaller basin of
    # # attraction) is listed first in the above ssrParams function; if an error
    # # occurs, try switching the order of the two steady states
    # # ax = plot_ND_separatrix(p, s, ax, save_plot=True, label='high-dimensional separatrix', color='grey')
    #
    # # Generate random initial conditions
    # ic = torch.rand(20, 11)
    # traj1,traj2 = solve(p, ic, 1000)
    #
    #
    # plt.figure()
    # plt.plot(traj2[:,:,10].numpy())
    # plt.show()
    #
    # # Perform PCA on traj2
    # from sklearn.decomposition import PCA
    #
    # # Reshape traj2 to 2D array (samples x features)
    # n_timesteps, n_trajectories, n_species = traj2.shape
    # X = traj2.reshape(n_trajectories * n_timesteps, n_species).numpy()
    #
    # # Fit and transform PCA
    # pca = PCA(n_components=2)
    # X_pca = pca.fit_transform(X)
    #
    # # Reshape back to trajectory format
    # X_pca = X_pca.reshape(n_timesteps, n_trajectories, 2)
    #
    # # Plot PCA trajectories
    # plt.figure(figsize=(8,6))
    # # for trajectory in X_pca:
    # plt.plot(X_pca[:,:,0], X_pca[:,:,1], alpha=0.5)
    # # Plot initial points in red
    # plt.scatter(X_pca[0,:,0], X_pca[0,:,1], c='green', label='Initial points', s=50)
    #
    # # Plot final points in blue
    # plt.scatter(X_pca[-1,:,0], X_pca[-1,:,1], c='red', label='Final points', s=50)
    # # Plot steady states
    # for state_name, steady_state in ssr_steady_states.items():
    #     if state_name == 'E' or state_name == 'C':
    #         steady_state_pca = pca.transform(steady_state.reshape(1, -1).numpy())
    #         plt.scatter(steady_state_pca[:,0], steady_state_pca[:,1],
    #                    marker='*', s=200, label=f'Steady State {state_name}',alpha=0.3)
    #
    # plt.legend()
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.title('PCA of Species Trajectories')
    # plt.show()
    #
    #
    #
    #