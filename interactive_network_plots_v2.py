import itertools
import math
import numpy as np
import os
import pandas as pd
import pathlib
import sys

# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# EQTK to calculate equilibrium concentrations - install instructions here https://eqtk.github.io/getting_started/eqtk_installation.html
import eqtk

# One can bound figure attributes to other widget values.
import ipywidgets

# Plotting settings
rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size":9,
    "axes.titlesize":9,
    "axes.labelsize":9,
    "xtick.labelsize":9,
    "ytick.labelsize":9,
    "savefig.dpi": 900, 
    'figure.figsize': [6.0, 4.0],
    'figure.dpi': 150,
    'pdf.fonttype':42,
    'ps.fonttype':42,
})

def num_combos(n,r):
    # Number of combinations of n items with r items per group
    return int(math.factorial(n)/(math.factorial(r)*math.factorial(n-r)))

def num_combos_with_replacement(n,r):
    # Number of combinations with replacement of n items with r items per group
    return int(math.factorial(n+r-1)/(math.factorial(r)*math.factorial(n-1)))

def number_of_dimers(m):
    """
    Calculate the number of distinct dimers from input number (m) of monomers.
    """
    return int(m*(m+1)/2)

def number_of_heterodimers(m):
    """
    Calculate the number of distinct heterodimers from input number (m) of monomers.
    """
    return int(m*(m-1)/2)

def number_of_species(m):
    """
    Calculate the number of monomers + dimers from input number (m) of monomers
    """
    return m + number_of_dimers(m)

def make_nXn_species_names(m):
    """
    Enumerate names of monomers and dimers for ordering stoichiometry matrix of nXn rxn network

    Parameters
    ----------
    m : int
        Number of monomer species.

    Returns
    -------
    names : list, length (m(m+1)/2 + m)
        List where each element (string) represents a reacting species.
    """
    monomers = [f'M_{i+1}' for i in range(m)]
    combinations = itertools.combinations_with_replacement(range(m), 2)
    dimers = [f'D_{i+1}_{j+1}' for i,j in combinations]
    return monomers + dimers

def make_Kij_names(m, n_input = 2, rxn_ordered = True):
    """
    Create Kij names for ordering parameters
    """
    n_accesory = m - n_input
    if rxn_ordered:
        return [f'K_{i[0]}_{i[1]}' for i in itertools.combinations_with_replacement(range(1, m+1), 2)]
    else:
        names = []
        #add input homodimers
        names.extend([f'K_{i}_{i}' for i in range(1, n_input+1)])
        #add input heterodimers
        names.extend([f'K_{i[0]}_{i[1]}' for i in itertools.combinations(range(1, n_input+1), 2)])
        #add input-acc heterodimers
        names.extend([f'K_{i[0]}_{i[1]}' for i in itertools.product(range(1, n_input+1),
                                                                  range(n_input+1, n_input+n_accesory+1))])
        #add accessory homodimers
        names.extend([f'K_{i}_{i}' for i in range(n_input+1, n_input+n_accesory+1)])
        #add accessory heterodimers
        names.extend([f'K_{i[0]}_{i[1]}' for i in itertools.combinations(range(n_input+1, n_input+n_accesory+1), 2)])

        return names
    
def make_nXn_dimer_reactions(m):
    """
    Generate all pairwise reactions for n monomer species to be parsed by EQTK.

    Parameters
    ----------
    m : int
        Number of monomer species.

    Returns
    -------
    reactions : string
        Set of dimerization reactions for specified numbers of monomers, one reaction
        per line.
    """
    combinations = list(itertools.combinations_with_replacement(range(m), 2))
    reactions = [f'M_{i+1} + M_{j+1} <=> D_{i+1}_{j+1}\n' for i,j in combinations]
    return ''.join(reactions).strip('\n')


def make_nXn_stoich_matrix(m):
    """
    For the indicated number of monomers, generate stochiometry matrix for dimerization reactions.
    Parameters
    ----------
    m : int
        Number of monomer species.

    Returns
    -------
    N : array_like, shape (m * (m-1)/2 + m, m(m+1)/2 + m)
        Array where each row corresponds to distinct dimerization reaction
        and each column corresponds to a reaction component (ordered monomers then dimers)
    """
    reactions = make_nXn_dimer_reactions(m)
    names = make_nXn_species_names(m)
    N = eqtk.parse_rxns(reactions)
    return N[names].to_numpy()


def make_C0_grid(m=2, M0_min=-3, M0_max=3, num_conc=10):
    """
    Construct grid of initial monomer and dimer concentrations.
    Initial dimer concentrations set to 0.

    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.
    M0_min : array_like, shape (m,) or (1,)
        Lower limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^-3.
    M0_max : array_like, shape (m,) or (1,)
        Upper limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is 3,
        corresponding to a lower limit of 10^3.
    num_conc : array_like, shape (m,) or (1,)
        Number of concentrations for each monomer, sampled logarithmically.
        Scalar values set the same limit for all ligands. Default is 10.

    Returns
    -------
    C0 :  array_like, shape (numpy.product(n_conc), (m(m+1)/2 + m)) or (n_conc ** m, (m(m+1)/2 + m))
        Each row corresponds to distinct set of species concentrations.

    """
    num_dimers = number_of_dimers(m)
    if np.size(M0_min) == 1 and np.size(M0_max) == 1 and np.size(num_conc) == 1:
        titration = [np.logspace(M0_min, M0_max, num_conc)]*m
    elif np.size(M0_min) == m and np.size(M0_max) == m and  np.size(num_conc) == m:
        titration = [np.logspace(M0_min[i], M0_max[i], num_conc[i]) for i in range(m)]
    else:
        raise ValueError('Incorrect size of M0_min, M0_max, or num_conc.')
    titration = np.meshgrid(*titration, indexing='ij')
    M0 = np.stack(titration, -1).reshape(-1,m)
    return np.hstack((M0, np.zeros((M0.shape[0], num_dimers))))


def run_eqtk(N, C0, params, acc_monomer_ind):
    """
    Run eqtk.solve given the input stoichiometry matrix, initial concentrations, and parameters.

    See eqtk.solve documentation for more details on the syntax.
    (https://eqtk.github.io/user_guide/generated/highlevel/eqtk.solve.html)

    Parameters
    ----------
    N : Array-like, shape (num_rxns, num_species)
        Input stoichiometry matrix
    C0 : Array-like, shape (num_simulation_points, num_species)
        Initial concentrations of all species for each simulation point.
        Accessory monomer levels will be set based on params.
    params : List-like, len num_combos_with_replacement(m,2) + (m-1)
        Parameters including Kij values and accessory monomer levels
    acc_monomer_ind : int  
        Index of accessory monomers in species list
    
    Returns
    -------
    C: Array-like
        Equilibrium concentrations of all species
    """
    num_rxns = N.shape[0]
    # Extract Kij values
    K = params[:num_rxns]
    # Set accessory monomer levels
    C0[:,acc_monomer_ind] = params[num_rxns:]

    return eqtk.solve(c0=C0, N=N, K=K)

def simulate_networks(m, num_inputs, param_sets, t = 12, input_lb = -3, input_ub = 3):
    """
    Run simulations for dimer networks, over a titration of concentrations for each input monomer. 
    When >1 input monomers are varied (e.g., a 2D matrix), the first monomer is sampled outermost, 
    while the second is sampled inner to that.
    
    Parameters
    ----------
    m : int. 
        Number of monomer species in the network.
    num_inputs: int
        Number of monomers to count as inputs (those titrated). 
        E.g., if num_inputs=2, the first two monomers will be treated as inputs.
    t : int. Default 12. 
        Number of values to titrate each input monomer species. 
        Values spaced evenly on a log10 scale
    input_lb : int. Default -3
        lower bound for titrating the input monomer species. Log10 scale
    input_ub : int. Default 3
        upper bound for titrating the input monomer species.Llog10 scale
    param_sets : Array-like, shape (num_sets, num_parameters)
        Parameter sets for simulating multiple dimerization networks.  
    Returns
    -------
    C0 : Array-like, shape (t, number of species)
        Initial concentration array used for eqtk.solve
    S_all : Array-like, shape (t, number of species, num_sets) 
        Equlibrium concentration of all species
        for all parameter sets and each input titration point.
        
    """
    # Create stoichiometry matrix 
    num_sets = param_sets.shape[0]
    N = make_nXn_stoich_matrix(m)
    num_rxns = N.shape[0]

    # Create initial concentration array
    M0_min = [input_lb]*num_inputs + [0] * (m-num_inputs) # Species concentrations at min of inputs
    M0_max = [input_ub]*num_inputs + [0] * (m-num_inputs) # Species concentrations at max of inputs
    num_conc = [t]*num_inputs + [1] * (m-num_inputs) # Number of concentrations to titrate for each species
    C0 = make_C0_grid(m, M0_min=M0_min, M0_max=M0_max, num_conc=num_conc) 
        
    acc_monomer_ind = np.arange(num_inputs,m) # Indices of accessory monomers
    S_all = np.zeros((C0.shape[0], C0.shape[1], num_sets))
    # For each parameter set, run eqtk.solve
    for pset_index, pset in enumerate(param_sets):
        S_all[:,:,pset_index] = run_eqtk(N, C0.copy(), pset, acc_monomer_ind)
    return C0,S_all  

def calculate_object_size(input_,lower_input_bound,upper_input_bound,lower_size_bound,upper_size_bound):
    '''
    Takes some input, and uses some input bounds to rescale that input to some output bounds (such as width of a line in points).
    '''
    if input_<lower_input_bound:
        return 0
    elif input_>upper_input_bound:
        return upper_size_bound
    else:
        return lower_size_bound+((input_-lower_input_bound)/(upper_input_bound-lower_input_bound))*(upper_size_bound-lower_size_bound)

def get_poly_vertices(n, r = 1, dec = 3, start = math.pi/2):
    """
    Get x and y coordinates of n-polygon with radius r.
    """
    #This could be broadcast with numpy
    #i.e. x = r * np.cos(2*np.pi * np.arange(1,n+1)/n)
    #but I think it's easier to follow as list comprehension
    x = np.array([round(r * math.cos((2*math.pi*i/n)+start), dec) for i in range(n)])
    y = np.array([round(r * math.sin((2*math.pi*i/n)+start), dec) for i in range(n)])
    return x,y

def make_network_nodes_polygon(m, r, n_input,start_angle):
    '''
    Make a dataframe of nodes with columns for species name, type, and x and y coordinates
    '''
    x, y = get_poly_vertices(m, r=r,start=start_angle)
    species = [f'M_{i}' for i in range(1,m+1)]
    species_type = ['input']*n_input + ['accessory']*(m - n_input)
    node_df = pd.DataFrame({'species': species, 'species_type':species_type, 
                            'x': x, 'y': y})
    return node_df

def make_self_edges(m, r_node, r_edge,start_angle):
    '''
    Make a dataframe of self-edges with columns for x and y coordinates and \
    Kij name
    '''
    edge_df_list = [0] * m
    x, y = get_poly_vertices(m, r=r_node+r_edge,start=start_angle)
    # weights_scaled = np.array(K)*edge_scale
    for i in range(m):
        # Set center of self-edge to be r_edge further from the origin
        angle =  math.atan2(y[i],x[i])
        y_new = (1+r_edge)*np.sin(angle)
        x_new = (1+r_edge)*np.cos(angle)
        x_new = x[i]
        y_new = y[i]
        center = [[x_new, y_new]]
        tmp_df = pd.DataFrame(center, columns=['x', 'y'])
        tmp_df['Kij_names'] = [f'K_{i+1}_{i+1}']
        edge_df_list[i] = tmp_df
        
    return pd.concat(edge_df_list)

def make_heterodimer_edges(m, node_df):
    '''
    Make a dataframe of heterodimer edges with columns for Kij name and x and y coordinates
    '''
    pairs = itertools.combinations(range(m), 2)
    n_heterodimers = number_of_heterodimers(m)
    x = [0]*n_heterodimers
    x_end = [0]*n_heterodimers
    y = [0]*n_heterodimers
    y_end = [0]*n_heterodimers
    names = [0]*n_heterodimers
    for i, comb in enumerate(pairs):
        x[i] = node_df.loc[comb[0],'x']
        x_end[i] = node_df.loc[comb[1],'x']
        y[i] = node_df.loc[comb[0],'y']
        y_end[i] = node_df.loc[comb[1],'y']
        names[i] = f'K_{comb[0]+1}_{comb[1]+1}'

    edge_df = pd.DataFrame({'Kij_names': names,
                          'x': x, 'x_end': x_end,
                          'y': y, 'y_end': y_end})
    return edge_df


def get_circle_intersections(x0, y0, r0, x1, y1, r1):
    '''
    Get the intersection points of two circles
    FROM: https://stackoverflow.com/questions/55816902/finding-the-intersection-of-two-circles
    circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1
    '''
    d=math.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    # non intersecting
    if d > r0 + r1 :
        return None
    # One circle within other
    if d < abs(r0-r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a=(r0**2-r1**2+d**2)/(2*d)
        h=math.sqrt(r0**2-a**2)
        x2=x0+a*(x1-x0)/d   
        y2=y0+a*(y1-y0)/d   
        x3=x2+h*(y1-y0)/d     
        y3=y2-h*(x1-x0)/d 

        x4=x2-h*(y1-y0)/d
        y4=y2+h*(x1-x0)/d
        
        return [[x3, y3], [x4, y4]]


def make_network_plots_polygon(fig,ax,m, n_input, univs_to_plot, param_sets, dimers_of_interest=None,input_node_values = np.array([0]), ncols = 1, r_node = 1, r_loop = 0.5,
                            node_scales = [-3,3,5,50], K_edge_scales = [-5,7,4,10],saveto='',input_cmap='Pastel1',\
                            fontname = 'Helvetica',fontsize=16,non_output_dimer_color='gray',labels=True,dpi=72,\
                            upscale_arrowhead = 1.2,node_edge_width=0.5,padding=0.1):
    """
    Load a subset of networks from a parameter file and plot the affinity parameters between monomers.
    
    Parameters
    --------------
    m: Int
        Number of total monomers in the network.
    n_input: Int
        Number of input monomers
    univs_to_plot: array-like
        Array of which universes (parameter sets) to plot.
    param_sets: Array-like, shape (n_univ, n_parameters)
        Array of parameters to plot. Alternatively, can use:
    dimers_of_interest: array-like or None
        If specified, will draw the arrows for dimers of interest in black and all others in gray
    input_node_values: Array-like of len n_input
        Parameter value(s), in log10-scale, to use for the input monomer expression level(s).
    ncols: Int
        Number of columns to use for multiple subplots.
    node_scales: List
        Scaling information for the sizes of the nodes. Defines marker size, the square of the marker 
        diameter in points. Note that for input, the value is just one number (size to use).
        Values: [lower value bound (log10), upper value bound (log10), lower size bound, upper size bound]
    K_edge_scales: Dict
        Scaling information for the widths of the edges in points.
        Values: [lower value bound (log10), upper value bound (log10), lower size bound, upper size bound]
    saveto: str
        Directory to save the figure to, without extension. If '', will not save.
        Saves both pdf and png
    fontname: str. Name of font to use.
    fontsize: Int. Size of the labels.
    non_output_dimer_color: str. Color used for non-output dimers.
    labels: Bool. Whether to label the nodes. Default True.
    dpi: Int. Resolution of the figure, default 72. This will significantly affect the proportions of the figure.
    upscale_arrowhead: Float. Multiplied factor to determine the length and width of the arrowheads from the linewidth.
    node_edge_width: Float. Width of node edges in points.
    padding: Float. Extra space to add around each edge of the figure in inches.

    Returns:
        fig, axs: Created plot
    """
    param_sets = param_sets[univs_to_plot,:]
    num_plots = len(univs_to_plot)
    species_names = np.array(make_nXn_species_names(m))
    dimer_names = species_names[m:]
    Kij_labels = make_Kij_names(n_input = n_input, m=m)
    num_rxns = len(Kij_labels)
    input_node_names = ["M_{}".format(i+1) for i in range(n_input)]

    # Make dataframe containing node positions. Color accessory monomer nodes and scale size by parameter value
    
    node_df_list = [0]*num_plots
    for i in range(num_plots):
        #scale acc monomer weights
        acc_weights = np.log10(param_sets[i,num_rxns:])
        node_df_list[i] = make_network_nodes_polygon(m=m, r=r_node, n_input=n_input,\
                                                     start_angle=((math.pi/2)-((n_input-1)*(2*math.pi/(2*m)))))
    node_df_combined = pd.concat(node_df_list, keys=univs_to_plot).reset_index()
    node_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)

    node_weights = param_sets[np.arange(num_plots), num_rxns:]
    node_weights = np.hstack((np.tile(10**(input_node_values.astype(np.float64)),reps=(num_plots,1)), node_weights))
    node_weights = np.log10(node_weights)

    node_df_combined['weight'] = node_weights.flatten()

    #Make dataframe for self-loops. Scale width by Kii value
    self_edge_df = make_self_edges(m, r_node, r_loop,\
                                                     start_angle=((math.pi/2)-((n_input-1)*(2*math.pi/(2*m)))))
    self_edge_df_combined = pd.concat([self_edge_df]*num_plots, keys=np.arange(num_plots)).reset_index() # Combine dfs
    self_edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    self_edge_labels = [f'K_{i}_{i}' for i in range(1,m+1)] # Add labels
    self_edge_index = np.where(np.isin(Kij_labels, self_edge_labels))[0]

    self_edge_weights = param_sets[np.arange(num_plots)[:,np.newaxis], self_edge_index] # Get weights
    self_edge_weights = np.log10(self_edge_weights)

    self_edge_df_combined['weight'] = np.repeat(self_edge_weights.flatten(), self_edge_df_combined.level_1.max()+1)
    
    #Make dataframe for heterodimer edges. Scale width by Kij value
    hetero_edge_df = make_heterodimer_edges(m, node_df_combined)
    hetero_edge_df_combined = pd.concat([hetero_edge_df]*num_plots, keys=np.arange(num_plots)).reset_index()
    hetero_edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    hetero_edge_index = np.where(~np.isin(Kij_labels, self_edge_labels))[0]
    hetero_edge_weights = param_sets[np.arange(num_plots)[:,np.newaxis], hetero_edge_index]
    hetero_edge_weights = np.log10(hetero_edge_weights)
    hetero_edge_df_combined['weight'] = hetero_edge_weights.flatten()
    
    if type(input_cmap)==str:
        cmap = plt.get_cmap(input_cmap)
    else:
        cmap = input_cmap.copy()

    if len(univs_to_plot)==1:
        ncols = 1
    
    nrows = math.ceil(len(univs_to_plot)//ncols)
    figsize = (2*(r_node+(2*r_loop)+padding)*ncols,2*(r_node+(2*r_loop)+padding)*nrows)
    # Create subplots but with no padding
    # fig, axs = plt.subplots(nrows=nrows, ncols=ncols,figsize=figsize,dpi=dpi,squeeze=False,\
    #                     gridspec_kw={'hspace':0,'wspace':0,'left':0,'right':1,'top':1,'bottom':0})

    # for univ in range(len(univs_to_plot)):
    univ=0
    row = univ//ncols
    col = univ%ncols
    # ax = axs[row,col]
    ax.axis('off') # Hide axes
    ax.set_xlim(-(r_node+(2*r_loop)+padding),(r_node+(2*r_loop)+padding))
    ax.set_ylim(-(r_node+(2*r_loop)+padding),(r_node+(2*r_loop)+padding))
    # Calculate conversion between point size and data coordinates
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
    yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
    x_points_per_coord = (bbox.width*fig.get_dpi())/xrange
    y_points_per_coord = (bbox.height*fig.get_dpi())/yrange
    points_per_coord = np.mean([x_points_per_coord,y_points_per_coord])
    # FOR HOMODIMERS
    for i, edge in self_edge_df_combined.iterrows():
        if dimers_of_interest is not None:
            if Kij_labels.index(edge['Kij_names']) in dimers_of_interest[univ]:
                arrow_color = 'k'
                zorder = 3
            else:
                arrow_color = non_output_dimer_color
                zorder = 1
        else:
            arrow_color = 'k'
            zorder = 1
        node_name = "M_{}".format(edge["Kij_names"].split('_')[1])
        # Coordinate of node center
        node_coord = np.array(node_df_combined.query('species == @node_name').reset_index(drop=True).loc[0,['x','y']])
        # Node radius in coordinate units
        node_radius = (calculate_object_size(node_df_combined.query('species == @node_name').reset_index(drop=True).loc[0,'weight'],\
            node_scales[0],node_scales[1],node_scales[2],node_scales[3])/2)/points_per_coord
        # Coordinate of center of the "self_edge" 
        self_edge_coord = np.array([edge['x'],edge['y']])
        # Write node name
        if labels:
            if node_name in input_node_names:
                ax.text(edge['x'],edge['y'],node_name.replace('_',''),c='k',ha='center',va='center',fontsize=fontsize,fontweight='bold',fontname=fontname)
            else:
                ax.text(edge['x'],edge['y'],node_name.replace('_',''),c=non_output_dimer_color,ha='center',va='center',fontsize=fontsize,fontweight='bold',fontname=fontname)
        # Draw arrow
        # Arrow size in pts (line width)
        weight = calculate_object_size(edge['weight'],K_edge_scales[0],K_edge_scales[1],K_edge_scales[2],K_edge_scales[3])
        if weight==0:
            continue
        # Arrow size in coordinate units
        arrow_size = weight*2/points_per_coord
        # intersection of the node circle and self-arrow
        circle_intersections = get_circle_intersections(node_coord[0],node_coord[1],node_radius,self_edge_coord[0],self_edge_coord[1],r_loop)
        if circle_intersections is not None:
            # Arc angle filled by arrowhead
            arrow_arc_angle = arrow_size/r_loop
            # Vector from the center of the "self-edge" circle to the node center
            self_edge_center_to_node = (node_coord-self_edge_coord)/np.linalg.norm(node_coord-self_edge_coord)
            # Angle from the center of the "self-edge" circle to the node center
            node_angle_from_self_edge_center = math.atan2(self_edge_center_to_node[1],self_edge_center_to_node[0])
            # Calculate angles formed between each intersection point and the self-edge center
            norm_relative_intersect_points = [(coord-self_edge_coord)/np.linalg.norm((coord-self_edge_coord),ord=2) for coord in circle_intersections]
            intersect_angles = [math.atan2(coord[1],coord[0]) for coord in norm_relative_intersect_points]
            # If angle is right at the edges of the angle range, convert
            if (intersect_angles[0]<=np.pi/2 and intersect_angles[1]>=np.pi/2) or \
                (intersect_angles[1]<=np.pi/2 and intersect_angles[0]>=np.pi/2):
                intersect_angles = (2*np.pi + np.array(intersect_angles))*(np.array(intersect_angles)<0) +\
                                    np.array(intersect_angles)*(np.array(intersect_angles)>0)
            # Use the one with the most negative angle (for counterclockwise arrows)
            intersect_to_use_id = np.argsort(intersect_angles)[0]
            intersect_to_use = circle_intersections[intersect_to_use_id]
            # Calculate angle from center of "self-edge" circle to center of arrowhead
            arrow_center_angle = intersect_angles[intersect_to_use_id]-arrow_arc_angle
            # Get coordinate of center of arrowhead
            arrow_coord = self_edge_coord+r_loop*np.array([np.cos(arrow_center_angle),np.sin(arrow_center_angle)])
            # Get vector pointing in direction of arrow
            arrow_vector = (intersect_to_use-arrow_coord)/np.linalg.norm(intersect_to_use-arrow_coord,ord=2)
            # Get coordinate of arrow "base"
            arrow_base = arrow_coord-(arrow_vector*(arrow_size/2))
            # Make arrow
            plt.arrow(arrow_base[0], arrow_base[1], (intersect_to_use-arrow_base)[0], (intersect_to_use-arrow_base)[1], \
                  length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,color=arrow_color,linewidth=0,zorder=zorder)

            # Use the one with the most positive angle (for clockwise arrows)
            intersect_to_use_id = np.argsort(intersect_angles)[-1]
            intersect_to_use = circle_intersections[intersect_to_use_id]
            # Calculate angle from center of "self-edge" circle to center of arrowhead
            arrow_center_angle = intersect_angles[intersect_to_use_id]+arrow_arc_angle
            # Get coordinate of center of arrowhead
            arrow_coord = self_edge_coord+r_loop*np.array([np.cos(arrow_center_angle),np.sin(arrow_center_angle)])
            # Get vector pointing in direction of arrow
            arrow_vector = (intersect_to_use-arrow_coord)/np.linalg.norm(intersect_to_use-arrow_coord,ord=2)
            # Get coordinate of arrow "base"
            arrow_base = arrow_coord-(arrow_vector*(arrow_size/2))
            # Make arrow
            plt.arrow(arrow_base[0], arrow_base[1], (intersect_to_use-arrow_base)[0], (intersect_to_use-arrow_base)[1], \
                  length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,color=arrow_color,linewidth=0,zorder=zorder,width=0)

            # Add arc for arrow
            arc = mpatches.Arc(xy=np.array([edge['x'],edge['y']]), width=2*r_loop, height=2*r_loop,
                      angle=0, linewidth=weight, fill=False, zorder=zorder,color=arrow_color,\
                    theta1=np.degrees(intersect_angles[np.argsort(intersect_angles)[-1]]+arrow_arc_angle),\
                    theta2=np.degrees(intersect_angles[np.argsort(intersect_angles)[0]]-arrow_arc_angle))
            ax.add_patch(arc)

    # FOR HETERODIMERS
    for i, edge in hetero_edge_df_combined.iterrows():
        if dimers_of_interest is not None:
            if Kij_labels.index(edge['Kij_names']) in dimers_of_interest[univ]:
                arrow_color = 'k'
                zorder = 3
            else:
                arrow_color = non_output_dimer_color
                zorder = 1
        else:
            arrow_color = 'k'
            zorder = 1
        arrow_size = calculate_object_size(edge['weight'],K_edge_scales[0],K_edge_scales[1],K_edge_scales[2],K_edge_scales[3])*2/points_per_coord
        if arrow_size==0:
            continue
        start_node_coord = np.array([edge['x'],edge['y']])
        end_node_coord = np.array([edge['x_end'],edge['y_end']])
        start_node_name = "M_{}".format(edge["Kij_names"].split('_')[1])
        end_node_name = "M_{}".format(edge["Kij_names"].split('_')[2])
        start_node_size = calculate_object_size(node_df_combined.query('species == @start_node_name')['weight']\
                                              .reset_index(drop=True)[0],\
                                              node_scales[0],node_scales[1],node_scales[2],node_scales[3])
        start_node_radius_dataunits = (start_node_size/2)/points_per_coord
        end_node_size = calculate_object_size(node_df_combined.query('species == @end_node_name')['weight']\
                                              .reset_index(drop=True)[0],\
                                              node_scales[0],node_scales[1],node_scales[2],node_scales[3])
        end_node_radius_dataunits = (end_node_size/2)/points_per_coord
        start_end_vector = end_node_coord-start_node_coord
        start_end_vector = start_end_vector/np.linalg.norm(start_end_vector,ord=2)
        start_arrow_tip = start_node_coord+start_end_vector*start_node_radius_dataunits
        start_arrow_base = start_node_coord+start_end_vector*(start_node_radius_dataunits+arrow_size)
        end_arrow_tip = end_node_coord-(start_end_vector*end_node_radius_dataunits)
        end_arrow_base = end_node_coord-(start_end_vector*(end_node_radius_dataunits+arrow_size))
        # Make arrows 
        plt.arrow(start_arrow_base[0], start_arrow_base[1], (start_arrow_tip-start_arrow_base)[0], (start_arrow_tip-start_arrow_base)[1], \
                  length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,color=arrow_color,linewidth=0,zorder=zorder,width=0)
        plt.arrow(end_arrow_base[0], end_arrow_base[1], (end_arrow_tip-end_arrow_base)[0], (end_arrow_tip-end_arrow_base)[1], \
                  length_includes_head=True, head_width=arrow_size*upscale_arrowhead, head_length=arrow_size*upscale_arrowhead, overhang=0,color=arrow_color,linewidth=0,zorder=zorder,width=0)
        ax.plot([start_arrow_base[0], end_arrow_base[0]],[start_arrow_base[1], end_arrow_base[1]], color = arrow_color,\
                lw=calculate_object_size(edge['weight'],K_edge_scales[0],K_edge_scales[1],K_edge_scales[2],K_edge_scales[3]),\
                marker=None,zorder=zorder+1)

    # Make nodes
    # Note that the edges are normally applied half-in, half-out. We reduce the diameter by one edge width because we lose half an edge width on both sides.
    ax.scatter(node_df_combined['x'],node_df_combined['y'],\
        s=[(calculate_object_size(x,\
        node_scales[0],node_scales[1],node_scales[2],node_scales[3])-node_edge_width)**2 for x in node_df_combined['weight']],\
        color=[cmap(i) for i in range(m)]*num_plots,linewidths=node_edge_width,edgecolors='k',zorder=4)
        
    if saveto!='':
        plt.savefig(saveto+'.pdf',pad_inches=0,bbox_inches='tight',transparent=True)
        fig.patch.set_facecolor('white')
        plt.savefig(saveto+'.png',dpi=900,bbox_inches='tight')
    
    return fig, ax


def plot_responses_oneinput_static(plot_fig, plot_ax,m,param_sets,dimer_of_interest,t,plot_all_dimers):
  '''
  plot_fig, plot_ax: Pre-existing figure and axis
  m: int
    Number of monomers
  param_sets: Array of shape (1,num_combos_with_replacement(m,2)+m-1)
    Parameters to simulate
  dimer_of_interest: int or None
    Dimer to use as output, or None (only if plot_all_dimers)
  t: int
    Number of input titration points
  plot_all_dimers: bool
    Whether to plot all dimers, or just dimer of interest
  '''
  # plot_fig, plot_ax = plt.subplots(figsize=figsize)

  ####### Define plotting parameters
  # t=120 # Number of input titration points
  input_lb = -3 # Lower bound for titrating the input monomer species, log10 scale
  input_ub = 3 # Upper bound for titrating the input monomer species, log10 scale

  out_range = [10**input_lb, 10**input_ub]
  # plot_all_dimers = False # Whether to plot all dimers, or just dimer of interest
  min_affinity = 1e-5 # Will not plot dimers below this affinity (assumed not to dimerize)

  ####### Perform simulation
  C0,S_all  = simulate_networks(m, num_inputs=1,param_sets=param_sets, t = t, input_lb = input_lb, input_ub = input_ub)

  ####### Make initial plot
  monomer_cmap = plt.get_cmap('Set2')
  dimer_cmap = plt.get_cmap('tab10')
  # dimer_cmap = plt.get_cmap('tab20')

  x_points = np.logspace(input_lb,input_ub,t,endpoint=True)

  if plot_all_dimers:
      for species in range(number_of_species(m)):
          alpha=0.5
          if species<m:
              color = monomer_cmap(species)
          else:
              color = dimer_cmap(species-m)
              if param_sets[0,species-m]<min_affinity:
                  continue
          if dimer_of_interest is not None:
              if species==m+dimer_of_interest:
                  color='k'
                  alpha=1
          plot_ax.plot(x_points,S_all[:,species,0],marker=None,linestyle='-',color=color,\
                      alpha=alpha,lw=1.5)
  else:
      if dimer_of_interest is not None:
          plot_ax.plot(x_points,S_all[:,m+dimer_of_interest,0],marker=None,linestyle='-',color='k',\
                          alpha=1,lw=1.5)
      else:
          pass


  plot_ax.set_yscale('log')
  plot_ax.set_xscale('log')
  plot_ax.set_xlim([10**input_lb,10**input_ub])
  plot_ax.set_ylim(out_range)
  plot_ax.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
  plot_ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
  _ = plot_ax.set_xlabel('Input M1')
  _ = plot_ax.set_ylabel('Concentration')

  # Create legend, if necessary
  if plot_all_dimers:
      patches={}
      for species in range(m):
          patches[f'M{species+1}'] = mpatches.Patch(facecolor=monomer_cmap(species),linewidth=0,linestyle='',\
                                                    alpha=0.5,label=f'Free M{species+1}',edgecolor=None)
      for species in range(m,number_of_species(m)):
          if param_sets[0,species-m]<min_affinity:
              continue
          color = dimer_cmap(species-m)
          alpha=0.5
          if dimer_of_interest is not None:
              if species==m+dimer_of_interest:
                  color='k'
                  alpha=1
          patches[f'D{species+1}'] = mpatches.Patch(facecolor=color,linewidth=0,linestyle='',\
                                                    alpha=alpha,label=make_nXn_species_names(m = m)[species],\
                                                    edgecolor=None)
      leg = plot_ax.legend(bbox_to_anchor=(1, 0.5), loc='center left',facecolor='none',\
          handles=list(patches.values()),\
          edgecolor='none')
      _ = plot_ax.set_title('Responses')
  else:
    if dimer_of_interest is not None:
        _ = plot_ax.set_title('Response of {}'.format(make_nXn_species_names(m)[m+dimer_of_interest].replace('_','')))
    else:
        _ = plot_ax.set_title('Responses')

  return plot_fig,plot_ax

def network_sandbox_oneinput(m,app_width,app_height,schematic_size,plot_width,plot_height):
  '''
  Function to create plotting interface given m, the number of network monomers
  '''
  num_cols = 4
  num_rows = 3+num_combos_with_replacement(m,2)
  app_width_px = app_width*plt.rcParams['figure.dpi']
  app_height_px = app_height*plt.rcParams['figure.dpi']
  app_grid = ipywidgets.GridspecLayout(num_rows,num_cols,width=f'{app_width_px}px',heigth=f'{app_height_px}px')

  ############## Initialize affinities and expression levels randomly
  rng = np.random.default_rng()
  # Affinities
  K = 10**rng.uniform(-5,7,size=num_combos_with_replacement(m,2))
  # Expression levels
  A = 10**rng.uniform(-3,3,size=m-1)
  param_sets = np.expand_dims(np.hstack((K,A)),axis=0)
  # Output dimer
  dimer_of_interest = rng.integers(num_combos_with_replacement(m,2)) # Index of dimer to use as output (D_1_1 = index 0)

  ############## Plot initial
  plt.ioff()

  t=120 # Number of input titration points
  plot_all_dimers = False
  plot_fig, plot_ax = plt.subplots(figsize=(plot_width,plot_height))
  plot_fig.subplots_adjust(left=0.2,right=0.9,bottom=0.2,top=0.9)
  plot_responses_oneinput_static(plot_fig, plot_ax, m,param_sets,dimer_of_interest,t,plot_all_dimers)


  ####### Plot Schematic
  padding=0.1
  loop_node_ratio = 0.4
  r_node = (schematic_size-(2*padding))/(2+(4*loop_node_ratio))
  r_loop = 0.4*r_node
  schematic_fig, schematic_ax = plt.subplots(figsize=(schematic_size,schematic_size),dpi=72)
  node_scales = [-3,3,3*schematic_size,6*schematic_size]
  K_edge_scales = [-5,7,0.5*schematic_size,2*schematic_size]

  schematic_fig, schematic_ax = make_network_plots_polygon(schematic_fig, schematic_ax,m=m, # Number of monomers
                                        n_input=1, # Number of inputs
                                        param_sets=param_sets, # Parameter sets to draw from
                                        univs_to_plot=np.array([0]), # Indicies of param_sets to plot
                                        dimers_of_interest=np.array([[dimer_of_interest]]), # Index of output dimer
                                        input_node_values=np.array([0]), # Abundances to use for input node(s), log scale
                                        ncols = 1, # Number of columns in figure
                                        r_node = r_node, # Radius of nodes around center
                                        r_loop = r_loop, # Radius of loops around nodes 
                                        node_scales = node_scales, # Scales for node sizes (lower and upper bounds in log scale, min and max sizes)
                                        K_edge_scales = K_edge_scales, # Scales for edge widths (lower and upper bounds in log scale, min and max widths)
                                        input_cmap='Set2', # Colormap for nodes
                                        fontname='Dejavu Sans', # Font name
                                        fontsize=12, # Font size
                                        non_output_dimer_color='gray',
                                        upscale_arrowhead=1.2,
                                        node_edge_width=1,
  )

  ############## Make Widgets
  output_dimer_dropdown_widget = ipywidgets.Dropdown(
      options=make_nXn_species_names(m)[m:],
      value=make_nXn_species_names(m)[m+dimer_of_interest],
      description='Output Dimer',
      disabled=False,
      layout=ipywidgets.Layout(height='auto', width=f'auto'),
  )

  plot_all_dimers_checkbox = ipywidgets.Checkbox(
    value=plot_all_dimers,
    description='Plot responses of all dimers',
    disabled=False,
    indent=False,
    layout=ipywidgets.Layout(height='auto', width=f'auto'),
  )

  t_int_input = ipywidgets.BoundedIntText(
    value=t,
    min=1,
    max=1000,
    step=1,
    description='Number of simulated titration points:',
    disabled=False,
    layout=ipywidgets.Layout(height='auto', width=f'auto'),
  )

  K_widgets = []
  for K_i in range(num_combos_with_replacement(m,2)):
      K_widgets.append(ipywidgets.FloatLogSlider(
        value=param_sets[0,K_i],
        base=10,
        min=-5, # max exponent of base
        max=7, # min exponent of base
        step=0.1, # exponent step
        description=make_Kij_names(m, n_input = 1)[K_i],
        layout=ipywidgets.Layout(height='auto', width=f'auto'),
    ))
  
  A_widgets = []
  for A_i in range(m-1):
      A_widgets.append(ipywidgets.FloatLogSlider(
        value=param_sets[0,num_combos_with_replacement(m,2)+A_i],
        base=10,
        min=-3, # max exponent of base
        max=3, # min exponent of base
        step=0.1, # exponent step
        description=f'M_{A_i+2}',
        layout=ipywidgets.Layout(height='auto', width=f'auto'),
    ))
  
  ############## Define Callback
  def update_dimer_of_interest(change):
    nonlocal plot_fig, plot_ax, schematic_fig, schematic_ax
    nonlocal dimer_of_interest

    dimer_of_interest = make_nXn_species_names(m).index(change['new'])-m

    plot_ax.clear()
    schematic_ax.clear()
    plot_fig,plot_ax = plot_responses_oneinput_static(plot_fig, plot_ax, m,param_sets,dimer_of_interest,t,plot_all_dimers)

    schematic_fig, schematic_ax = make_network_plots_polygon(schematic_fig, schematic_ax,m=m, # Number of monomers
                                        n_input=1, # Number of inputs
                                        param_sets=param_sets, # Parameter sets to draw from
                                        univs_to_plot=np.array([0]), # Indicies of param_sets to plot
                                        dimers_of_interest=np.array([[dimer_of_interest]]), # Index of output dimer
                                        input_node_values=np.array([0]), # Abundances to use for input node(s), log scale
                                        ncols = 1, # Number of columns in figure
                                        r_node = r_node, # Radius of nodes around center
                                        r_loop = r_loop, # Radius of loops around nodes 
                                        node_scales = node_scales, # Scales for node sizes (lower and upper bounds in log scale, min and max sizes)
                                        K_edge_scales = K_edge_scales, # Scales for edge widths (lower and upper bounds in log scale, min and max widths)
                                        input_cmap='Set2', # Colormap for nodes
                                        fontname='Dejavu Sans', # Font name
                                        fontsize=12, # Font size
                                        non_output_dimer_color='gray',
                                        upscale_arrowhead=1.2,
                                        node_edge_width=1,
    )

    plot_fig.canvas.draw()
    plot_fig.canvas.flush_events()

    schematic_fig.canvas.draw()
    schematic_fig.canvas.flush_events()

    return

  def update_plot_all_dimers(change):
    nonlocal plot_fig, plot_ax, schematic_fig, schematic_ax
    nonlocal plot_all_dimers
    plot_all_dimers = change['new']

    plot_ax.clear()
    schematic_ax.clear()
    plot_fig,plot_ax = plot_responses_oneinput_static(plot_fig, plot_ax, m,param_sets,dimer_of_interest,t,plot_all_dimers)

    schematic_fig, schematic_ax = make_network_plots_polygon(schematic_fig, schematic_ax,m=m, # Number of monomers
                                        n_input=1, # Number of inputs
                                        param_sets=param_sets, # Parameter sets to draw from
                                        univs_to_plot=np.array([0]), # Indicies of param_sets to plot
                                        dimers_of_interest=np.array([[dimer_of_interest]]), # Index of output dimer
                                        input_node_values=np.array([0]), # Abundances to use for input node(s), log scale
                                        ncols = 1, # Number of columns in figure
                                        r_node = r_node, # Radius of nodes around center
                                        r_loop = r_loop, # Radius of loops around nodes 
                                        node_scales = node_scales, # Scales for node sizes (lower and upper bounds in log scale, min and max sizes)
                                        K_edge_scales = K_edge_scales, # Scales for edge widths (lower and upper bounds in log scale, min and max widths)
                                        input_cmap='Set2', # Colormap for nodes
                                        fontname='Dejavu Sans', # Font name
                                        fontsize=12, # Font size
                                        non_output_dimer_color='gray',
                                        upscale_arrowhead=1.2,
                                        node_edge_width=1,
    )

    if plot_all_dimers:
      plot_fig.subplots_adjust(right=0.65)  # Increase the right margin
    else:
      plot_fig.subplots_adjust(right=0.9)  # Reset the right margin

    plot_fig.canvas.draw()
    plot_fig.canvas.flush_events()

    schematic_fig.canvas.draw()
    schematic_fig.canvas.flush_events()

    return

  def update_t(change):
    nonlocal plot_fig, plot_ax, schematic_fig, schematic_ax
    nonlocal t

    t = change['new']

    plot_ax.clear()
    schematic_ax.clear()
    plot_fig,plot_ax = plot_responses_oneinput_static(plot_fig, plot_ax, m,param_sets,dimer_of_interest,t,plot_all_dimers)

    schematic_fig, schematic_ax = make_network_plots_polygon(schematic_fig, schematic_ax,m=m, # Number of monomers
                                        n_input=1, # Number of inputs
                                        param_sets=param_sets, # Parameter sets to draw from
                                        univs_to_plot=np.array([0]), # Indicies of param_sets to plot
                                        dimers_of_interest=np.array([[dimer_of_interest]]), # Index of output dimer
                                        input_node_values=np.array([0]), # Abundances to use for input node(s), log scale
                                        ncols = 1, # Number of columns in figure
                                        r_node = r_node, # Radius of nodes around center
                                        r_loop = r_loop, # Radius of loops around nodes 
                                        node_scales = node_scales, # Scales for node sizes (lower and upper bounds in log scale, min and max sizes)
                                        K_edge_scales = K_edge_scales, # Scales for edge widths (lower and upper bounds in log scale, min and max widths)
                                        input_cmap='Set2', # Colormap for nodes
                                        fontname='Dejavu Sans', # Font name
                                        fontsize=12, # Font size
                                        non_output_dimer_color='gray',
                                        upscale_arrowhead=1.2,
                                        node_edge_width=1,
    )

    plot_fig.canvas.draw()
    plot_fig.canvas.flush_events()

    schematic_fig.canvas.draw()
    schematic_fig.canvas.flush_events()

    return
  
  output_dimer_dropdown_widget.observe(update_dimer_of_interest, names='value')
  plot_all_dimers_checkbox.observe(update_plot_all_dimers, names='value')
  t_int_input.observe(update_t, names='value')

  K_update_functions = []
  for K_i in range(num_combos_with_replacement(m,2)):
    def temp_update_func(change,K_i=K_i):
      nonlocal plot_fig, plot_ax, schematic_fig, schematic_ax
      nonlocal param_sets

      param_sets[0,K_i] = change['new']

      plot_ax.clear()
      schematic_ax.clear()
      plot_fig,plot_ax = plot_responses_oneinput_static(plot_fig, plot_ax, m,param_sets,dimer_of_interest,t,plot_all_dimers)

      schematic_fig, schematic_ax = make_network_plots_polygon(schematic_fig, schematic_ax,m=m, # Number of monomers
                                          n_input=1, # Number of inputs
                                          param_sets=param_sets, # Parameter sets to draw from
                                          univs_to_plot=np.array([0]), # Indicies of param_sets to plot
                                          dimers_of_interest=np.array([[dimer_of_interest]]), # Index of output dimer
                                          input_node_values=np.array([0]), # Abundances to use for input node(s), log scale
                                          ncols = 1, # Number of columns in figure
                                          r_node = r_node, # Radius of nodes around center
                                          r_loop = r_loop, # Radius of loops around nodes 
                                          node_scales = node_scales, # Scales for node sizes (lower and upper bounds in log scale, min and max sizes)
                                          K_edge_scales = K_edge_scales, # Scales for edge widths (lower and upper bounds in log scale, min and max widths)
                                          input_cmap='Set2', # Colormap for nodes
                                          fontname='Dejavu Sans', # Font name
                                          fontsize=12, # Font size
                                          non_output_dimer_color='gray',
                                          upscale_arrowhead=1.2,
                                          node_edge_width=1,
      )

      plot_fig.canvas.draw()
      plot_fig.canvas.flush_events()

      schematic_fig.canvas.draw()
      schematic_fig.canvas.flush_events()

      return

    K_update_functions.append(temp_update_func)
    K_widgets[K_i].observe(temp_update_func, names='value')
  
  A_update_functions = []
  for A_i in range(m-1):
    def temp_update_func(change,A_i=A_i):
      nonlocal plot_fig, plot_ax, schematic_fig, schematic_ax
      nonlocal param_sets

      param_sets[0,num_combos_with_replacement(m,2)+A_i] = change['new']

      plot_ax.clear()
      schematic_ax.clear()
      plot_fig,plot_ax = plot_responses_oneinput_static(plot_fig, plot_ax, m,param_sets,dimer_of_interest,t,plot_all_dimers)

      schematic_fig, schematic_ax = make_network_plots_polygon(schematic_fig, schematic_ax,m=m, # Number of monomers
                                          n_input=1, # Number of inputs
                                          param_sets=param_sets, # Parameter sets to draw from
                                          univs_to_plot=np.array([0]), # Indicies of param_sets to plot
                                          dimers_of_interest=np.array([[dimer_of_interest]]), # Index of output dimer
                                          input_node_values=np.array([0]), # Abundances to use for input node(s), log scale
                                          ncols = 1, # Number of columns in figure
                                          r_node = r_node, # Radius of nodes around center
                                          r_loop = r_loop, # Radius of loops around nodes 
                                          node_scales = node_scales, # Scales for node sizes (lower and upper bounds in log scale, min and max sizes)
                                          K_edge_scales = K_edge_scales, # Scales for edge widths (lower and upper bounds in log scale, min and max widths)
                                          input_cmap='Set2', # Colormap for nodes
                                          fontname='Dejavu Sans', # Font name
                                          fontsize=12, # Font size
                                          non_output_dimer_color='gray',
                                          upscale_arrowhead=1.2,
                                          node_edge_width=1,
      )

      plot_fig.canvas.draw()
      plot_fig.canvas.flush_events()

      schematic_fig.canvas.draw()
      schematic_fig.canvas.flush_events()

      return

    A_update_functions.append(temp_update_func)
    A_widgets[A_i].observe(temp_update_func, names='value')

  ######################################

  ############## Define App
  # plot_fig.canvas.capture_scroll = True # If true then scrolling while the mouse is over the canvas will not move the entire notebook
  plot_fig.canvas.header_visible = False # Hide the Figure name at the top of the figure
  plot_fig.canvas.toolbar_visible = True
  plot_fig.canvas.resizable = False

  schematic_fig.canvas.header_visible = False # Hide the Figure name at the top of the figure
  schematic_fig.canvas.toolbar_visible = False
  schematic_fig.canvas.footer_visible = False
  schematic_fig.canvas.resizable = False

  app_grid[:,0] = schematic_fig.canvas
  app_grid[:,1] = plot_fig.canvas
  app_grid[0,2] = plot_all_dimers_checkbox
  app_grid[1,2] = t_int_input
  app_grid[:2,3] = output_dimer_dropdown_widget
  app_grid[2,2] = ipywidgets.HTML(value="<b>Dimerization Affinities</b>",layout=ipywidgets.Layout(height='auto', width=f'auto'))
  app_grid[2,3] = ipywidgets.HTML(value="<b>Accessory Protein Expression Levels</b>",layout=ipywidgets.Layout(height='auto', width=f'auto'))
  for K_i in range(num_combos_with_replacement(m,2)):
      app_grid[3+K_i,2] = K_widgets[K_i]
  for A_i in range(m-1):
      app_grid[3+A_i,3] = A_widgets[A_i]

  # Display the interface
  display(app_grid)

  return
