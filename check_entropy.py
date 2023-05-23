import numpy as np
import multiprocessing as mp
import math
from multiprocessing import shared_memory
import sys
import queue
import copy
import tables as pt
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Gillespie_backend/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Gillespie_backend/')
import Gillespie_backend as gil
sys.path.append('/home/hcleroy/PostDoc/aging_condensates/Simulation/Gillespie/Analysis/')
sys.path.append('/home/hugo/PostDoc/aging_condensates/Gillespie/Analysis/')
from ToolBox import *

def extrapolate(X, Y, x0):
    # Find the indices of the two closest points in X such that x1 < x0 < x2
    X = np.asarray(X)
    idx = np.searchsorted(X, x0)

    # Handle the case where x0 is outside the range of X
    if idx == 0:
        return Y[0]
    elif idx == len(X):
        return Y[-1]

    # Compute the distance-weighted average of the function values at the two closest points
    weight = (x0 - X[idx - 1]) / (X[idx] - X[idx - 1])
    y0 = (1 - weight) * Y[idx - 1] + weight * Y[idx]

    return y0

def moving_average(X, Y, window_size):
    """
    Compute a moving average.
    
    Args:
        X: np.array of x values
        Y: np.array of y values
        window_size: size of the window for moving average.
        
    Returns:
        X_av: X values corresponding to the averaged Y values
        Y_av: Averaged Y values
    """
    window = np.ones(int(window_size))/float(window_size)
    Y_av = np.convolve(Y, window, 'valid')

    # For moving average with 'valid' mode, the length of Y_av is reduced 
    # at the beginning and end. We have to remove the corresponding X values.
    cut_size_start = (window_size - 1) // 2
    cut_size_end = (window_size - 1) - cut_size_start
    X_av = X[cut_size_start: -cut_size_end or None]

    return X_av,Y_av



def point_to_curve_distance(X, Y, x, y):
    # Use the extrapolate function to find the y-coordinate of the point on the curve
    y_on_curve = extrapolate(X, Y, x)

    # Compute the distance between the point (x, y) and the point on the curve
    distance = np.abs(y - y_on_curve)

    return distance
def Take_picture(gillespie, step_tot, check_steps, epsilon, X, Y):
    """
    Perform a Gillespie simulation and record the state of the system at specific times.

    Arguments:
    gillespie (Gillespie object) : System on which to perform the simulation.
    step_tot (int) : Total number of evolution steps.
    check_steps (int) : Number of steps between two checks of the entropy value.

    Returns:
    R : a list of 2D arrays, each array has a shape (Nlinker+2,3) and contains the positions of the linkers and 
    the entropy and time values. The entropy and time are appended as [entropy, NaN, NaN] and [time, NaN, NaN], 
    respectively.
    """
    R = np.zeros((step_tot//check_steps,gillespie.get_r_gillespie_size()//3+2,3))
    ell_coordinates = np.zeros((step_tot//check_steps,gillespie.get_r_gillespie_size()//3))
    current_time = 0

    for i in range(step_tot // check_steps):
        moves, times = gillespie.evolve(check_steps)
        current_time += sum(times)
        
        # Estimate the expected Y value at the current time using extrapolation
        expected_Y = extrapolate(X, Y, current_time)
        
        # Check if the entropy is close to the expected value
        S_current = gillespie.get_S()-gillespie.ell_tot*np.log(np.pi*4)
        if abs(S_current - expected_Y) <= epsilon:
            R[i,0] =  [S_current, np.nan, np.nan] 
            R[i,1] = [current_time, np.nan, np.nan]
            R[i,2:] = gillespie.get_r()
            for j,ell in enumerate(gillespie.get_ell_coordinates()):
                ell_coordinates[i,j] = ell
            for j in range(gillespie.get_ell_coordinates().shape[0],gillespie.get_r_gillespie_size()//3):
                ell_coordinates[i,j] = np.nan
        else:
            ell_coordinates[i] = [np.nan for _ in range(ell_coordinates[i].shape[0])]
            R[i,0] = [np.nan,np.nan,np.nan]
            R[i,1] = [np.nan,np.nan,np.nan]
    return R,ell_coordinates
def  Run(inqueue,output,step_tot,check_steps,epsilon,X,Y):
    # simulation_name is a "f_"+float.hex() 
    """
    Each run process fetch a set of parameters called args, and run the associated simulation until the set of arg is empty.
    The simulation consists of evolving the gillespie, every check_steps it checks if the entropy of the system is close enough
    to a given entropy function. If it is the case it adds the position of the linkers associated to this state + the value of the entropy
    and the time associated to this measurement. the position of the linkers is a (Nlinker,3) array to which we add the value of the
    entropy S, and time t as [S, Nan, Nan], and [t,Nan,nan].
    parameters:
    inqueue (multiprocessing.queue) : each entry of q is  a set of parameters associated with a specific gillespie simulation.
    output (multiprocessing.queue) : it just fetch the data that has to be outputed inside this queue
    step_tot (int) : total number of steps in the simulation
    check_step (int) : number of steps between two checking
    epsilon (float): minimum distances (in entropy unit) for the picture to be taken
    X,Y : the average entropy curve of reference.
    """
    for args in iter(inqueue.get,None):
        # create the associated gillespie system
        Nlinker = args[4] 
        ell_tot = args[0]
        kdiff = args[2]
        Energy = args[1]
        seed = args[3]
        dimension = args[5]
        # create the system
        gillespie = gil.Gillespie(ell_tot=ell_tot, rho0=0., BindingEnergy=Energy, kdiff=kdiff,
                            seed=seed, sliding=False, Nlinker=Nlinker, old_gillespie=None, dimension=dimension)
        # pass it as an argument, R returns an array of size (step_tot//check_steps,Nlinker+2,3)
        R,ell_coordinates = Take_picture(gillespie,step_tot,check_steps,epsilon,X,Y)
        output.put(('create_array',('/','ell_coordinates_'+hex(seed),ell_coordinates)))
        output.put(('create_array',('/',"R_"+hex(seed),R)))
def handle_output(output,filename,header):
    """
    This function handles the output queue from the Simulation function.
    It uses the PyTables (tables) library to create and write to an HDF5 file.

    Parameters:
    output (multiprocessing.Queue): The queue from which to fetch output data.

    The function retrieves tuples from the output queue, each of which 
    specifies a method to call on the HDF5 file (either 'createGroup' 
    or 'createArray') and the arguments for that method. 

    The function continues to retrieve and process data from the output 
    queue until it encounters a None value, signaling that all simulations 
    are complete. At this point, the function closes the HDF5 file and terminates.
    """
    hdf = pt.open_file(filename, mode='w') # open a hdf5 file
    while True: # run until we get a False
        args = output.get() # access the last element (if there is no element, it keeps waiting for one)
        if args: # if it has an element access it
            method, args = args # the elements should be tuple, the first element is a method second is the argument.
            getattr(hdf, method)(*args) # execute the method of hdf with the given args
        else: # once it receive a None
            break # it break and close
    hdf.close()
def make_header(args,sim_arg):
    header = 'this file contains pictures of the system. Pictures are only taken if  the entropy of the picture '
    header +='is close enough to the average entropy curve (that has been computed by averaging 50 to 100 systems) '
    header += 'the file is composed of arrays, each array name can be written : h_X...X where X...X represent an hexadecimal '
    header+= 'name for an integer that corresponds to the seed of the simulation. Each array is made of the position of N '
    header+= 'linkers. Additionnally, the two first entry of the array are [S,NaN,Nan] and [t,NaN,NaN] that are respectively  '
    header+= 'the value of the entropy and time of the given picture.\n'
    header += 'Parameters of the simulation : '
    header +='Nlinker = '+str(args[4])+'\n'
    header +='ell_tot = '+str(args[0])+'\n'
    header += 'kdiff = '+str(args[2])+'\n'
    header += 'Energy =  '+str(args[1])+'\n'
    header += 'seed = '+str(args[3])+'\n'
    header += 'dimension = '+str(args[5])+'\n'
    header+='step_tot = '+str(sim_arg[0])+'\n'
    header+='check_steps = '+str(sim_arg[1])+'\n'

def  Parallel_pictures(args,step_tot,check_steps,filename,epsilon,X,Y):
    """
    take pictures of a system, making sure that it is close enough to the average entropy
    parameters:
    args (iterable) : arguments of the gillespie system in the following order : ell_tot, energy, kdiff,seed,Nlinker,dimension
    step_tot (int) : total number of timesteps use.
    check_step : number of steps between two pictures
    filename : name of the file to save the pictures.
    epsilon (float) : minimum distance for the picture to be taken
    X,Y (arrays) : average entropy of reference.
    return:
    nothing, but create a file with the given name
    """
    num_process = mp.cpu_count()
    output = mp.Queue() # shared queue between process for the output
    inqueue = mp.Queue() # shared queue between process for the inputs used to have less process that simulations
    jobs = [] # list of the jobs for  the simulation
    header = make_header(args,[step_tot,check_steps])
    proc = mp.Process(target=handle_output, args=(output,filename,header)) # start the process handle_output, that will only end at the very end
    proc.start() # start it
    for i in range(num_process):
        p = mp.Process(target=Run, args=(inqueue, output,step_tot,check_steps,epsilon,X,Y)) # start all the 12 processes that do nothing until we add somthing to the queue
        jobs.append(p)
        p.start()
    for arg in args:
        inqueue.put(arg)  # put all the list of tuple argument inside the input queue.
    for i in range(num_process): # add a false at the very end of the queue of argument
        inqueue.put(None) # we add one false per process we started... We need to terminate each of them
    for p in jobs: # wait for the end of all processes
        p.join()
    output.put(False) # now send the signal for ending the output.
    proc.join() # wait for the end of the last process