## Curvature superposition based on CC kinematics
## robot base is along z axis with initial bent around y axis

import numpy as np
try:
    import transforms3d as tr
except ImportError:
    from scipy.spatial.transform import Rotation as _Rotation

    class _QuaternionFallback:
        @staticmethod
        def mat2quat(R):
            q_xyzw = _Rotation.from_matrix(R).as_quat()
            x, y, z, w = q_xyzw
            return np.array([w, x, y, z])

    class _Transforms3DFallback:
        quaternions = _QuaternionFallback()

    tr = _Transforms3DFallback()
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def superPosKin(CTR_par, inputs, sim_par):

    ## robot dimentions
    n_t = CTR_par['n_t'] # 3 # number of tubes
    l_t = CTR_par['l_t'] # [ [ 321e-3 , 120e-3 ] , [ 229e-3 , 86e-3 ] , [ 0e-3 , 173e-3] ] # [m] tube lengthes [straight, curve] innermost to outermost 
    E = CTR_par['E'] # [ 75e9 , 75e9 , 75e9 ] # [Pa] elasticity modulus
    kappa_0 = CTR_par['kappa_0'] # [ 86 , 65 , 14 ] # [1/m] curvature around x0 axis
    r = CTR_par['r'] # [ [ 0.49e-3 , 1.1e-3 ] , [ 1.37e-3 , 1.76e-3 ] , [ 1.83e-3 , 2.39e-3 ] ] # [m] tube [inner , outer ] radius: innermost to outermost

    ## robot input parameters
    ul = inputs['ul'] # [ 60e-3 , 40e-3 , 20e-3 ] # [m] input translation: length out of the insertion point at frame origin
    uphi = inputs['uphi'] # [ 90*np.pi/180 , 60*np.pi/180 , 30*np.pi/180 ] # [deg] input rotation: tube base rotation 

    ## simulation parameters
    n_p = sim_par['n_p'] # 20 # number of segments per tube for plotting
    isPlot = sim_par['isPlot'] # True # plot the results or not

    ## preprocessing
    EI = [ 0 for _ in range(n_t) ] # bending stiffness per length: innermost to outermost tube
    kappa_xy0 = [ [ 0 , 0 ] for _ in range(n_t) ] # [1/m] [kappa_x,kappa_y] for innermost to outermost tube
    # ul_key = [ [ 0 , 0 ] for _ in range(n_t) ] # ul with tube number as key [i_t , ul]
    for i_t in range(n_t):
        EI[i_t] = E[i_t] * np.pi * ( np.power( r[i_t][1] , 4 ) - np.power( r[i_t][0] , 4 ) ) / 4
        kappa_xy0[i_t][0] = kappa_0[i_t] * np.cos( uphi[i_t] ) # kappa_x
        kappa_xy0[i_t][1] = kappa_0[i_t] * np.sin( uphi[i_t] ) # kappa_y
        # ul_key[i_t][0] = i_t
        # ul_key[i_t][1] = ul

    ## function to sort lengthes
    # def keySort(list):
    #     return list[1]

    ## curvature superposition
    # ul_sorted = sorted( ul , key=keySort ) # sorted lengthes
    ul_sorted = sorted( ul ) # sorted lengthes
    dl_sorted = [ 0 for _ in range(n_t) ] # ssection lengths
    Kappa_xy = [ [ 0 , 0 ] for _ in range(n_t) ] # mean [ kappa_x, kappa_y ] for each robot segment
    Kappa_w = [ 0 for _ in range(n_t) ] # mean kappa_xy0 weight summation for each robot segment
    Kappa = [ 0 for _ in range(n_t) ] # mean overall bending curvature for section i_l
    phi = [ 0 for _ in range(n_t) ] # mean bending plane polar angle for each robot segment
    for i_l in range(n_t): # itterate over the sorted tubes' length
        if i_l == 0:
            dl_sorted[i_l] = ul_sorted[i_l]
        else:
            dl_sorted[i_l] = ul_sorted[i_l] - ul_sorted[i_l-1]
        for i_t in range(n_t): # check the length of all tubes
            if ul[i_t] >= ul_sorted[i_l]:
                Kappa_xy[i_l][0] = Kappa_xy[i_l][0] + EI[i_t] * kappa_xy0[i_t][0]
                Kappa_xy[i_l][1] = Kappa_xy[i_l][1] + EI[i_t] * kappa_xy0[i_t][1]
                Kappa_w[i_l] = Kappa_w[i_l] + EI[i_t] # mean kappa_0 weight summation for section i_l
        Kappa_xy[i_l][0] = Kappa_xy[i_l][0] / Kappa_w[i_l] # mean kappa_x for section i_l
        Kappa_xy[i_l][1] = Kappa_xy[i_l][1] / Kappa_w[i_l] # mean kappa_y for section i_l
        Kappa[i_l] = np.sqrt( np.power( Kappa_xy[i_l][0] , 2 ) + np.power( Kappa_xy[i_l][1] , 2 ) )
        # Robust bending-plane angle:
        # a zero-curvature section (e.g. the straight inner tube alone) has no
        # defined bending plane, so use zero. This does not alter its straight
        # backbone position. For curved sections use atan2 to avoid division by zero.
        if np.isclose(Kappa[i_l], 0.0):
            phi[i_l] = 0.0
        else:
            phi[i_l] = np.arctan2(Kappa_xy[i_l][0], Kappa_xy[i_l][1])

    ## shape generation function
    rhoQ = [ [ [ 0 for i_p in range(n_p+1) ] for i_t in range(7) ] for i_l in range(0,n_t) ] # positions/orientation [x, y, z, q0, qv1, qv2, qv3] along each segment of the robot backbone
    s = [ [ i_p*dl_sorted[i_l]/n_p for i_p in range(n_p+1) ] for i_l in range(0,n_t) ] # unit length location along backbone
    T0 = np.identity( 4 ) # previous section transformation
    for i_l in range(n_t):
        for i_p in range(n_p+1):
            theta = s[i_l][i_p] * Kappa[i_l]
            if Kappa[i_l] == 0:
                rho = [ 0, 0, s[i_l][i_p] ]
            else:
                rho = [ -(np.cos(phi[i_l])*(np.cos(Kappa[i_l]*s[i_l][i_p]) - 1))/Kappa[i_l] , 
                        -(np.sin(phi[i_l])*(np.cos(Kappa[i_l]*s[i_l][i_p]) - 1))/Kappa[i_l] ,
                        np.sin(Kappa[i_l]*s[i_l][i_p])/Kappa[i_l] ]
            R = np.array( # robot base is along z axis with initial bent around y axis
                [ [ np.cos(phi[i_l]), -np.sin(phi[i_l]), 0 ] ,
                [ np.sin(phi[i_l]), np.cos(phi[i_l]), 0 ] ,
                [ 0, 0, 1 ] ] ) @ np.array(
                [ [ np.cos(theta), 0 , np.sin(theta) ] ,
                [ 0, 1, 0 ] ,
                [ -np.sin(theta), 0 , np.cos(theta) ] ] ) # CC relation
            Tl = np.array(
                [ [ R[0][0] , R[0][1] , R[0][2] , rho[0] ] ,
                [ R[1][0] , R[1][1] , R[1][2] , rho[1] ] ,
                [ R[2][0] , R[2][1] , R[2][2] , rho[2] ] ,
                [ 0 , 0 , 0 , 1 ] ] ) # rotation matrix part
            T = T0 @ Tl # matrix multiplication
            rhoQ[i_l][0][i_p] = T[0][3] # x
            rhoQ[i_l][1][i_p] = T[1][3] # y
            rhoQ[i_l][2][i_p] = T[2][3] # z
            q = tr.quaternions.mat2quat( R )
            rhoQ[i_l][3][i_p] = q[0]
            rhoQ[i_l][4][i_p] = q[1]
            rhoQ[i_l][5][i_p] = q[2]
            rhoQ[i_l][6][i_p] = q[3]
        T0 = T

    ## postprocessing
    R_tip = T0[np.ix_([0,1,2],[0,1,2])] # tip rotation matrix
    t_tip = R_tip @ np.array( [ [0] , [0] , [1] ] ) # tip tangent
    rhoQ_tip = [ rhoQ[n_t-1][0][n_p] , rhoQ[n_t-1][1][n_p] , rhoQ[n_t-1][2][n_p] , rhoQ[n_t-1][3][n_p] , rhoQ[n_t-1][4][n_p] , rhoQ[n_t-1][5][n_p] , rhoQ[n_t-1][6][n_p]] # tip positiona dn orientation
    # print( rhoQ_tip )

    ## plotting
    if isPlot:
        clr = [ '-b' , '-r' , '-g' , '-c' , '-y' ] # tube colors
        ax = plt.axes(projection='3d')
        for i_l in range(n_t):
            ax.plot3D( rhoQ[i_l][0] , rhoQ[i_l][1] , rhoQ[i_l][2] , clr[i_l] )
        plt.show()
        
    ## terminate
    # input('Save the plot and hit enter...')

    ## return
    return rhoQ_tip, R_tip, s, rhoQ, Kappa, Kappa_xy, phi, ul_sorted, t_tip

