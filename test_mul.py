"""
Kalman filter multi-dimension system

initial state : state matrix, process covariance matrix (error in estimation)
new state predict : x, p
update new measure and kalman gain
current -> previous

"""

import numpy as np

# State matrix

# Observation
obs_x = [4000, 4260, 4550, 4860, 5110]
obs_xdot = [280, 282, 285, 286, 290] 

# Given
init_x = 4000
init_xdot = 280

# Initial condition
dt = 1
dx = 25
x_dot = 280
a = 2                        
                                                              
# Process errors in process covariance matrix
dp_x = 20
dp_xdot = 5

# Observation error
ob_ex = 25
ob_exdot = 6

var_x = dp_x**2
var_xdot = dp_xdot**2               
covar = dp_x*dp_xdot

# Initial state
X = np.array([[init_x],
              [init_xdot]])

P = np.array([[var_x, covar],
              [covar, var_xdot]])

for i in range(4):

    print(f"\n============= iteration {i+1} =============\n")
    """
    1. New predicted state
    """
    A = np.array([[1, dt],             
                [0,  1]])

    B = np.array([[0.5*dt**2],
                [dt]])

    U = np.array([[a]])

    W = np.zeros((2,1))

    X_k = np.matmul(A,X) + np.matmul(B,U) + W

    # print(f"=== iteration {i+1} ===")
    print("====== New Predicted State:\n",X_k)
    init_x = X_k[0,0]
    init_xdot = X_k[1,0]

    """
    The Covariance Matrix

    P = state covariance matrix (error in estimate)
    Q = process noice matrix 
    R = measurement covariance matrix
    K = Kalman gain

    P = A*P*A_T +Q
    K = P*H_T/(H*P*H_T + R)

    if R -> 0 then K -> 1 (adjust primarily with the measurement update)
    if R -> large then K -> 0 (adjust primarily with the predicted update)
    if P -> 0 then measurement update are mostly ignored

    """

    """
    2. Initial process covariance matrix

    """

    P[0,1] = 0
    P[1,0] = 0
    print("====== Initial Process Covariance:\n", P)

    """
    3. Predicted covariance matrix

    """
    Q = np.zeros((2,2))
    P_predict = np.matmul(np.matmul(A,P),np.transpose(A)) + Q
    print("====== Predicted Covariance Matrix:\n", P_predict)

    # Simplify predicted covariance matrix

    P_predict[0,1] = 0
    P_predict[1,0] = 0
    print("====== Simplify Predicted Covariance Matrix:\n", P_predict)

    """
    4. Kalman Gain

    """

    H = np.array([[1, 0],
                [0, 1]])

    var_Rx = ob_ex**2
    var_Rxdot = ob_exdot**2

    R = np.array([[var_Rx, 0],
                [0, var_Rxdot]])

    np.seterr(invalid='ignore')
    K = np.divide(np.matmul(P_predict,np.transpose(H)),(np.matmul(np.matmul(H,P_predict),np.transpose(H))+R))
    K[0,1] = 0
    K[1,0] = 0
    print("====== Kalman Gain:\n",K)

    """

    5. Update the state matrix using measurement representation
    Y = Obsevation matrix

    """

    C = np.array([[1,0],
                [0,1]])

    Y = np.matmul(C,np.array([[obs_x[i+1]],[obs_xdot[i+1]]])) # Observation matrix
    print("====== Observation matrix :\n", Y)

    """
    6. Calculate current state
    X = X_k + K*[Y-H*X_k]

    """

    X_cal = X_k + np.matmul(K,(Y-np.matmul(H,X_k)))
    print("====== Current State Matrix (Kalman Filter):\n",X_cal) 

    """
    7. Update covariance matrix
    P = (I-K*H)*P_previous

    """

    P_new = np.matmul((np.ones((2,2))-np.matmul(K,H)),P_predict)
    P_new[0,1] = 0
    P_new[1,0] = 0
    print("====== Update Covariance matrix:\n",P_new)

    X = X_cal
    P = P_new





