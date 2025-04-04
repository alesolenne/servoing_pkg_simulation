#!/usr/bin/env python3

import rospy
import numpy as np
import tf
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from std_srvs.srv import SetBool
import matplotlib.pyplot as plt

def get_tf_mat(i, dh):
    """Calculate DH modified transformation matrix from DH parameters for the i joint.
    
    Parameters
    ----------
    dh : [nx4] np.ndarray
         Matrix of DH parameters for n joints
    i : int
        Index of the selected joints
            
    Returns
    -------
    T : [4x4] np.ndarray
        Homogeneus transformation matrix of DH for the i joint    
    """    
    a = dh[i][0]
    d = dh[i][1]
    alpha = dh[i][2]
    theta = dh[i][3]
    q = theta

    T = np.array([[np.cos(q), -np.sin(q), 0, a],
                     [np.sin(q) * np.cos(alpha), np.cos(q) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
                     [np.sin(q) * np.sin(alpha), np.cos(q) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
                     [0, 0, 0, 1]])

    return T

def get_jacobian(joint_angles):
    """Calculate the geometric Jacobian for Panda robot using DH modified convenction and Direct Kinematics.
    
    Parameters
    ----------
    joint_angles : [7x1] np.ndarray
                   Joint state vector
            
    Returns
    -------
    J : [6x7] np.ndarray
        Geometric Jacobian for Panda robot   
    """    
    dh_params = np.array([[0, 0.333, 0, joint_angles[0]],
                 [0, 0, -np.pi / 2, joint_angles[1]],
                 [0, 0.316, np.pi / 2, joint_angles[2]],
                 [0.0825, 0, np.pi / 2, joint_angles[3]],
                 [-0.0825, 0.384, -np.pi / 2, joint_angles[4]],
                 [0, 0, np.pi / 2, joint_angles[5]],
                 [0.088, 0, np.pi / 2, joint_angles[6]],
                 [0, 0.107, 0, 0]], dtype=np.float64)

    T_EE = np.identity(4)
    for i in range(8):
        T_EE = np.matmul(T_EE, get_tf_mat(i, dh_params))

    J = np.zeros((6, 7))
    T = np.identity(4)
    for i in range(7):
        T = np.matmul(T, get_tf_mat(i, dh_params))

        p = T_EE[:3, 3] - T[:3, 3]
        z = T[:3, 2]

        J[:3, i] = np.cross(z, p)
        J[3:, i] = z

    return J[:, :7], T_EE

def rotationMatrixToQuaternion(R):
    """Creates a quaternion from a rotation matrix defining a given orientation.
    
    Parameters
    ----------
    R : [3x3] np.ndarray
        Rotation matrix
            
    Returns
    -------
    q : [4x1] np.ndarray
        quaternion defining the orientation    
    """    

    u_q0 = np.sqrt((1 + R[0,0] + R[1,1] + R[2,2])/4) # the prefix u_ means unsigned
    u_q1 = np.sqrt((1 + R[0,0] - R[1,1] - R[2,2])/4)
    u_q2 = np.sqrt((1 - R[0,0] + R[1,1] - R[2,2])/4)
    u_q3 = np.sqrt((1 - R[0,0] - R[1,1] + R[2,2])/4)
    
    q = np.array([u_q0, u_q1, u_q2, u_q3])
    
    if u_q0 == max(q):
        q0 = u_q0
        q1 = (R[2,1] - R[1,2])/(4*q0)
        q2 = (R[0,2] - R[2,0])/(4*q0)
        q3 = (R[1,0] - R[0,1])/(4*q0)
        
    if u_q1 == max(q):
        q1 = u_q1
        q0 = (R[2,1] - R[1,2])/(4*q1)
        q2 = (R[0,1] + R[1,0])/(4*q1)
        q3 = (R[0,2] + R[2,0])/(4*q1)
    
    if u_q2 == max(q):
        q2 = u_q2
        q0 = (R[0,2] - R[2,0])/(4*q2)
        q1 = (R[0,1] + R[1,0])/(4*q2)
        q3 = (R[1,2] + R[2,1])/(4*q2)    
        
    if u_q3 == max(q):
        q3 = u_q3
        q0 = (R[1,0] - R[0,1])/(4*q3)  
        q1 = (R[0,2] + R[2,0])/(4*q3)
        q2 = (R[1,2] + R[2,1])/(4*q3)  
      
    q = np.array([q0, q1, q2, q3])   
    return q

def quaternionToAxisAngle(p):
    """Compute rotation parameters (axis and angle in radians) from a quaternion p defining a given orientation.
    
    Parameters
    ----------
    p : [4x1] np.ndarray
        quaternion defining a given orientation
            
    Returns
    -------
    axis : [3x1] np.ndarray, when undefined=[0. 0. 0.]
    angle : float      
    """
    if isinstance(p, list) and len(p)==4:
        e0 = np.array(p[0])
        e = np.array(p[1:])  
    elif isinstance(p, np.ndarray) and p.size==4:
        e0 = p[0]
        e = p[1:]
    else:
        raise TypeError("The quaternion \"p\" must be given as [4x1] np.ndarray quaternion or a python list of 4 elements")    
    
    if np.linalg.norm(e) == 0:
        axis = np.array([1,0,0]) # To be checked again
        angle = 0
    elif np.linalg.norm(e) != 0:
        axis = e/np.linalg.norm(e) 
        if e0 == 0:
            angle = np.pi
        else:
            angle = 2*np.arctan(np.linalg.norm(e)/e0) 
          
    return axis, angle

def quaternionToRotationMatrix(p):
    """Computes the rotation matrix given a quaternion p defining an orientation.
    
    Parameters
    ----------
    p : [4x1] np.ndarray
        quaternion defining a given orientation
            
    Returns
    -------
    rotation_matrix : [3x3] np.ndarray
    """   
    if isinstance(p, list) and len(p)==4:
        p = np.array(p) 
    elif isinstance(p, np.ndarray) and p.size==4:
        pass
    else:
        raise TypeError("The quaternion must be given as [4x1] np.ndarray vector or a python list of 4 elements")
    
    e0 = p[0]
    e1 = p[1]
    e2 = p[2]
    e3 = p[3]

    rotation_matrix = np.array([[e0*e0 + e1*e1 - e2*e2 - e3*e3, 2*e1*e2 - 2*e0*e3, 2*e0*e2 + 2*e1*e3],
                                [2*e0*e3 + 2*e1*e2, e0*e0 - e1*e1 + e2*e2 - e3*e3, 2*e2*e3 - 2*e0*e1],
                                [2*e1*e3 - 2*e0*e2, 2*e0*e1 + 2*e2*e3, e0*e0 - e1*e1 - e2*e2 + e3*e3]])        
      
    return rotation_matrix

def skew_matrix(v):
    """Create the skew matrix from a given vector.

    Parameters
    ----------
    v : [3x1] np.ndarray
        vector
            
    Returns
    -------
    skv_matr : [3x3] np.ndarray
               skew matrix of the given vector v
    """
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    skv_matr = skv - skv.T
    return(skv_matr)

def hom_matrix(t,q):
    """Create the homogeneus matrix transformation given translation vector t and quaternion q.

    Parameters
    ----------
    t : [3x1] np.ndarray
        translation vector
    
    q : [4x1] np.ndarray
        quaternion [x y z w] where x y z are the imaginary part and w the real part
            
    Returns
    -------
    T : [4x4] np.ndarray
        homogeneus matrix of t and q.
    """
    T = np.identity(4)
    T[:3,3]=t
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    q_new = [w,x,y,z]   # different order of elements in the representation of the quaternion
    R = quaternionToRotationMatrix(q_new)
    T[0:3,0:3] = R
    return(T)

def feedback(trans_0B, rot_0B, trans_EV, rot_EV, q):
    """PBVS Visual servoing control law.

    Parameters
    ----------
    trans_ij : [3x1] np.ndarray
               translation vector of j with respect to i
    
    rot_ij : [3x3] np.ndarray
             rotation matrix from i to j
            
    Returns
    -------
    dq : [7x1] np.ndarray
         PBVS control law joints velocity

    e: [6x1] np.ndarray
       Position and orientation error vector
    """
    (J, T_0E) = get_jacobian(q)      # Calcolo del Jacobiano geometrico e della posizione di panda link 8 rispetto allo 0
    
    # Calcolo T_0B
    T_0B = hom_matrix(trans_0B, rot_0B)
    T_EV = hom_matrix(trans_EV, rot_EV)
    T_0V = np.linalg.multi_dot([T_0E, T_EV])


    # Calcolo vettore errore
    t_0V = T_0V[:3,3]
    t_0B = T_0B[:3,3]
    R_0V = T_0V[:3,:3]
    R_0B = T_0B[:3,:3]

    R_e = np.matmul(R_0B, np.transpose(R_0V))
    q_e = rotationMatrixToQuaternion(R_e)
    (r,theta) = quaternionToAxisAngle(q_e)
 
    e_t = (t_0B - t_0V)
    e_o = np.sin(theta)*r        # Errore di orientamento
    e = np.array([e_t, e_o], dtype='float').reshape((6,1))
    
    # Calcolo matrice L
    I = np.identity(3)
    Z = np.zeros((3,3))
    L_e = -0.5*(np.matmul(skew_matrix(R_0B[0:3,0]), skew_matrix(R_0V[0:3,0])) + np.matmul(skew_matrix(R_0B[0:3,1]), skew_matrix(R_0V[0:3,1])) + np.matmul(skew_matrix(R_0B[0:3,2]), skew_matrix(R_0V[0:3,2])))
    L_inv = np.linalg.pinv(L_e)

    L = np.block([[I, Z], [Z, L_inv]])

    # Calcolo inversa
    J_trans = np.transpose(J)
    J_1 = np.matmul(J, J_trans)
    J_pinv = np.matmul(J_trans, np.linalg.inv(J_1))
    
    # Calcolo legge di controllo
    K_p = 0.5*np.identity(3)
    K_o = 0.5*np.identity(3)
    K = np.block([[K_p, Z], [Z, K_o]])
    q_dot = np.linalg.multi_dot([J_pinv, L, K, e])

    return q_dot, e 

def plot_reseults(q_plot, dq_plot, e_plot):
    """PBVS Visual servoing plot for joints position, velocity and error vector.
    """
    q_plot = q_plot[0:i+1,:]
    dq_plot = dq_plot[0:i+1,:]
    e_plot = e_plot[0:i+1,:]
   
    x = np.zeros((1,i+1))
    for c in range (1,(i+1),1):
        x[:,c] = x[:,c-1]+1/f
    x = x[0]

    plt.figure(1)
    plt.plot(x, q_plot[:,0], color = 'r', label = 'q1')
    plt.plot(x, q_plot[:,1], color = 'g', label = 'q2')
    plt.plot(x, q_plot[:,2], color = 'b', label = 'q3')
    plt.plot(x, q_plot[:,3], color = 'y', label = 'q4')
    plt.plot(x, q_plot[:,4], color = 'c', label = 'q5')
    plt.plot(x, q_plot[:,5], color = 'm', label = 'q6')
    plt.plot(x, q_plot[:,6], color = 'k', label = 'q7')
    plt.title("Andamento posizione giunti")
    plt.grid()
    plt.xlabel("tempo [s]")
    plt.ylabel("posizione giunto [rad]")
    plt.legend(loc = 'upper right')

    plt.figure(2)
    plt.plot(x, dq_plot[:,0], color = 'r', label = 'dq1')
    plt.plot(x, dq_plot[:,1], color = 'g', label = 'dq2')
    plt.plot(x, dq_plot[:,2], color = 'b', label = 'dq3')
    plt.plot(x, dq_plot[:,3], color = 'y', label = 'dq4')
    plt.plot(x, dq_plot[:,4], color = 'c', label = 'dq5')
    plt.plot(x, dq_plot[:,5], color = 'm', label = 'dq6')
    plt.plot(x, dq_plot[:,6], color = 'k', label = 'dq7')
    plt.title("Andamento velocita giunti")
    plt.grid()
    plt.xlabel("tempo [s]")
    plt.ylabel("velocita giunto [rad/s]")
    plt.legend(loc = 'upper right')

    plt.figure(3)
    plt.plot(x, e_plot[:,0], color = 'r', label = 'e1')
    plt.plot(x, e_plot[:,1], color = 'g', label = 'e2')
    plt.plot(x, e_plot[:,2], color = 'b', label = 'e3')
    plt.plot(x, e_plot[:,3], color = 'y', label = 'e4')
    plt.plot(x, e_plot[:,4], color = 'c', label = 'e5')
    plt.plot(x, e_plot[:,5], color = 'm', label = 'e6')
    plt.title("Andamento errore nel tempo")
    plt.grid()
    plt.xlabel("tempo [s]")
    plt.ylabel("errore posizione [m] e orientamento")
    plt.legend(loc = 'upper right')
    plt.show()

def check_joints_position(q):
    """Physical limits for Panda joint position.
    """
    q_limits = np.array([[-2.8973, 2.8973],
                         [-1.7628, 1.7628],
                         [-2.8973, 2.8973],
                         [-3.0718, -0.0698],
                         [-2.8973, 2.8973],
                         [-0.0175, 3.7525],
                         [-2.8973, 2.8973]], dtype=np.float64)
    

    for i in range(7):
        if (q[i] > q_limits[i][0] and q[i] < q_limits[i][1]):
            continue
        else:  #saturazione
            if q[i] < q_limits[i][0]:
                q[i] = q_limits[i][0]
                rospy.logwarn("Giunto %s al limite di posizione %s", i+1, q_limits[i][0])
            else:
                q[i] = q_limits[i][1]
                rospy.logwarn("Giunto %s al limite di posizione %s" ,i+1 ,q_limits[i][1])
    
    # rospy.loginfo("Posizione di controllo: (%s, %s, %s, %s ,%s, %s, %s)" , q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0])
    return q

def check_joints_velocity(q_dot):
    """Physical limits for Panda joint velocity.
    """
    q_dot_limits = np.array([[-2.1750, 2.1750],
                             [-2.1750, 2.1750],
                             [-2.1750, 2.1750],
                             [-2.1750, 2.1750],
                             [-2.6100, 2.6100],
                             [-2.6100, 2.6100],
                             [-2.6100, 2.6100]], dtype=np.float64)
    
    for i in range(7):
        if (q_dot[i] > q_dot_limits[i][0] and q_dot[i] < q_dot_limits[i][1]):
            continue
        else:  #saturazione
            if q_dot[i] < q_dot_limits[i][0]:
                q_dot[i] = q_dot_limits[i][0]
                rospy.logwarn("Giunto %s al limite di velocita %s", i+1, q_dot_limits[i][0])
            else:
                q_dot[i] = q_dot_limits[i][1]
                rospy.logwarn("Giunto %s al limite di velocita %s", i+1, q_dot_limits[i][1])
    
    # rospy.loginfo("Velocita di controllo: (%s, %s, %s, %s ,%s, %s, %s)" , q_dot[0][0], q_dot[1][0], q_dot[2][0], q_dot[3][0], q_dot[4][0], q_dot[5][0], q_dot[6][0])
    return q_dot



i=0  # Indice temporale
x= 0.0  # Tempo di campionamento

# Vettori per i grafici
q_plot = np.zeros((1000, 7))
dq_plot = np.zeros((1000, 7))
e_plot = np.zeros((1000, 6))

if __name__ == '__main__':
    rospy.init_node('controller')
    listener = tf.TransformListener()
    pub_tf = rospy.Publisher("/position_joint_trajectory_controller/command", JointTrajectory, queue_size=10)
    f = 10.0      # Frequenza di spin del nodo
    rate = rospy.Rate(f)
    initialized = False
    pick = False

    # Chiamata al primo servizio per afferrare il tool
    rospy.loginfo("Inizio della fase di grasp del tool")
    rospy.wait_for_service("grasp_tool_task")
    rospy.loginfo("Server online")

    try:

      client = rospy.ServiceProxy("grasp_tool_task", SetBool)
      request = True
      response = client(request)
      rospy.loginfo(response.message)

    except rospy.ServiceException as e:
      rospy.loginfo("Chiamata al servizio fallita: %s" %e)

    # rospy.sleep(2)
    rospy.loginfo("Inizio della fase di visual sevoing")

    # Inizio della fase di visual servoing
    while not rospy.is_shutdown():

        while not initialized:
            joint_states = rospy.wait_for_message('joint_states',JointState)
            if joint_states.name == ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']:
                q = np.array(joint_states.position).reshape((7,1))     # Stato iniziale letto dal joint states del robot
                # Acquisizione trasformazioni

                try:
                    (t_0B, q_0B) = listener.lookupTransform('/panda_link0', '/object', rospy.Time(0))
                    initialized = True

                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    continue     # Se ci sono errori prova di nuovo a prendere le tf
            else:
                continue # Aspetta che arrivi stato iniziale

        try:
            (t_EV, q_EV) = listener.lookupTransform('/panda_link8', '/tool_extremity', rospy.Time(0))

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue     # Se ci sono errori prova di nuovo a prendere le tf

        # Conversione in vettori
        t_0B = np.array(t_0B)
        q_0B = np.array(q_0B)
        t_EV = np.array(t_EV)
        q_EV = np.array(q_EV)
         
        # Calcolo del nuovo stato con integrazione discreta e legge di controllo PBVS
        (q_dot, e) = feedback(t_0B, q_0B, t_EV, q_EV, q)             # Legge di Controllo PBVS, errore traslazione e orientamento
        q = q + q_dot*1/f                                            # Calcolo lo stato successivo

        # Check se limiti fisici del robot rispettati
        q = check_joints_position(q)
        q_dot = check_joints_velocity(q_dot)
        print(q)
        # Conversione dei dati per controllore di posizione
        q_tolist = [q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0]]
        dq_tolist = [q_dot[0][0], q_dot[1][0], q_dot[2][0], q_dot[3][0], q_dot[4][0], q_dot[5][0], q_dot[6][0]]
            
        # Pubblica sul topic del controllore il comando
        joints_str = JointTrajectory()
        joints_str.header = Header()
        joints_str.header.stamp = rospy.Time.now()
        joints_str.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        point=JointTrajectoryPoint()
        point.positions = q_tolist
        point.velocities = dq_tolist
        x = x + 1/f
        point.time_from_start = rospy.Duration(x)
        joints_str.points.append(point)
        
        pub_tf.publish(joints_str)      # Comando al controllore del robot

        # Criterio di arresto dell'algoritmo
        norm_e_t = np.linalg.norm(e[0:3,:], 2)
        norm_e_o = np.linalg.norm(e[4:7,:], 2)
        rospy.loginfo('La norma dell\'errore di traslazione: %s' %norm_e_t)
        rospy.loginfo('La norma dell\'errore di orientamento: %s' %norm_e_o)
            
        dq_plot[i, :] = np.array([q_dot[0][0], q_dot[1][0], q_dot[2][0], q_dot[3][0], q_dot[4][0], q_dot[5][0], q_dot[6][0]])
        e_plot[i, :] = np.array([e[0][0], e[1][0], e[2][0], e[3][0], e[4][0], e[5][0]])
        q_plot[i, :] = np.array([q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0]])

        if (norm_e_t < 0.001 and norm_e_o < 0.001): # Criterio di raggiungimento regime
            pick = True
            break
        print("----------------------------------------------------------------------------------------------")

        i = i+1 # Avanzamento indice temporale

        rate.sleep()
    
    rospy.loginfo('Visual servoing completato!')
    # rospy.sleep(2)

    # Inizio della fase di pick e throw
    if pick:
        # Chiamata al secondo servizio per il pick e throw
        rospy.loginfo("Inizio della fase di pick and throw dell'oggetto")
        rospy.wait_for_service("throw_tool_task")

        try:

            client = rospy.ServiceProxy("throw_tool_task", SetBool)
            request = True
            response = client(request)
            rospy.loginfo(response.message)

        except rospy.ServiceException as e:
            rospy.loginfo("Chiamata al servizio fallita: %s"%e)

        rospy.loginfo("Task completato!")

# Grafici sul visual servoing
plot_reseults(q_plot, dq_plot, e_plot)