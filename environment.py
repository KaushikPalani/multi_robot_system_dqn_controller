import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import lsqr 
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import PatchCollection
from math import sin, cos, atan2, pi

class Environment():  
    def __init__(self) -> None:
        self._object_centre = None
        self.goal_robot1 = None
        self.goal_robot2 = None
        self.goal_threshold = None 
        # Attitude of the object
        self.psi : float
        self.theta : float = 0.
        self.phi : float = 0.
        self.robot1_position_bodyframe = np.array([[0.5],[0],[0]])
        self.robot2_position_bodyframe = np.array([[-0.5],[0],[0]])  
        self.robot1_strt_to_goal_dist : float 
        self.robot2_strt_to_goal_dist : float
        self.prev_state1 = []
        self.prev_state2 = []
        self.action_codebook = np.array([[0., 0.15],  # up [velocity_x, velocity_y]]
                                         [0., -0.15], # down 
                                         [-0.15, 0.], # left
                                         [0.15, 0.]]) # right 
        self.observation_space : int = 4
        self.action_space : int = len(self.action_codebook)
        
    @property
    def object_centre(self):
        return self._object_centre
    
    @object_centre.setter
    def object_centre(self, value): 
        '''
        value : TYPE : ndarray[float]
                DESCRIPTION : Object centre coordinates 
        Calculate the position of the robots, and MRS attitude based on the object centre 
        
        Returns
        -------
        None.
        '''
        self._object_centre = value
        r = self.DCM()
        self.Robot1_position = self.object_centre + np.dot(r,self.robot1_position_bodyframe)
        self.Robot2_position = self.object_centre + np.dot(r,self.robot2_position_bodyframe) 
        self.attitude = np.array([[self.psi],[self.theta],[self.phi]])

    def DCM(self):
        '''
        Returns
        -------
        r : TYPE : ndarray[float]
            DESCRIPTION: Direction cosine matrix 
        '''
        r = np.array([[cos(self.theta)*cos(self.psi),                                             
                       cos(self.theta)*sin(self.psi),                                            
                       -sin(self.theta)     ],
                      [sin(self.phi)*sin(self.theta)*cos(self.psi) - cos(self.phi)*sin(self.psi), 
                       sin(self.phi)*sin(self.theta)*sin(self.psi) + cos(self.phi)*cos(self.psi), 
                       sin(self.phi)*cos(self.theta)],
                      [cos(self.phi)*sin(self.theta)*cos(self.psi) + sin(self.phi)*sin(self.psi), 
                       cos(self.phi)*sin(self.theta)*sin(self.psi) - sin(self.phi)*cos(self.psi), 
                       cos(self.phi)*cos(self.theta)]])
        return r
    
    def reset(self, start_position, orientation, goal, goal_threshold):    
        '''
        Resets the environment to the given start position and returns the states 

        Parameters
        ----------
        start_position : TYPE : ndarray[float]
            DESCRIPTION: Starting coordinates for the object centre 
        orientation : TYPE : float 
            DESCRIPTION: psi value of the MRS  
        goal : TYPE : Tuple[ndarray, ndarray]
            DESCRIPTION. Goal coordinates for both the robots 
        threshold : TYPE : float 
            DESCRIPTION: Threshold value for the goal 

        Returns
        -------
        state1 : TYPE : ndarray[float]
            DESCRIPTION: Robot 1 state
        state2 : TYPE : ndarray[float]
            DESCRIPTION: Robot 2 state

        '''
        self.psi = orientation
        self.object_centre = start_position
        self.goal_robot1, self.goal_robot2 = goal
        self.goal_threshold = goal_threshold
        ''' Distance between the starting point and the goal'''
        self.robot1_start_to_goal, _ = self.calcDistAndAngle(np.array([self.Robot1_position[0][0],
                                                                       self.Robot1_position[1][0]]), 
                                                             np.array([self.goal_robot1[0],
                                                                       self.goal_robot2[1]]))
        self.robot2_start_to_goal, _ = self.calcDistAndAngle(np.array([self.Robot2_position[0][0],
                                                                       self.Robot2_position[1][0]]), 
                                                             np.array([self.goal_robot1[0],
                                                                       self.goal_robot2[1]]))
        state1, state2 = self.getStates()
        self.prev_state1, self.prev_state2 = state1.copy(), state2.copy()
        return state1, state2 

    def step(self, action):  
        '''
        Executes the step or the motion based on the given action 

        Parameters
        ----------
        action : TYPE: int
            DESCRIPTION: action to be executed w.r.t the action code book

        Returns
        ----------
        state1 : TYPE : ndarray[float]
            DESCRIPTION: Robot 1 state
        state2 : TYPE : ndarray[float]
            DESCRIPTION: Robot 2 state
        reward : TYPE: float
            DESCRIPTION: Reward obtained based on the current state
        done : TYPE: bool
            DESCRIPTION: Indicates the end of an episode

        '''
        prev_object_centre = self.object_centre.copy()
        prev_Robot1_pose, prev_Robot2_pose = self.Robot1_position.copy(), self.Robot2_position.copy()
        prev_attitude = self.attitude.copy()
        self.leastSqSolution(action, prev_object_centre, prev_attitude)
        state1, state2 = self.getStates()

        robot1_to_goal_prev, angleR1_prev = self.calcDistAndAngle(np.array([prev_Robot1_pose[0][0],
                                                                            prev_Robot1_pose[1][0]]), 
                                                                  np.array([self.goal_robot1[0],
                                                                            self.goal_robot1[1]]))
        robot2_to_goal_prev, angleR2_prev = self.calcDistAndAngle(np.array([prev_Robot2_pose[0][0],
                                                                            prev_Robot2_pose[1][0]]),
                                                                  np.array([self.goal_robot2[0],
                                                                            self.goal_robot2[1]]))
        '''
        Calculate reward
        '''
        p = 0.05
        R1 = -self.normalizeSqDistance(self.robot1_to_goal, self.robot1_start_to_goal**2) - \
                                            self.normalizeSqDistance(self.robot2_to_goal, 
                                                                     self.robot2_start_to_goal**2)
        R2 = -self.normalizeSqAngle(self.limitCalculatedAngleRange(angleR1_prev-self.angle1)) - \
                        self.normalizeSqAngle(self.limitCalculatedAngleRange(angleR2_prev-self.angle2))
        reward = (R1+R2)*p
        
        '''
        Check crash - 1. MRS should be within the walls4
                      2. Check if reached goal considering the threshold
        '''
        done = False
        if not(0<self.Robot1_position[0][0]<10 and 0<self.Robot1_position[1][0]<10) or \
            not(0<self.Robot2_position[0][0]<10 and 0<self.Robot2_position[1][0]<10):
            done = True
        elif (self.goal_robot1[0]-self.goal_threshold) < self.Robot1_position[0][0] < (self.goal_robot1[0]+self.goal_threshold) and \
              (self.goal_robot1[1]-self.goal_threshold) < self.Robot1_position[1][0] < (self.goal_robot1[1]+self.goal_threshold):
            if (self.goal_robot2[0]-self.goal_threshold) < self.Robot2_position[0][0] < (self.goal_robot2[0]+self.goal_threshold) and \
                (self.goal_robot2[1]-self.goal_threshold) < self.Robot2_position[1][0] < (self.goal_robot2[1]+self.goal_threshold):
               reward = 150
               done = True  

        self.prev_state1, self.prev_state2 = state1.copy(), state2.copy()
        return state1, state2, reward, done
    
    def getStates(self):
        '''
        Calculates the states of the robots

        Parameters
        ----------
        None.
        
        Returns
        -------
        state1 : TYPE : ndarray[float]
            DESCRIPTION: Robot 1 state
        state2 : TYPE : ndarray[float]
            DESCRIPTION: Robot 2 state

        '''
        # print( self.Robot1_position)
        self.robot1_to_goal, self.angle1 = self.calcDistAndAngle(np.array([self.Robot1_position[0][0], 
                                                                           self.Robot1_position[1][0]]), 
                                                                 np.array([self.goal_robot1[0][0],
                                                                           self.goal_robot1[1][0]]))
        self.robot2_to_goal, self.angle2 = self.calcDistAndAngle(np.array([self.Robot2_position[0][0], 
                                                                           self.Robot2_position[1][0]]), 
                                                                 np.array([self.goal_robot2[0][0],
                                                                           self.goal_robot2[1][0]]))
        ''' 
        Group angle value based on angle quadrant
        '''
        angle = [0.25 if  (-pi/4 < i <= pi/4) else 0.5 if (pi/4 < i <= 3*(pi/4)) else 0.75 \
                 if ((3*(pi/4) < i <= pi) or (-pi <= i <= -3*(pi/4))) else 1 for i in [self.angle1, self.angle2]]
        state1 = np.array([self.normalizeDistance(self.robot1_to_goal, self.robot1_start_to_goal), angle[0],
                           self.normalizeDistance(self.robot2_to_goal, self.robot2_start_to_goal), angle[1]])
        state2 = np.array([self.normalizeDistance(self.robot2_to_goal, self.robot2_start_to_goal), angle[1],  
                           self.normalizeDistance(self.robot1_to_goal, self.robot1_start_to_goal), angle[0]])
        return state1, state2
    
    def leastSqSolution(self, action, prev_object_centre, prev_attitude) -> None:
        '''
        Kinmeatics based on the ref published paper 

        Parameters
        ----------
        action : TYPE: int
            DESCRIPTION: action to be executed w.r.t the action code book 
        prev_object_centre : TYPE: ndarray[float]
            DESCRIPTION: Previous step's object centre coordinates 
        prev_attitude : TYPE: ndarray[float]
            DESCRIPTION: Previous step's attitude coordinates 

        Returns
        -------
        None.

        '''
        # Velocity input of Robot1 and Robot2     
        B = np.array([[self.action_codebook[action[0]][0]], #r_dot1x
                      [self.action_codebook[action[0]][1]], #r_dot1y
                      [0],                                  #r_dot1z
                      [self.action_codebook[action[1]][0]], #r_dot2x
                      [self.action_codebook[action[1]][1]], #r_dot2y
                      [0]])                                 #r_dot2z
        C=self.DCM() # Direction cosine matrix calculation for the current attitude data
        P1=self.robot1_position_bodyframe # [P1x, P1y, P1z]
        P2=self.robot2_position_bodyframe # [P2x, P2y, P2z]
        A = np.array([[1,0,0, C[0][2]*P1[1]-C[0][1]*P1[2], C[0][0]*P1[2]-C[0][2]*P1[0], C[0][1]*P1[0]-C[0][0]*P1[1]],
                      [0,1,0, C[1][2]*P1[1]-C[1][1]*P1[2], C[1][0]*P1[2]-C[1][2]*P1[0], C[1][1]*P1[0]-C[1][0]*P1[1]],
                      [0,0,1, C[2][2]*P1[1]-C[2][1]*P1[2], C[2][0]*P1[2]-C[2][2]*P1[0], C[2][1]*P1[0]-C[2][0]*P1[1]],
                      [1,0,0, C[0][2]*P2[1]-C[0][1]*P2[2], C[0][0]*P2[2]-C[0][2]*P2[0], C[0][1]*P2[0]-C[0][0]*P2[1]],
                      [0,1,0, C[1][2]*P2[1]-C[1][1]*P2[2], C[1][0]*P2[2]-C[1][2]*P2[0], C[1][1]*P2[0]-C[1][0]*P2[1]],
                      [0,0,1, C[2][2]*P2[1]-C[2][1]*P2[2], C[2][0]*P2[2]-C[2][2]*P2[0], C[2][1]*P2[0]-C[2][0]*P2[1]]], dtype=object)
        # Finding X = [rc_dotx, rc_doty, rc_dotz, wx, wy, wz] using least square approach, X=inv(A)*B  
        X = lsqr(A,B,show=False)
        # Kinematic differential equation to map wx, wy, wz to Euler_angle_rate psi_dot, theta_dot and phi_dot
        D = (1/cos(self.theta))*np.array([[0,               sin(self.phi),                  cos(self.phi)                ],
                                          [0,               cos(self.phi)*cos(self.theta), -sin(self.phi)*cos(self.theta)],
                                          [cos(self.theta), sin(self.phi)*sin(self.theta),  cos(self.phi)*sin(self.theta)]])
        Euler_angle_rate  =  D @ np.array([[X[0][3][0]], 
                                           [X[0][4][0]],  
                                           [X[0][5][0]]])
        time_step = 1
        object_centre_x = prev_object_centre[0][0] + X[0][0] * time_step
        object_centre_y = prev_object_centre[1][0] + X[0][1] * time_step
        object_centre_z = prev_object_centre[2][0] + X[0][2] * time_step
        self.psi   = self.limitCalculatedAngleRange(prev_attitude[0][0] + Euler_angle_rate[0][0] * time_step)
        self.theta = self.limitCalculatedAngleRange(prev_attitude[1][0] + Euler_angle_rate[1][0] * time_step)
        self.phi   = self.limitCalculatedAngleRange(prev_attitude[2][0] + Euler_angle_rate[2][0] * time_step)
        self.object_centre = np.array([[object_centre_x.item()],[object_centre_y.item()],[object_centre_z.item()]])
     
    @staticmethod
    def calcDistAndAngle(pose, target):
        Distance = (np.sum((pose-target)**2))**0.5
        angle = atan2((pose[1]-target[1]),(pose[0]-target[0]))       
        return Distance, angle  
        
    @staticmethod
    def limitCalculatedAngleRange(angle):
        # Limit angle between [-pi, pi]
        angle = (angle + np.pi) % (2*np.pi) - np.pi
        return angle
    
    @staticmethod
    def normalizeDistance(dist, max_dist):
        dist_norm = (dist-0)/(max_dist-0)
        return dist_norm
    
    @staticmethod
    def normalizeSqDistance(dist, max_dist):
        dist_norm = (dist**2-0)/(max_dist**2-0)
        return dist_norm 
    
    @staticmethod
    def normalizeSqAngle(angle):
        angle_norm = (angle**2-0)/(3.14**2-0)
        return angle_norm
    
    def render(self):         
        self.fig = plt.figure(figsize=(28,14))
        self.ax1 = plt.subplot(2,2,(1,3))
        self.ax2 = plt.subplot(2,2,2)
        self.ax3 = plt.subplot(2,2,4)
        self.ax1.axis(np.array([-1, 11, -1, 11]))
        self.ax1.set_xticks(np.arange(-1, 11))
        self.ax1.set_yticks(np.arange(-1, 12))
        self.ax1.grid(color='grey', linestyle=':', linewidth=0.5)  
        
        # Threshold square side length
        top_wall = Rectangle(xy=(-0.5,10), width=11, height=.5, fc='gray')
        left_wall = Rectangle(xy=(-0.5,-.5), width=.5, height=11, fc='gray')
        right_wall = Rectangle(xy=(10,-.5), width=.5, height=11, fc='gray')
        bottom_wall = Rectangle(xy=(-0.5,-.5), width=11, height=.5, fc='gray')
        
        goal_area_robot1 = Rectangle(xy=(self.goal_robot1[0]-self.goal_threshold, 
                                         self.goal_robot1[1]-self.goal_threshold), 
                                     width=self.goal_threshold*2, 
                                     height=self.goal_threshold*2, fc='cyan')
        goal_area_robot2 = Rectangle(xy=(self.goal_robot2[0]-self.goal_threshold, 
                                         self.goal_robot2[1]-self.goal_threshold), 
                                     width=self.goal_threshold*2, 
                                     height=self.goal_threshold*2, fc='cyan')
        
        patch_list = [top_wall, left_wall, right_wall, bottom_wall, goal_area_robot1, goal_area_robot2]
        
        # add wall patches
        c0pat = Circle(
            xy=(self.Robot1_position[0][0], self.Robot1_position[1][0]), 
            radius=.05, 
            ec='black',
            fc='white')
        patch_list.append(c0pat)
        c1pat = Circle(
            xy=(self.Robot2_position[0][0], self.Robot2_position[1][0]), 
            radius=.05, 
            fc='black')
        patch_list.append(c1pat)
        pc = PatchCollection(patch_list, match_original=True) # match_origin prevent PatchCollection mess up original color
        # plot patches
        self.ax1.add_collection(pc)
        # plot rod
        self.ax1.plot(
            [self.Robot1_position[0][0], self.Robot2_position[0][0]],
            [self.Robot1_position[1][0], self.Robot2_position[1][0]],
            color='darkorange')
        plt.pause(0.02) 
    
    