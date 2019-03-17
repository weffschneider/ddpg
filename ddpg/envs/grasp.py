"""
Free-flyer Gripper Grasping. For model-free RL learning of trajectory to grasp an object.

"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def soft_abs(x, alpha=1.0, d=0):
    z = np.sqrt(alpha**2 + x**2)
    if d == 0:
        return z - alpha
    if d == 1:
        return x/z
    if d == 2:
        return alpha**2 / z**3
    
def vector_cross(x,y):
    """
    Does cross product of two 3x1 np arrays. 
    Normal numpy cross product only takes vectors. 
    """
    assert x.shape[0] == 3
    assert y.shape[0] == 3
    return np.expand_dims(np.cross(x[:,0],y[:,0]), axis=-1)
    
def vector_dot(x,y):
    """
    Does dot product of two 3x1 np arrays. 
    Normal numpy dot product only takes vectors. 
    """
    assert x.shape[0] == 3
    assert y.shape[0] == 3
    return np.dot(x[:,0],y[:,0])
    
def norm_angle(th):
    while th > math.pi:
        th -= math.pi
    while th < -math.pi:
        th += math.pi
    return th
    
logger = logging.getLogger(__name__)

class GraspEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,costf='simple',randomize_params=False,rand_init=True):
        self.s_dim = 12  # state: xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho
        self.a_dim = 3

        self.costf = 'simple'
        self.randomize_params = randomize_params
        self.rand_init = rand_init
        
        #spacecraft params:
        self.ms = 6700.                     # SSL-1300 bus
        self.Js = 1/12 * 6700 * (5^2+5^2)   # cube
        self.rs = 2.5 
        self.Ls = 1.5
        
        #object params:
        self.mo_nom = 1973.                     # Landsat-7 bus
        self.Jo_nom = 1/12 * self.mo_nom * (4^2 + 4^2) # cube
        self.ro = 1.5
        self.Lo = 1.5
        
        #interface params:
        self.kx = 0.5
        self.ky = 0.5
        self.kth = 0.5
        self.dx = 0.2
        self.dy = 0.2
        self.dth = 0.25
        
        self.dt = 0.1

        # Randomization limits
        self.panel1_len_nom = 5.
        self.panel1_angle_nom = math.pi/2.
        
        self.panel2_len_nom = 5.
        self.panel2_angle_nom = 3.*math.pi/2.
                
        # State + action bounds
        # state: xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho
        self.x_upper = 40.
        self.x_lower = -40.
        self.y_upper = self.x_upper
        self.y_lower = self.x_lower
        
        self.v_limit = 5. #vel limit for all directions
        self.angle_limit = 4.
        self.angle_deriv_limit = 2.
        
        self.f_upper = 5.               # Aerojet Rocketdyne MR-111
        self.f_lower = 0.
        self.M_lim = 0.075              # Rockwell Collins RSI 4-75
        
        # -- simple cost terms
        self.simple_x_cost = 0.1
        self.simple_y_cost = 0.1
        self.simple_f1_cost = 0.5
        self.simple_f2_cost = 0.5
        self.simple_m_cost = 0.5
        # --

        # I think this is from CM-gripper to CM-object
        self.offset_distance = self.rs + self.ro + self.Ls + self.Lo

        # define initial state
        # TODO: usually use randomized initial state, but should this be more interesting anyway?
        self.start_state = np.zeros(self.s_dim)
        self.start_state[0] = -5.
        self.start_state[6] = 5.

        # define goal region
        # TODO: this should use the force limit surface from ICRA paper
        # if (outside of gripper range)
        #     nope
        # if (inside gripper range, close enough to gripper)
        #     limit surface
        self.goal_state = np.zeros(self.s_dim)
        self.goal_eps_r = 0.5
        
        # TODO define spaces
        high_ob = [self.x_upper,
                   self.y_upper,
                   self.angle_limit,
                   self.v_limit,
                   self.v_limit,
                   self.angle_deriv_limit,
                   self.x_upper,
                   self.y_upper,
                   self.angle_limit,
                   self.v_limit,
                   self.v_limit,
                   self.angle_deriv_limit]

        low_ob = [self.x_lower,
                  self.y_lower,
                  -self.angle_limit,
                  -self.v_limit,
                  -self.v_limit,
                  -self.angle_deriv_limit,
                  self.x_lower,
                  self.y_lower,
                  -self.angle_limit,
                  -self.v_limit,
                  -self.v_limit,
                  -self.angle_deriv_limit]
        
        high_state = high_ob
        low_state = low_ob
        
        high_state = np.array(high_state)
        low_state = np.array(low_state)
        high_obsv = np.array(high_ob)
        low_obsv = np.array(low_ob)

        high_actions = np.array([self.f_upper,
                                 self.f_upper,
                                 self.M_lim])
        
        low_actions = np.array([-self.f_upper,
                                -self.f_upper,
                                -self.M_lim])

        self.action_space = spaces.Box(low=low_actions, high=high_actions)
        self.state_space = spaces.Box(low=low_state, high=high_state)
        self.observation_space = self.state_space #spaces.Box(low=low_obsv, high=high_obsv)

        self.seed(2017)
        self.viewer = None
        
    def get_ac_sample(self):
        thrust1 = np.random.uniform(-self.f_upper,self.f_upper)*0.1
        thrust2 = np.random.uniform(-self.f_upper,self.f_upper)*0.1
        m = np.random.uniform(-self.M_lim,self.M_lim)*0.1
        return [thrust1,thrust2,m]
    
    def get_ob_sample(self):
        #currently setting random state, not doing trajs
        z = self.observation_space.sample()
        # ********************** TODO ***********************
        # right now this trains a lopsided NN, since the spacecraft always starts in the same
        # direciton relative to the object. just did this for now to keep them from starting
        # on top of each other
        z[0] = np.random.uniform(-10,-2)
        z[1] = np.random.uniform(-10,-2)
        z[2] = np.random.randn()
        z[3] = np.random.uniform(-0.5,0.5)
        z[4] = np.random.uniform(-0.5,0.5)
        z[5] = np.random.uniform(-0.1,0.1)
        
        noise_ampl = 0.2
        z[6] = np.random.uniform(2,10)
        z[7] = np.random.uniform(2,10)
        z[8] = np.random.randn()
        z[9] = np.random.uniform(-0.5,0.5)
        z[10] = np.random.uniform(-0.5,0.5)
        z[11] = np.random.uniform(-0.1,0.1)

        # if (z[0], z[1]) is close to (z[6], z[7])
        #      move the object somewhere else
        
        return z

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def plot_quad_in_map(self):
        # TODO
        pass 
    
    def _in_obst(self, state):
        # TODO
        return False
        
    def _get_obs(self, state):
        return state 
        
    def _gen_state_rew(self,state):
        # TODO
        pass
    
    def _gen_control_rew(self,u):
        # TODO
        pass

    
    def _goal_dist(self, state):
        # TODO: change this
        return soft_abs(state[0]-self.goal_state[0],1.0) 
        + soft_abs(state[1]-self.goal_state[1],1.0)
        + soft_abs(np.sin(state[2])-np.sin(self.goal_state[2]),1.0)
        + soft_abs(np.cos(state[2])-np.cos(self.goal_state[2]),1.0)
        + soft_abs(state[6]-self.goal_state[6],1.0) 
        + soft_abs(state[7]-self.goal_state[7],1.0)
        + soft_abs(np.sin(state[8])-np.sin(self.goal_state[8]),1.0)
        + soft_abs(np.cos(state[8])-np.cos(self.goal_state[8]),1.0)

    def simple_cost(self,s,a):
        xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho = s
        f1, f2, m = a
        
        x_pen = self.simple_x_cost * soft_abs(xo - self.goal_state[6])
        y_pen = self.simple_y_cost * soft_abs(yo - self.goal_state[7])

        f1_pen = self.simple_f1_cost * soft_abs(f1)
        f2_pen = self.simple_f2_cost * soft_abs(f2)
        m_pen = self.simple_m_cost * soft_abs(m)
        
        return x_pen + y_pen + f1_pen + f2_pen + m_pen
    
    def x_dot(self,z,u):
        xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho = z
        fxs, fys, m = u # specific accel (per unit mass)
        
        # velocity terms
        xs_d = vxs
        ys_d = vys
        ths_d = vths
        
        xo_d = vxo
        yo_d = vyo
        tho_d = vtho
        
        # acceleration terms
        vxs_d = fxs
        vys_d = fys
        vths_d = m

        vxo_d  = 0
        vyo_d  = 0
        vtho_d = 0
        
        return [xs_d, ys_d, ths_d, vxs_d, vys_d, vths_d, 
                xo_d, yo_d, tho_d, vxo_d, vyo_d, vtho_d]

        
    def forward_dynamics(self,x,u):
        clipped_thrust = np.clip(u[:2],-self.f_upper,self.f_upper)
        clipped_moment = np.clip(u[2],-self.M_lim,self.M_lim)

        action = np.concatenate((clipped_thrust[:], clipped_moment), axis=None)
    
        if sum(np.isnan(action)) > 0:
            raise ValueError("Passed in nan to step! Action: " + str(action));
        
        old_state = x.copy() #np.array(self.state)

        t = np.arange(0, self.dt, self.dt*0.1)

        integrand = lambda x,t: self.x_dot(x, action)

        x_tp1 = odeint(integrand, old_state, t)
        updated_state = x_tp1[-1,:]
        return updated_state
    
    def step(self, action):
        # state: x,y,z, vx,vy,vz, phi,th,psi, phid, thd, psid,
         # r, rd, beta, gamma, betad, gammad
        # control: f1, f2, M
        
        old_state = self.state.copy()
        
        self.state = self.forward_dynamics(old_state,action)
        
        reward = -1* self.simple_cost(old_state,action)
        
        done = False
        if self._goal_dist(old_state) <= self.goal_eps_r:
            done = True
            reward += 100.

        return self._get_obs(self.state), reward, done, {}
    
    def reset(self):
        
        self.panel1_len = self.panel1_len_nom
        self.panel1_angle = self.panel1_angle_nom
        
        self.panel2_len = self.panel2_len_nom
        self.panel2_angle = self.panel2_angle_nom
            
        self.mo = self.mo_nom
        self.Jo = self.Jo_nom
        
        if self.rand_init:
            self.state = self.get_ob_sample()
        else:
            self.state = self.start_state.copy()
        
        return self._get_obs(self.state)

    def render(self, mode='human', close=False):
        from gym.envs.classic_control import rendering

        # uniform width/height for window for now
        screen_width, screen_height = 600,600 

        scale_x = screen_width/(self.x_upper-self.x_lower)
        scale_y = screen_height/(self.y_upper-self.y_lower)
        scale = 3*scale_x
        if scale_x != scale_y:
          scale = np.min((scale_x,scale_y))
          print('Scales not matching')

        if self.viewer is None:
          # Define viewer 
          self.viewer = rendering.Viewer(screen_width,screen_height)

          # Draw base
          base = rendering.make_circle(scale*self.rs)
          base.set_color(0.,0.,0.)
          self.basetrans = rendering.Transform()
          base.add_attr(self.basetrans)
          self.viewer.add_geom(base)

          # Draw link 1
          xs = np.linspace(0,scale*self.Ls,100)
          ys = np.zeros(xs.shape)
          xys = list(zip(xs,ys))
          l1 = rendering.make_polyline(xys) # draw a straight line
          l1.set_color(1.,0.,0.)
          l1.set_linewidth(3)
          self.l1trans = rendering.Transform() # create transform object for that line
          l1.add_attr(self.l1trans)
          self.viewer.add_geom(l1)

          # Draw link 2
          xs = np.linspace(0,scale*self.Lo,100)
          ys = np.zeros(xs.shape)
          xys = list(zip(xs,ys))
          l2 = rendering.make_polyline(xys)
          l2.set_color(0.,1.,0.)
          l2.set_linewidth(3)
          self.l2trans = rendering.Transform()
          l2.add_attr(self.l2trans)
          self.viewer.add_geom(l2)

          # Draw obj 
          obj = rendering.make_circle(scale*self.ro)
          obj.set_color(.5,.5,.5)
          self.objtrans = rendering.Transform()
          obj.add_attr(self.objtrans)
          self.viewer.add_geom(obj)

          # Draw panel 1
          xs = np.linspace(0,scale*self.panel1_len,100)
          ys = np.zeros(xs.shape)
          xys = list(zip(xs,ys))
          p1 = rendering.make_polyline(xys)
          p1.set_color(0.,0.,1.)
          p1.set_linewidth(4)
          self.p1trans = rendering.Transform()
          p1.add_attr(self.p1trans)
          self.viewer.add_geom(p1)

          # Draw panel 2
          xs = np.linspace(0,scale*self.panel2_len,100)
          ys = np.zeros(xs.shape)
          xys = list(zip(xs,ys))
          p2 = rendering.make_polyline(xys)
          p2.set_color(0.,0.,1.)
          p2.set_linewidth(4)
          self.p2trans = rendering.Transform()
          p2.add_attr(self.p2trans)
          self.viewer.add_geom(p2)

        # Calculate poses for geometries
        xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho = self.state 

        # NOTE: x_conn_s&y_conn_s definitions are NOT same as defined above
        x_conn_s = xs + np.cos(ths) * self.rs
        y_conn_s = ys + np.sin(ths) * self.rs
        x_conn_o = xo - np.cos(tho) * (self.ro + self.Lo) 
        y_conn_o = yo - np.sin(tho) * (self.ro + self.Lo)

        xp1 = xo - np.cos(tho+self.panel1_angle)*(self.ro+self.panel1_len)
        yp1 = yo - np.sin(tho+self.panel1_angle)*(self.ro+self.panel1_len)
        xp2 = xo - np.cos(tho+self.panel2_angle)*(self.ro+self.panel2_len)
        yp2 = yo - np.sin(tho+self.panel2_angle)*(self.ro+self.panel2_len)

        # Update poses for geometries
        self.basetrans.set_translation(
                    screen_width/2+scale*xs,
                    screen_height/2+scale*ys)
        self.basetrans.set_rotation(ths)

        self.l1trans.set_translation(
                    screen_width/2+scale*x_conn_s,
                    screen_height/2+scale*y_conn_s)
        self.l1trans.set_rotation(ths)

        self.l2trans.set_translation(
                    screen_width/2+scale*x_conn_o,
                    screen_height/2+scale*y_conn_o)
        self.l2trans.set_rotation(tho)

        self.objtrans.set_translation(
                    screen_width/2+scale*xo,
                    screen_height/2+scale*yo)
        self.objtrans.set_rotation(tho)
        
        self.p1trans.set_translation(
                    screen_width/2+scale*xp1,
                    screen_height/2+scale*yp1)
        self.p1trans.set_rotation(tho+self.panel1_angle)

        self.p2trans.set_translation(
                    screen_width/2+scale*xp2,
                    screen_height/2+scale*yp2)
        self.p2trans.set_rotation(tho+self.panel2_angle)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
