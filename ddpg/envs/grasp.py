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

def exp_barrier(x, alpha=1., d=0):
    if d == 0:
        return alpha**2*(np.exp(x/alpha) - 0.5*(x/alpha)**2 - (x/alpha) - 1)*(x>0)
    elif d == 1:
        return alpha*(np.exp(x/alpha) - (x/alpha) - 1)*(x>0)
    elif d == 2:
        return (np.exp(x) - 1)*(x>0)

def soft_abs(x, alpha=1.0, d=0):
    z = np.sqrt(alpha**2 + x**2)
    if d == 0:
        return z - alpha
    if d == 1:
        return x/z
    if d == 2:
        return alpha**2 / z**3
    
def soft_plus(x, alpha=1.0, d=0):
    f = alpha * np.log(1 + np.exp(x/alpha))
    if d == 0:
        return f
    df = np.exp(x/alpha) / (1 + np.exp(x/alpha))
    if d == 1:
        return df
    if d == 2:
        return 1/alpha * df * ( 1 - df)
    
def barrier_func(x, alpha=1.0, d=0):
    if d == 0:
        return alpha**2*(np.cosh(x / alpha) - 1)
    if d == 1:
        return alpha*(np.sinh(x/alpha))
    if d == 2:
        return np.cosh(x/alpha)

def hat(x):
    """Forms a vector into the cross-product (AKA hat) matrix."""
    x1,x2,x3 = x
    return np.array([[0, -x3, x2],
                     [x3, 0, -x1],
                     [-x2, x1, 0]])
    
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
        
        self.decay_scale_factor_lower = 0.1
        self.decay_scale_factor_upper = 0.5
        
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
        
        self.sr2inv = 1./np.sqrt(2)
        
        # TODO process noise sigma

        # TODO ----- Cost terms
        # -- simple cost terms
        self.simple_x_cost = 1.
        self.simple_y_cost = 1.
        self.simple_f1_cost = 0.1
        self.simple_f2_cost = 0.1
        self.simple_m_cost = 0.5
        
        self.f_barrier_offset = 0.01
        self.m_barrier_offset = 70.
        self.f_barrier_alpha = 1.
        self.m_barrier_alpha = 0.02
        # --
        
        self.weight_control = 1.
        self.weight_goal = 1.
        self.weight_saturation = 1.

        # define initial state
        self.offset_distance = self.rs + self.ro + self.Ls + self.Lo # what is L?
        self.start_state = np.zeros(self.s_dim)
        self.start_state[0] = -5.
        # object starts at offset_distance from spacecraft
        self.start_state[6] = self.start_state[0] + self.offset_distance

        # define goal region
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

        self.action_space = spaces.Box(low=low_actions,high=high_actions, dtype=np.float32)
        self.state_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
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
        z[0] = np.random.uniform(-10,10)
        z[1] = np.random.uniform(-10,10)
        z[2] = np.random.randn()
        z[3] = np.random.uniform(-0.5,0.5)
        z[4] = np.random.uniform(-0.5,0.5)
        z[5] = np.random.uniform(-0.1,0.1)
        
        noise_ampl = 0.2
        z[6] = z[0] + np.cos(z[2]) * self.offset_distance + np.random.randn() * noise_ampl #xo
        z[7] = z[1] + np.sin(z[2]) * self.offset_distance + np.random.randn() * noise_ampl #yo
        z[8] = z[2] + np.random.randn() * noise_ampl #tho
        z[9] = z[3] - np.sin(z[2]) * z[5] * self.offset_distance + np.random.randn() * noise_ampl #vxo
        z[10] = z[4] + np.cos(z[2]) * z[5] * self.offset_distance + np.random.randn() * noise_ampl #vyo
        z[11] =  z[5] + np.random.randn() * noise_ampl #vtho
        
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
        th_pen = self.simple_m_cost * soft_abs(tho - self.goal_state[8])
        
        return x_pen + y_pen + th_pen
    
    def x_dot(self,z,u):
        xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho = z
        fxs, fys, m = u
        
        # velocity terms
        xs_d = vxs
        ys_d = vys
        ths_d = vths
        
        xo_d = vxo
        yo_d = vyo
        tho_d = vtho
        
        # acceleration terms
        
        # computing the link points of each body, for spring force calc
        # also computing the velocities, for damping
        
        # these are in global frame
        # TODO double check the signs on these rotations
        x_conn_s = xs + np.cos(ths) * (self.rs + self.Ls)
        y_conn_s = ys + np.sin(ths) * (self.rs + self.Ls)
        
        vx_conn_s = vxs - np.sin(ths) * (self.rs + self.Ls) * (vths)
        vy_conn_s = vys + np.cos(ths) * (self.rs + self.Ls) * (vths)
        
        x_conn_o = xo + np.cos(tho) * (-1) * (self.ro + self.Lo)
        y_conn_o = yo + np.sin(tho) * (-1) * (self.ro + self.Lo)
        
        vx_conn_o = vxo - np.sin(tho) * (-1) * (self.ro + self.Lo) * (vtho)
        vy_conn_o = vyo + np.cos(tho) * (-1) * (self.ro + self.Lo) * (vtho)
        
        x_diff = x_conn_s - x_conn_o
        y_diff = y_conn_s - y_conn_o
        
        vx_diff = vx_conn_s - vx_conn_o
        vy_diff = vy_conn_s - vy_conn_o
        
        # map displacement and vel diff to local frame of spacecraft for spring calcs
        x_diff_localref = np.cos(ths) * x_diff + np.sin(ths) * y_diff
        y_diff_localref = - np.sin(ths) * x_diff + np.cos(ths) * y_diff

        vx_diff_localref = np.cos(ths) * x_diff + np.sin(ths) * y_diff
        vy_diff_localref = - np.sin(ths) * x_diff + np.cos(ths) * y_diff
        
        # compute forces in local frame, then map back
        # spring force applied to spacecraft in spacecraft frame
        fkx_localref = -self.kx * x_diff_localref
        fky_localref = -self.ky * y_diff_localref
        
        # rotate back to global ref frame
        # TODO move these mappings to a function with forward/backward args
        fkx = np.cos(ths) * fkx_localref - np.sin(ths) * fky_localref
        fky = np.sin(ths) * fkx_localref + np.cos(ths) * fky_localref
        
        mk = - self.kth * norm_angle(ths - tho) #torsional spring
        
        # damping
        fdx_localref = -self.dx * vx_diff_localref
        fdy_localref = -self.dy * vy_diff_localref

        # rotate back to global ref frame
        fdx = np.cos(ths) * fdx_localref - np.sin(ths) * fdy_localref
        fdy = np.sin(ths) * fdx_localref + np.cos(ths) * fdy_localref
        
        md = - self.dth * (vths - vtho) #torsional damping
        
        #spring torque on spacecraft
        ms = (fdy_localref + fky_localref) * (self.Ls + self.rs)
        
        #spring torque on object
        mo = (self.Lo + self.ro) * (np.cos(tho) * (fky + fdy) - np.sin(tho) * (fkx + fdx))
        
        sr2inv = self.sr2inv
        
        # should use standardized rotation function for this transform
        vxs_d = fxs + fkx + fdx
        vys_d = fys + fky + fdy
        vths_d = m + ms + mk + md

        # TODO: add in functionality for post-grasp
        vxo_d  = - fkx - fdx
        vyo_d  = - fky - fdy
        vtho_d = mo - mk - md
        
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
        # control: f, M
        
        old_state = self.state.copy()
        
        self.state = self.forward_dynamics(old_state,action)
        
        # TODO add process noise
        
        costfun = self.simple_cost
        reward = -1* costfun(old_state,action)
        
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
        
        self.decay_scale_factor = self.decay_scale_factor_lower
        
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
          l1 = rendering.make_polyline(xys)
          l1.set_color(1.,0.,0.)
          l1.set_linewidth(3)
          self.l1trans = rendering.Transform()
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
