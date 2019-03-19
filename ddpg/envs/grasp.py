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
        self.panel1_angle_nom = math.pi/3.
        
        self.panel2_len_nom = 5.
        self.panel2_angle_nom = -math.pi/3.
                
        # State + action bounds
        # state: xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho
        self.x_upper = 10.
        self.x_lower = -10.
        self.y_upper = self.x_upper
        self.y_lower = self.x_lower
        
        self.v_limit = 0.5 #vel limit for all directions
        self.angle_limit = math.pi/4.
        self.angle_deriv_limit = math.pi/16.
        
        self.f_upper = 5.               # Aerojet Rocketdyne MR-111
        self.f_lower = 0.
        self.M_lim = 0.075              # Rockwell Collins RSI 4-75
        
        # -- simple cost terms
        self.simple_dist_cost = 0.1
        self.simple_angle_cost = 0.1
        self.simple_vel_cost = 0.2
        self.simple_f1_cost = 0.5
        self.simple_f2_cost = 0.5
        self.simple_m_cost = 0.7
        # --

        # I think this is from CM-gripper to CM-object
        self.offset_distance = self.rs + self.ro + self.Ls + self.Lo

        # define default initial state (note: not used if rand_init=True)
        self.start_state = np.zeros(self.s_dim)
        self.start_state[0] = -5.
        self.start_state[6] = 5.

        # define goal region
        # TODO: this should use the force limit surface from ICRA paper
        # if (outside of gripper range)
        #     nope
        # if (inside gripper range, close enough to gripper)
        #     limit surface
        self.goal_eps_norm = 0.5
        self.goal_eps_tan = 1.0
        self.goal_eps_ang = math.pi/4.
        self.goal_eps_vel = 0.5
        
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
        # currently setting random state, not doing trajs
        z = self.state_space.sample()

        # train always in the same-ish direction
        z[0] = np.random.uniform(-5, -2)
        z[1] = np.random.uniform(-5, -2)
        z[2] = np.random.uniform(-math.pi, math.pi)
        # start at zero velocity
        z[3] = 0 #np.random.uniform(-0.1,0.1)
        z[4] = 0 #np.random.uniform(-0.1,0.1)
        z[5] = 0

        z[6] = np.random.uniform(2,5)
        z[7] = np.random.uniform(2,5)
        z[8] = 0 # doesn't matter
        z[9] = np.random.uniform(-0.1,0.1)
        z[10] = np.random.uniform(-0.1,0.1)
        z[11] = 0 # doesn't matter

        # # keep moving object until they're not on top of each other
        # while np.sqrt((z[6]-z[0])**2 + (z[7]-z[1])**2) < 1.2*(self.ro+self.rs):
        #     z[6] = np.random.uniform(self.x_lower, self.x_upper)
        #     z[7] = np.random.uniform(self.y_lower, self.y_upper)
        
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
        xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho = state
        s2o = np.array([xo-xs,yo-ys]);
        xs_hat = np.array([[np.cos(ths)],[np.sin(ths)]])
        ys_hat = np.array([[-np.sin(ths)],[np.cos(ths)]])
        norm_dist_to_object = soft_abs(np.dot(s2o,xs_hat) - (self.ro+self.rs), 1.0)
        tan_dist_to_object = soft_abs(np.dot(s2o,ys_hat), 1.0)
        angle_to_gripper = soft_abs(ths - np.arctan2(yo-ys,xo-xs), 1.0)

        # TODO: add distance for force limit surface (e.g. velocity limits)
        return (norm_dist_to_object, tan_dist_to_object, angle_to_gripper)

    def simple_cost(self,s,a):
        xs, ys, ths, vxs, vys, vths, xo, yo, tho, vxo, vyo, vtho = s
        f1, f2, m = a
        
        dist_to_object = np.sqrt((xo-xs)**2 + (yo-ys)**2)
        dist_pen = self.simple_dist_cost * soft_abs(dist_to_object)

        angle_to_gripper = soft_abs(ths - np.arctan2(yo-ys,xo-xs), 1.0)
        ang_pen = self.simple_angle_cost * angle_to_gripper

        f1_pen = self.simple_f1_cost * soft_abs(f1)
        f2_pen = self.simple_f2_cost * soft_abs(f2)
        m_pen = self.simple_m_cost * soft_abs(m)

        # TODO: add cost for being at the goal position but going too fast...
        
        return dist_pen + ang_pen + f1_pen + f2_pen + m_pen
    
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
        norm_dist, tan_dist, angle = self._goal_dist(old_state)
        if (soft_abs(norm_dist) <= self.goal_eps_norm and
            soft_abs(tan_dist)  <= self.goal_eps_tan):# and
            #angle     <= self.goal_eps_ang):
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

        scale_x = screen_width/(4*(self.x_upper-self.x_lower))
        scale_y = screen_height/(4*(self.y_upper-self.y_lower))
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

        # velocity direction
        ths_vel = np.arctan2(vys, vxs)
        tho_vel = np.arctan2(vyo, vxo)

        # NOTE: x_conn_s&y_conn_s definitions are NOT same as defined above
        x_conn_s = xs + np.cos(ths_vel) * self.rs
        y_conn_s = ys + np.sin(ths_vel) * self.rs
        x_conn_o = xo + np.cos(tho_vel) * self.ro 
        y_conn_o = yo + np.sin(tho_vel) * self.ro

        xp1 = xs - np.cos(ths+self.panel1_angle)*(self.rs+self.panel1_len)
        yp1 = ys - np.sin(ths+self.panel1_angle)*(self.rs+self.panel1_len)
        xp2 = xs - np.cos(ths+self.panel2_angle)*(self.rs+self.panel2_len)
        yp2 = ys - np.sin(ths+self.panel2_angle)*(self.rs+self.panel2_len)

        # Update poses for geometries
        self.basetrans.set_translation(
                    screen_width/2+scale*xs,
                    screen_height/2+scale*ys)
        self.basetrans.set_rotation(ths)

        self.l1trans.set_translation(
                    screen_width/2+scale*x_conn_s,
                    screen_height/2+scale*y_conn_s)
        # pointing along spacecraft velocity
        self.l1trans.set_rotation(ths_vel)

        self.l2trans.set_translation(
                    screen_width/2+scale*x_conn_o,
                    screen_height/2+scale*y_conn_o)
        self.l2trans.set_rotation(tho_vel)

        self.objtrans.set_translation(
                    screen_width/2+scale*xo,
                    screen_height/2+scale*yo)
        self.objtrans.set_rotation(tho)
        
        self.p1trans.set_translation(
                    screen_width/2+scale*xp1,
                    screen_height/2+scale*yp1)
        self.p1trans.set_rotation(ths+self.panel1_angle)

        self.p2trans.set_translation(
                    screen_width/2+scale*xp2,
                    screen_height/2+scale*yp2)
        self.p2trans.set_rotation(ths+self.panel2_angle)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
