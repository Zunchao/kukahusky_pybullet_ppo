import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import mmKukaHusky
import random
from pkg_resources import parse_version
import pyglet
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

pyglet.clock.set_fps_limit(10000)

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960



class MMKukaHuskyGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=parentdir,
                 actionRepeat=1,
                 isEnableSelfCollision=True,
                 renders=False,
                 isDiscrete=False,
                 maxSteps=1000,
                 action_dim = 9,
                 rewardtype='rdense',
                 randomInitial=False):
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self.isEnableRandInit = randomInitial
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 4 #1.3
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._rewardtype = rewardtype
        self._action_dim = action_dim
        self.dis = 100
        self._dis_vor = 100 # changes between the current and former distance
        self.count = 0

        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        # timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
        self._seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())

        observation_high = np.array([largeValObservation] * observationDim)

        da = 0.01

        if (self._isDiscrete):
            self.action_space = spaces.Discrete(9)
        else:
            #self._action_bound = math.pi/2
            # action_high = np.array([self._action_bound] * action_dim)
            #if (self._action_dim == 5):
                # husky twist limits is from https://github.com/husky/husky/blob/kinetic-devel/husky_control/config/control.yaml
                #action_high = np.array([0.01, 0.01, 0.01, 1, 2])
            #elif (self._action_dim == 9):
            action_high = np.ones(self._action_dim)*da
            self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float32)
        self.viewer = None
        # help(mmKukaHusky)

    def _reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)

        p.setGravity(0, 0, -9.8)
        p.loadURDF(os.path.join(self._urdfRoot, "kukahusky_pybullet_ppo/data/plane.urdf"), [0, 0, 0])

        d_space_scale = len(str(abs(self.count)))*0.5
        print('scale : ', self.count, d_space_scale)
        # d_space_scale = 1
        xpos = random.uniform(-d_space_scale, d_space_scale) + 0.20
        ypos = random.uniform(-d_space_scale, d_space_scale) + 0.35
        zpos = random.uniform(0.5, 1.4)
        self.goal = [xpos, ypos, zpos]
        self.goalUid = p.loadURDF(os.path.join(self._urdfRoot, "kukahusky_pybullet_ppo/data/spheregoal.urdf"), xpos, ypos, zpos)

        self._mmkukahusky = mmKukaHusky.MMKukaHusky(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, randomInitial=self.isEnableRandInit)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        #self._observation.extend(list(self.goal))
        return self._observation

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        observation = self._mmkukahusky.getObservation()
        observation.extend(self.goal)
        #for i in [3,4,5]:
            #observation[i] = self.goal[i-3]
        # self._observation = [self.dis]
        self._observation = observation
        # print('obs', self._observation)
        return self._observation

    def _step(self, action):
        for i in range(self._actionRepeat):
            self._mmkukahusky.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1
        if self._renders:
            time.sleep(self._timeStep)
        self._observation = self.getExtendedObservation()
        self._actions = action

        done = self._termination()
        reward = self._reward()

        return self._observation, reward, done, {}

    def _render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._mmkukahusky.huskyUid)
        ''''''
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        # renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        # state = p.getLinkState(self._mmkukahusky.kukaUid, self._mmkukahusky.kukaEndEffectorIndex)
        # actualEndEffectorPos = state[0]
        self._observation = self.getExtendedObservation()
        # print('obs1', actualEndEffectorPos, self._observation)

        if (self.terminated or self._envStepCounter > self._maxSteps):
            # self._observation = self.getExtendedObservation()
            return True

        disvec = [x-y for x, y in zip(self._observation[0:3], self.goal)]
        dis = np.linalg.norm(disvec)

        if dis < 0.1:  # (actualEndEffectorPos[2] <= -0.43):
            self.terminated = 1
            self.count += 1
            # self._observation = self.getExtendedObservation()
            # print('terminate:', self._observation, dis,self.goal)
            # [-0.6356161906186968, 0.4866813952531867, 1.1774765260184725, -0.07900755219115511, 0.013299602972714528, -0.4413131443405152, -0.6369316683047506, 0.4457577316748338, 1.0863575155494019]
            # 0.09989569956249031 [-0.6369316683047506, 0.4457577316748338, 1.0863575155494019]
            return True
        return False

    def _reward(self):
        # state = p.getLinkState(self._mmkukahusky.kukaUid, self._mmkukahusky.kukaEndEffectorIndex)
        # actualEndEffectorPos = state[0]
        disvec = [x-y for x, y in zip(self._observation[0:3], self.goal)]
        self.dis = np.linalg.norm(disvec)
        delta_dis = self.dis - self._dis_vor
        self._dis_vor = self.dis
        if self._rewardtype == 'rdense':
            reward = -self.dis - np.linalg.norm(self._actions)
        elif self._rewardtype == 'rsparse':
            if delta_dis > 0:
                reward = 0
            else:
                reward = 1

        return reward

    def _sample_action(self):
        if self._isDiscrete == False:
            '''
            if (self._action_dim == 5):
                a = np.random.choice(list(range(3)))
                action = np.array([0.01, 0.01, 0.01, 1, 2])
            elif (self._action_dim == 9):
                action = np.random.choice(list(range(3)))
        else:
        '''
            if (self._action_dim == 5):
                action = np.array(
                    [random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01),
                     random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)])
            elif (self._action_dim == 9):
                action = np.array([random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01),
                                   random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01),
                                   random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01)])
                #a = np.random.uniform(*self.action_bound, size=self.action_dim)
        return action

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    if parse_version(gym.__version__) >= parse_version('0.9.6'):
        render = _render
        reset = _reset
        seed = _seed
        step = _step
