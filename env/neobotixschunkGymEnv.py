import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import time
import random
from . import neobotixschunk
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



class NeobotixSchunkGymEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 urdfRoot=parentdir,
                 actionRepeat=50,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False,
                 maxSteps=1000,
                 rewardtype='rdense'):
        # print("init")
        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._rewardtype = rewardtype
        self._maxSteps = maxSteps
        self._isDiscrete = isDiscrete
        self.terminated = 0
        self._cam_dist = 4
        self._cam_yaw = 180
        self._cam_pitch = -40
        self._p = p
        self._dis_vor = 100
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)
        self._seed()
        self.reset()
        observationDim = len(self.getExtendedObservation())
        # print("observationDim")
        # print(observationDim)
        # observation_high = np.array([np.finfo(np.float32).max] * observationDim)

        observation_high = np.array([largeValObservation] * observationDim)

        if (isDiscrete):
            self.action_space = spaces.Discrete(9)
        else:
            action_dim = 5
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.terminated = 0
        p.resetSimulation()
        p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(numSolverIterations=150)

        p.loadURDF(os.path.join(self._urdfRoot, "kukahusky_pybullet_ppo/data/plane.urdf"), [0, 0, 0])
        # print 'path', self._urdfRoot
        xpos = random.uniform(-1, 1)
        ypos = random.uniform(-1, 1)
        zpos = random.uniform(0.5, 1.4)
        self.goal = [xpos, ypos, zpos]

        self.goalUid = p.loadURDF(os.path.join(self._urdfRoot, "kukahusky_pybullet_ppo/data/spheregoal.urdf"), xpos, ypos, zpos)
        #private variante
        self._neobotixschunk = neobotixschunk.NeobotixSchunk(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        self._envStepCounter = 0
        p.stepSimulation()
        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    #return the endeffector vec9 [position(vec3),orientation(euler angles)(vec3),goalPosInEndeffector(vec3)],distance, goal position
    def getExtendedObservation(self):
        self._observation = self._neobotixschunk.getObservation()
        EndeffectorState = p.getLinkState(self._neobotixschunk.neobotixschunkUid, self._neobotixschunk.neobotixschunkEndEffectorIndex)
        # print('EndeffectorState',EndeffectorState)
        # show the position of endeffector, is vec3
        EndeffectorrPos = EndeffectorState[0]
        #show the orientation of endeffector, is vec4 in quaternion
        EndeffectorOrn = EndeffectorState[1]
        # returns the position(vec3) and quaternion orientation(vec4)
        goalPos, goalOrn = p.getBasePositionAndOrientation(self.goalUid)
        #show the inverse transformed matrix
        invEndeffectorPos, invEndeffectorOrn = self._p.invertTransform(EndeffectorrPos, EndeffectorOrn)
        #multiply the transformed matrix and goal position
        goalPosInEndeffector, goalOrnInEndeffector = self._p.multiplyTransforms(invEndeffectorPos, invEndeffectorOrn, goalPos, goalOrn)
        # print('goalPosInEndeffector',goalPosInEndeffector)
        goalInEndeffectorPosXYEulZ = [goalPosInEndeffector[0], goalPosInEndeffector[1], goalPosInEndeffector[2]]
        #at end of list is relative coordinate system (goal location in endeffector position)
        self._observation.extend(list(goalInEndeffectorPosXYEulZ))
        print('bav', p.getBaseVelocity(self._neobotixschunk.neobotixschunkUid))
        return self._observation


    def __del__(self):
        p.disconnect()

    def _step(self, action):
        for i in range(self._actionRepeat):
            self._neobotixschunk.applyAction(action)
            p.stepSimulation()
            if self._termination():
                break
            self._envStepCounter += 1

        if self._renders:
            time.sleep(self._timeStep)

        self._observation = self.getExtendedObservation()
        self._actions = action
        reward = self._reward()
        done = self._termination()
        return np.array(self._observation), reward, done, {}

    def _termination(self):
        state = p.getLinkState(self._neobotixschunk.neobotixschunkUid, self._neobotixschunk.neobotixschunkEndEffectorIndex)
        actualEndEffectorPos = state[0]

        if (self.terminated or self._envStepCounter > self._maxSteps):
            self._observation = self.getExtendedObservation()
            return True

        disvec = [x - y for x, y in zip(actualEndEffectorPos, self.goal)]
        #calculate the linear algebra normiert distance
        dis = np.linalg.norm(disvec)

        if dis < 0.1:
            self.terminated = 1
            self._observation = self.getExtendedObservation()
            print('terminate:', self._observation, dis, self.goal)
            return True

        return False

    def _reward(self):
    # rewards is accuracy of target position
    #     closestPoints = self._p.getClosestPoints(self._neobotixschunk.neobotixschunkUid, self.goalUid, 1000,self._neobotixschunk.neobotixschunkEndEffectorIndex,-1)
    #
    #     numPt = len(closestPoints)
    #     reward = -1000
    #     if (numPt > 0):
    #         reward = -closestPoints[0][8]  # contact distance
    #     return reward
        state = p.getLinkState(self._neobotixschunk.neobotixschunkUid, self._neobotixschunk.neobotixschunkEndEffectorIndex)
        actualEndEffectorPos = state[0]
        disvec = [x - y for x, y in zip(actualEndEffectorPos, self.goal)]
        self.dis = np.linalg.norm(disvec)
        delta_dis = self.dis - self._dis_vor
        self._dis_vor = self.dis
        if self._rewardtype == 'rdense':
            reward = -self.dis #- np.linalg.norm(self._actions)
        elif self._rewardtype == 'rsparse':
            if delta_dis > 0:
                reward = 0
            else:
                reward = 1
        return reward

    def _render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos, orn = self._p.getBasePositionAndOrientation(self._neobotixschunk.neobotixschunkUid)
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
            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    if parse_version(gym.__version__)>=parse_version('0.9.6'):
        render = _render
        reset = _reset
        seed = _seed
        step = _step

