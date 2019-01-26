import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)


import pybullet as p
import numpy as np
import pybullet_data
import math


class NeobotixSchunk:

    def __init__(self, urdfRootPath=parentdir, timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 1.5
        self.maxForce = 100
        self.useSimulation = 1
        self.useNullSpace = 0
        self.useOrientation = 1
        self.neobotixschunkEndEffectorIndex = 13
        self.reset()

        # lower limits for null space
        self.ll = [-3.1215926, -2.12, -3.1215926, -2.16, -3.1215926, -2.07, -2.94]
        # upper limits for null space
        self.ul = [3.1215926, 2.12, 3.1215926, 2.16, 3.1215926, 2.07, 2.94]
        # joint ranges for null space
        self.jr = [6.24, 4.24, 6.24, 4.32, 6.24, 4.14, 5.88]
        # joint damping coefficents
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    def reset(self):
        #os.path.join: path combination and return
        #load urdf returns a uniqueId,a noninteger value
        #self.neobotixschunkUid = p.loadURDF(os.path.join(self.urdfRootPath, "kukahusky_pybullet_ppo/data/neobotixschunk/NeobotixSchunk.urdf"))
        self.neobotixschunkUid = p.loadURDF(os.path.join(self.urdfRootPath, "kukahusky_pybullet_ppo/data/neobotixschunk/mp500lwa4d.urdf"))

        for i in range(p.getNumJoints(self.neobotixschunkUid)):
            print(p.getJointInfo(self.neobotixschunkUid, i))
        huskyPos, huskyOrn = p.getBasePositionAndOrientation(self.neobotixschunkUid)
        huskyEul = p.getEulerFromQuaternion(huskyOrn)
        # print('base',huskyPos, huskyOrn, huskyEul)
        self.wheels = [0, 1]
        self.joints = [6, 7, 8, 9, 10, 11, 12]

        initial_wheelVel=[0, 0]
        initial_jointstate=[0, 0, 0, 0, 0, 0, 0]
        self.basevelocity = 0
        self.baseangulervelocity = 0


        for wheel in self.wheels:
            p.resetJointState(self.neobotixschunkUid, wheel,initial_wheelVel[wheel])
            p.setJointMotorControl2(self.neobotixschunkUid, wheel, p.VELOCITY_CONTROL, targetVelocity=initial_wheelVel[wheel],force=self.maxForce)

        for joint in self.joints:
            p.resetJointState(self.neobotixschunkUid,joint,initial_jointstate[joint-6])
            p.setJointMotorControl2(self.neobotixschunkUid, joint, p.POSITION_CONTROL,targetPosition=initial_jointstate[joint-6], force=self.maxForce)

    def getActionDimension(self):
        return 5

    def getObservationDimension(self):
        return len(self.getObservation())

    #get the endeffector position and orientation(euler angles)
    def getObservation(self):
        observation = []
        state = p.getLinkState(self.neobotixschunkUid, self.neobotixschunkEndEffectorIndex)
        # print(state)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def check_baseV(self, base_vel, delta_bv):
        if (abs(base_vel) > 1.5):
            base_vel = base_vel - delta_bv
        return base_vel

    def check_baseA(self, base_ang, delta_ba):
        if (abs(base_ang) > 2):
            base_ang = base_ang - delta_ba
        return base_ang

    def applyAction(self, action):
        # dtargetVelocityL = action[0]
        # dtargetVelocityR = action[1]
        # targetvelocity=[dtargetVelocityL,dtargetVelocityR]
        # print('targetVelocity=',targetvelocity)
        basevelocity = action[0]
        baseangulervelocity = action[1]
        djoint_1 = action[2]
        djoint_2 = action[3]
        djoint_3 = action[4]
        djoint_4 = action[5]
        djoint_5 = action[6]
        djoint_6 = action[7]
        djoint_7 = action[8]

        self.basevelocity = self.basevelocity + basevelocity
        self.baseangulervelocity = self.baseangulervelocity + baseangulervelocity

        self.basevelocity = self.check_baseV(self.basevelocity, basevelocity)
        self.baseangulervelocity = self.check_baseA(self.baseangulervelocity, baseangulervelocity)
        #vel relations from datasheet and https://github.com/neobotix/neo_kinematics_differential/blob/master/common/src/DiffDrive2WKinematics.cpp
        self.wheelVelR = ( self.basevelocity + 0.2535 * self.baseangulervelocity) / 0.13
        self.wheelVelL = ( self.basevelocity - 0.2535 * self.baseangulervelocity) / 0.13

        self.wheelstate = [self.wheelVelL , self.wheelVelR]
        #print('wheel vel:', self.wheelVelR, self.wheelVelL, self.basevelocity)
        # dwe = []
        # for joint in self.wheels:
        #     self.Wheelstate = p.getJointState(self.neobotixschunkUid, joint)
        #     self.wheelstate = self.Wheelstate[1]
        #     dwe.append(self.wheelstate)

        # dwl = dtargetVelocityL+dwe[0]
        # dwr = dtargetVelocityR+dwe[1]
        # self.wheelstate= [dwl,dwr]
        dae = []
        for joint in self.joints:
            self.Jointstate=p.getJointState(self.neobotixschunkUid,joint)
            self.jointstate=self.Jointstate[0]
            dae.append(self.jointstate)

        # print('dae',dae)
        da_1 = djoint_1+dae[0]
        da_2 = djoint_2+dae[1]
        da_3 = djoint_3+dae[2]
        da_4 = djoint_4+dae[3]
        da_5 = djoint_5+dae[4]
        da_6 = djoint_6+dae[5]
        da_7 = djoint_7+dae[6]
        self.jointstate= [da_1,da_2,da_3,da_4,da_5,da_6,da_7]

        # print('targetAngle=', targetangle)
        for motor in self.wheels:
            p.setJointMotorControl2(self.neobotixschunkUid, motor,p.VELOCITY_CONTROL,
                                          targetVelocity=self.wheelstate[motor], force=self.maxForce)
        for motor_2 in self.joints:
            p.setJointMotorControl2(self.neobotixschunkUid, motor_2, p.POSITION_CONTROL,
                                          targetPosition=self.jointstate[motor_2-6],force=self.maxForce)
