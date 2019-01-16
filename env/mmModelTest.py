import pybullet as p
import pybullet_data as pd

import time

datapath = pd.getDataPath()
print(datapath)
p.connect(p.GUI)
p.setAdditionalSearchPath(datapath)
p.setGravity(0,0,-9.8)
plane= p.loadURDF("plane.urdf")

mm = p.loadURDF("/husky/husky.urdf")
#mm = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")
#for i in range(100):
#    time.sleep(1./10.)
#p.disconnect()
#p.setRealTimeSimulation(1)

for i in range(p.getNumJoints(mm)):
    print(p.getJointInfo(mm, i))

useSimulation = 0
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)
#cid = p.createConstraint(mm,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0,0,0],[0,0,1])
wheels = [0, 1]
arm = [7,8,9,10,11,12,13]

wheelVelocities=[0,0]
wheelDeltasTurn=[1,-1]
wheelDeltasFwd=[1,1]

targetVelocitySliderl = p.addUserDebugParameter("wheelVelocityleft",-30,30,0)
targetVelocitySliderr = p.addUserDebugParameter("wheelVelocityright",-30,30,0)
targetPositonSlider = p.addUserDebugParameter('joint10Position', -2.7, 2.7, 0)
maxForceSlider = p.addUserDebugParameter("maxForce",-10,10,10)
#steeringSlider = p.addUserDebugParameter("steering",-0.5,0.5,0)

while(True):
    maxForce = p.readUserDebugParameter(maxForceSlider)
    targetVelocityl = p.readUserDebugParameter(targetVelocitySliderl)
    targetVelocityr = p.readUserDebugParameter(targetVelocitySliderr)
    #steeringAngle = p.readUserDebugParameter(steeringSlider)
    targetArmPosition = p.readUserDebugParameter(targetPositonSlider)

    p.setJointMotorControl2(mm, 1, controlMode=p.VELOCITY_CONTROL, targetVelocity=targetVelocityl,force=maxForce)
    p.setJointMotorControl2(mm, 2, controlMode=p.VELOCITY_CONTROL, targetVelocity=targetVelocityr, force=maxForce)

    #p.setJointMotorControl2(mm, 9, controlMode=p.POSITION_CONTROL, targetPosition = targetArmPosition)

    keys = p.getKeyboardEvents()
    shift = 0.01
    wheelVelocities = [0, 0]
    speed = 1.0
    for k in keys:

        if p.B3G_LEFT_ARROW in keys:
            for i in range(len(wheels)):
                wheelVelocities[i] = wheelVelocities[i] - speed * wheelDeltasTurn[i]
        if p.B3G_RIGHT_ARROW in keys:
            for i in range(len(wheels)):
                wheelVelocities[i] = wheelVelocities[i] + speed * wheelDeltasTurn[i]
        if p.B3G_UP_ARROW in keys:
            for i in range(len(wheels)):
                wheelVelocities[i] = wheelVelocities[i] + speed * wheelDeltasFwd[i]
        if p.B3G_DOWN_ARROW in keys:
            for i in range(len(wheels)):
                wheelVelocities[i] = wheelVelocities[i] - speed * wheelDeltasFwd[i]

    for i in range(len(wheels)):
        p.setJointMotorControl2(mm, wheels[i], controlMode=p.VELOCITY_CONTROL, targetVelocity=wheelVelocities[i], force=1000)
    # p.resetBasePositionAndOrientation(kukaId,basepos,baseorn)#[0,0,0,1])
    if (useRealTimeSimulation):
        t = time.time()  # (dt, micro) = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f').split('.')
    # t = (dt.second/60.)*2.*math.pi
    else:
        t = t + 0.001

    if (useSimulation and useRealTimeSimulation == 0):
        p.stepSimulation()

    #if (0):
        #p.stepSimulation()
    #time.sleep(0.01)