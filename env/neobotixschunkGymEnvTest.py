from kukahusky_pybullet_ppo.env.neobotixschunkGymEnv import NeobotixSchunkGymEnv
def main():

    environment = NeobotixSchunkGymEnv(renders=True,isDiscrete=False, maxSteps=1e5, action_dim=9)
    dv = 0.01
    leftwheelVelocitySlider = environment._p.addUserDebugParameter("basevelocity",-dv,dv,0)
    rightwheelVelocitySlider = environment._p.addUserDebugParameter("baseangularvelocity",-dv,dv,0)
    joint_1_Slider=environment._p.addUserDebugParameter("arm_1_joint",-dv,dv,0)
    joint_2_Slider=environment._p.addUserDebugParameter("arm_2_joint",-dv,dv,0)
    joint_3_Slider=environment._p.addUserDebugParameter("arm_3_joint",-dv,dv,0)
    joint_4_Slider=environment._p.addUserDebugParameter("arm_4_joint",-dv,dv,0)
    joint_5_Slider=environment._p.addUserDebugParameter("arm_5_joint",-dv,dv,0)
    joint_6_Slider=environment._p.addUserDebugParameter("arm_6_joint",-dv,dv,0)
    joint_7_Slider=environment._p.addUserDebugParameter("arm_7_joint",-dv,dv,0)
    actionIds = [leftwheelVelocitySlider, rightwheelVelocitySlider, joint_1_Slider, joint_2_Slider, joint_3_Slider, joint_4_Slider, joint_5_Slider, joint_6_Slider, joint_7_Slider]

    done = 0
    while(not done):
        action = []
        for actionId in actionIds:
            action.append(environment._p.readUserDebugParameter(actionId))
        state, reward, done, info = environment.step(action)
        obs = environment.getExtendedObservation()

if __name__=="__main__":
    main()
