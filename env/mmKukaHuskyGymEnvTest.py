from env.mmKukaHuskyGymEnv import MMKukaHuskyGymEnv

def main():
    env = MMKukaHuskyGymEnv(renders=True, isDiscrete=False, action_dim = 9, rewardtype='rdense', randomInitial=False)

    motorsIds = []
    dv = 0.01
    if(env._action_dim == 5): # use ee position changes as arm action
        motorsIds.append(env._p.addUserDebugParameter("kuka_ee_dx", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_ee_dy", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_ee_dz", -dv, dv, 0))
    elif(env._action_dim == 9): # use joint states changes as arm action
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_0", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_1", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_2", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_3", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_4", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_5", -dv, dv, 0))
        motorsIds.append(env._p.addUserDebugParameter("kuka_arm_6", -dv, dv, 0))

    bs = 1
    # linear and angular velocity as mobile base
    # motorsIds.append(env._p.addUserDebugParameter("base_linear_speed", -bs, bs, 0))
    # motorsIds.append(env._p.addUserDebugParameter("base_angular_speed", -2*bs, 2*bs, 0))
    motorsIds.append(env._p.addUserDebugParameter("base_linear_speed", -dv, dv, 0))
    motorsIds.append(env._p.addUserDebugParameter("base_angular_speed", -dv, dv, 0))

    done = False
    while (not done):
        # env.reset()
        # env.render()
        action = []
        for motorId in motorsIds:
            action.append(env._p.readUserDebugParameter(motorId))
        # print('action。 ', action)
        state, reward, done, info = env.step(action)
        # print('r',reward)
        # state, reward, done, info = env.step(env._sample_action())
        obs = env.getExtendedObservation()
        # env.action_space()

if __name__ == "__main__":
    main()
