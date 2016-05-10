import gym

def transformEnvState(env, transform):
    """Transform an environment's state.

    Affects reset and step methods
    """
    oldStep = env.step
    oldReset = env.reset
    def newStep(*args, **kwargs):
        state, action, done, info = oldStep(*args, **kwargs)
        return transform(state), action, done, info
    def newReset(*args, **kwargs):
        return transform(oldReset(*args, **kwargs))
    env.step = newStep
    env.reset = newReset

def insertRewardAccumulator(env):
    """Adds a member to class which accumulates reward, and resets to 0 when
    reset() is called
    """
    oldStep = env.step
    oldReset = env.reset
    env.total_reward_since_reset = 0
    def newStep(*args, **kwargs):
        ret = oldStep(*args, **kwargs)
        reward = ret[1]
        env.total_reward_since_reset = reward + env.total_reward_since_reset
        return ret
    def newReset(*args, **kwargs):
        env.total_reward_since_reset = 0
        return oldReset(*args, **kwargs)
    env.step = newStep
    env.reset = newReset
