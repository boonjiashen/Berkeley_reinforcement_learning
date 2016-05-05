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

