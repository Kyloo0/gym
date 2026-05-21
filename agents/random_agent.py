class RandomAgent:

    def __init__(self, action_space):

        self.action_space = action_space

    def predict(self, observation):

        return self.action_space.sample()