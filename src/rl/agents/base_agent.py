class BaseAgent:
    def select_action(self, state):
        raise NotImplementedError
    def update(self, *args, **kwargs):
        raise NotImplementedError
    def save(self, path):
        raise NotImplementedError
    def load(self, path):
        raise NotImplementedError 