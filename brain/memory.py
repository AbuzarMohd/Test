class ChatMemory:
    def __init__(self):
        self.history = []

    def add(self, role, text):
        self.history.append((role, text))

    def last_is_user(self):
        return self.history and self.history[-1][0] == "user"
