class Settings:
    def __init__(self):
        self.oversample = False

    def enable_oversample(self):
        self.oversample = True
        return self
