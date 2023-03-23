class Settings:
    def __init__(self):
        self.oversample = False
        self.remove_stop_words = False

    def enable_oversample(self):
        self.oversample = True
        return self

    def enable_stop_word_removal(self):
        self.remove_stop_words = True
        return self
