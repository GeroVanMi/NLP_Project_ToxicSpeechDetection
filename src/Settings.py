class Settings:
    def __init__(self):
        self.oversample = False
        self.lower_case = False
        self.remove_stop_words = False

    def enable_oversample(self):
        self.oversample = True
        return self

    def enable_stop_word_removal(self):
        self.remove_stop_words = True
        return self

    def enable_lower_case(self):
        self.lower_case = True
        return self
