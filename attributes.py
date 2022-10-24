# This is an interface
class Attribute:
    # This is an abstract method and is meant to be overriden by inheriting classes
    def calculate_value(self, true_label, target_index, words):
        pass


class WordExistsBeforeAttribute(Attribute):
    def __init__(self, word):
        super().__init__()
        self.word = word

    def calculate_value(self, true_label, target_index, words):
        return target_index > 0 and words[target_index - 1] == self.word


class WordExistsAfterAttribute(Attribute):
    def __init__(self, word):
        super().__init__()
        self.word = word

    def calculate_value(self, true_label, target_index, words):
        return target_index < len(words) - 1 and words[target_index + 1] == self.word


