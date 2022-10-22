# This is an interface
class Attribute:
    # This is an abstract method and is meant to be overriden by inheriting classes
    def calculate_value(self, true_label, target_index, words):
        pass


class DeterminerBeforeOrAfterAttribute(Attribute):
    def calculate_value(self, true_label, target_index, words):
        if target_index > 0 and words[target_index - 1] in ["the", "a"]:
            return "Before"
        if target_index < len(words) - 1 and words[target_index + 1] in ["the", "a"]:
            return "After"
        return None


class OrAfterAttribute(Attribute):
    def calculate_value(self, true_label, target_index, words):
        return target_index < len(words) - 1 and words[target_index + 1] == "or"


class WordExistsAttribute(Attribute):
    def __init__(self, word):
        super().__init__()
        self.word = word

    def calculate_value(self, true_label, target_index, words):
        return self.word in words


