
class Node:

    def __init__(self, value):
        self.value = value
        self.children = list()
        self.occurrence = 1
        self.probability = 0

    def get_value(self):
        return self.value

    def set_occurrence(self, occurrence):
        self.occurrence = occurrence
    
    def get_probability(self):
        return self.probability

    def set_probability(self, prob):
        self.probability = prob

    def increment_occurrence(self):
        self.occurrence = self.occurrence+1

    def get_occurrence(self):
        return self.occurrence

    def set_value(self, value):
        self.value = value

    def add_child(self, child):
        self.children.append(child)

    def get_children(self):
        return self.children

    def get_children_by_value(self, value):
        for i in self.children:
            if i.get_value() == value:
                return i
        return None
