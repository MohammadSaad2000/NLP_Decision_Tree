import math


# used for calculation of entropy
# without having to worry about values <= 0
def safe_log2(value):
    return 0 if value <= 0 else math.log(value, 2)


def calculate_entropy(df):
    entropy = 0
    unique_labels = df['Label'].unique()
    for label in unique_labels:
        relative_freq = df['Label'].value_counts()[label] / len(df['Label'])
        entropy += -relative_freq * safe_log2(relative_freq)
    return entropy


def calculate_normalized_entropy_for_attribute(df, attribute):
    unique_labels = df['Label'].unique()
    unique_value_for_att = df[attribute].unique()
    entropy_normalized = 0
    for value in unique_value_for_att:
        entropy = 0
        for label in unique_labels:
            num_value_occurrences_for_label = len(df[(df['Label'] == label) & (df[attribute] == value)])
            relative_freq = num_value_occurrences_for_label / df[attribute].value_counts()[value]
            entropy += -relative_freq * safe_log2(relative_freq)
        num_value_occurrences = len(df.loc[df[attribute] == value])
        entropy_normalized += entropy * (num_value_occurrences / len(df[attribute]))
    return entropy_normalized


def calculate_info_gain_for_attribute(df, attribute):
    return calculate_entropy(df) - calculate_normalized_entropy_for_attribute(df, attribute)


def get_max_info_gain_attribute(df):
    data_only_df = df.drop('Label', axis=1)
    max_info_gain = -1
    max_info_gain_attribute = None
    for attribute in data_only_df.columns:
        current_att_info_gain = calculate_info_gain_for_attribute(df, attribute)
        if current_att_info_gain > max_info_gain:
            max_info_gain = current_att_info_gain
            max_info_gain_attribute = attribute
    return max_info_gain_attribute


def get_subtable_for_attribute_value(df, attribute, value):
    return df[df[attribute] == value].reset_index(drop=True)


def subtable_is_homogenous(df):
    return len(df['Label'].unique()) == 1


def test_accuracy(decision_tree, test_cases):
    correct_counter = 0
    for test_case in test_cases:
        predicted_label = decision_tree.decide(test_case)
        if predicted_label == test_case['Label']:
            correct_counter += 1
    return correct_counter / len(test_cases)


class Node:
    def __init__(self, subtable, attribute):
        self.subtable = subtable
        self.attribute = attribute
        self.children = []

    def get_label(self):
        return self.subtable['Label'].value_counts().idxmax()

    def __repr__(self):
        if self.attribute is None:
            return 'ROOT'
        s = self.attribute + ': ' + self.subtable[self.attribute][0]
        if len(self.children) == 0:
            s += (' --> ' + self.get_label())
        return s


class DecisionTree:

    def __init__(self, df, max_depth):
        self.root = Node(df, None)
        self.max_depth = max_depth

    def build_tree(self):
        stack = [{'current_node': self.root, 'current_level': 0}]
        while len(stack) > 0:
            current_level = stack[-1]['current_level']
            current_node = stack[-1]['current_node']
            stack.pop()

            if current_level >= self.max_depth:
                continue

            attribute_to_split_on = get_max_info_gain_attribute(current_node.subtable)
            unique_values_for_attribute = current_node.subtable[attribute_to_split_on].unique()
            for value in unique_values_for_attribute:
                new_node_subtable = get_subtable_for_attribute_value(current_node.subtable, attribute_to_split_on,
                                                                     value)
                new_node = Node(new_node_subtable, attribute_to_split_on)
                current_node.children.append(new_node)
                if not subtable_is_homogenous(new_node_subtable):
                    stack.append({'current_node': new_node, 'current_level': current_level + 1})

    def decide(self, test_case):
        stack = [self.root]
        while len(stack) > 0:
            current_node = stack.pop()
            if len(current_node.children) == 0:
                return current_node.get_label()
            for child in current_node.children:
                if child.subtable[child.attribute][0] == test_case[child.attribute]:
                    stack.append(child)

    def __repr__(self):
        s = ""
        stack = [{'current_node': self.root, 'current_level': 0}]
        while len(stack) > 0:
            current_level = stack[-1]['current_level']
            current_node = stack[-1]['current_node']
            stack.pop()

            if current_node is None:
                continue

            padding_str = ''.rjust(current_level, '\t')
            node_str = current_node.__repr__()
            s += (padding_str + node_str + '\n')

            for child in current_node.children:
                stack.append({'current_node': child, 'current_level': current_level + 1})
        return s
