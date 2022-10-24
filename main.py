import pandas as pd
import matplotlib.pyplot as plt

import attributes as att
import decision_tree as dt

def increment_dict_count(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1

def create_before_after_dictionaries(raw_data):
    words_before = dict()
    words_after = dict()
    for line in raw_data:
        line = line.strip().lower()
        if line == '':
            continue

        tokens = line.split(' ')
        target_index = int(tokens[1])
        words = tokens[2:len(tokens) - 1]

        if target_index < 0 or target_index >= len(words):
            continue

        if target_index > 0:
            increment_dict_count(words_before, words[target_index - 1])
        else:
            increment_dict_count(words_before, '<s>')

        if target_index < len(words) - 1:
            increment_dict_count(words_after, words[target_index + 1])
        else:
            increment_dict_count(words_after, '</s>')

    #only return top 50 most occurrences
    words_before = {k: v for k, v in sorted(words_before.items(), key=lambda item: item[1], reverse=True)[:50]}
    words_after = {k: v for k, v in sorted(words_after.items(), key=lambda item: item[1], reverse=True)[:50]}
    return words_before, words_after

def create_attributes_data(raw_data, attributes):
    formatted_data = []
    for line in raw_data:
        line = line.strip().lower()
        if line == '':
            continue
        tokens = line.split(' ')
        true_label = tokens[0]
        target_index = int(tokens[1])
        words = tokens[2:len(tokens) - 1]
        if target_index < 0 or target_index >= len(words):
            continue
        row = [str(attributes[attribute].calculate_value(true_label, target_index, words)) for attribute in attributes]
        row.append(true_label)
        formatted_data.append(row)
    return formatted_data

def create_test_cases(raw_data, attributes):
    test_cases = []
    for line in raw_data:
        line = line.strip().lower()
        if line == '':
            continue
        tokens = line.split(' ')
        true_label = tokens[0]
        target_index = int(tokens[1])
        words = tokens[2:len(tokens) - 1]
        if target_index < 0 or target_index >= len(words):
            continue
        test_case = {attribute : str(attributes[attribute].calculate_value(true_label, target_index, words)) for attribute in attributes}
        test_case['Label'] = true_label
        test_cases.append(test_case)
    return test_cases

def get_random_subset(df, num_examples):
    return df.sample(num_examples, ignore_index=True)


### read files
TRAIN_FILE_PATH = 'hw1.train.col'
TEST_FILE_PATH = 'hw1.test.col'     # change to 'hw1.dev.col' to test using dev data
train_file =  open(TRAIN_FILE_PATH, 'r', encoding='utf-8')
test_file = open(TEST_FILE_PATH, 'r', encoding='utf-8')
train_data = train_file.read().split('\n')
test_data = test_file.read().split('\n')
train_file.close()
test_file.close()

### create dictionary with mapping to word frequency
words_before, words_after = create_before_after_dictionaries(train_data)

### create attributes/features
attributes = dict()
# Add bag of words attributes for words before and after the label word
for word in words_before:
    word_count = words_before[word]
    attribute_name = '"' + word + '"_Before'
    attributes[attribute_name] = att.WordExistsBeforeAttribute(word)
for word in words_after:
    word_count = words_after[word]
    attribute_name = '"' + word + '"_After'
    attributes[attribute_name] = att.WordExistsAfterAttribute(word)


### create attributes table
data = create_attributes_data(train_data, attributes)
column_names = [attribute for attribute in attributes]
column_names.append('Label')
df = pd.DataFrame(data=data, columns=column_names)

### create test cases
test_cases = create_test_cases(test_data, attributes)

### driver code to test accuracy of decision tree
# change these 3 variables if needed
max_depth = 10
num_trials = 3
sample_percentage = [0.1, 0.2, 0.5, 0.8, 1]

result_pairs = []
divider_str = ''.rjust(50, '=')
for sample_percentage in sample_percentage:
    print(divider_str)
    accuracy_avg = 0
    num_train_examples = int(len(df.index) * sample_percentage)
    print('SAMPLE SIZE:', num_train_examples, '(' + str(sample_percentage * 100) + '% of full training data set)')
    for i in range(num_trials):
        train_subset = get_random_subset(df, num_train_examples)
        print('Building decision tree with max depth ' + str(max_depth) + '...')

        decision_tree = dt.DecisionTree(train_subset, max_depth)
        decision_tree.build_tree()

        print('GENERATED DECISION TREE:')
        print(decision_tree)

        trial_accuracy = dt.test_accuracy(decision_tree, test_cases)
        accuracy_avg += trial_accuracy
        print("Accuracy for trial #" + str(i + 1) + ": " + str(round(trial_accuracy * 100, 2)) + ".")
    accuracy_avg /= num_trials
    accuracy_percentage = round(accuracy_avg * 100, 2)
    result_pairs.append([num_train_examples, accuracy_percentage])
    print('\nAVERAGE ACCURACY: ' + str(accuracy_percentage) + '% (Averaged over ' + str(num_trials) + ' trials)')

print(divider_str)
print(divider_str)
print("RESULT:")
result_table = pd.DataFrame(data=result_pairs, columns=['# Train Examples', 'Avg Accuracy'])
print(result_table)
result_table.plot(x='# Train Examples', y='Avg Accuracy')
plt.title('Performance on ID3 DT with Max Depth of ' + str(max_depth))
plt.show()
