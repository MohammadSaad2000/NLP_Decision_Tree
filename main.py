import pandas as pd
import attributes as att
import decision_tree as dt

def create_dictionary(raw_data):
    word_dict = dict()
    for line in raw_data:
        line = line.strip().lower()
        if line == '':
            continue
        tokens = line.split(' ')
        words = tokens[2:len(tokens) - 1]
        for word in words:
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    return word_dict

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

def get_random_subset(df, fraction):
    return df.sample(frac=fraction, ignore_index=True)



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
word_dict = create_dictionary(train_data)

### create attributes/features
attributes = {
    'Determiner_Before/After': att.DeterminerBeforeOrAfterAttribute(),
    'Or_After': att.OrAfterAttribute()
}
# Add bag of words attributes
for word in word_dict:
    word_count = word_dict[word]
    # ignore words that appear less than 10% of the size the training data
    if word_count > len(train_data) * 0.10:
        attributes[word] = att.WordExistsAttribute(word)

### create attributes table
data = create_attributes_data(train_data, attributes)
column_names = [attribute for attribute in attributes]
column_names.append('Label')
df = pd.DataFrame(data=data, columns=column_names)

### create test cases
test_cases = create_test_cases(test_data, attributes)

### driver code to test accuracy of decision tree
max_depths = [5]
fractions = [0.01]
for max_depth in max_depths:
    for fraction in fractions:
        train_subset = get_random_subset(df, fraction)
        print('\n\n')
        print(train_subset)
        print('SAMPLE SIZE:', len(train_subset.index), '(' + str(fraction * 100) + '% of full training data set)')
        print('Building decision tree with max depth ' + str(max_depth) + '...')
        decision_tree = dt.DecisionTree(train_subset, max_depth)
        decision_tree.build_tree()

        print('Generated Decision Tree:')
        print(decision_tree)

        accuracy = dt.test_accuracy(decision_tree, test_cases)
        print('AVERAGE ACCURACY:', accuracy, '(Averaged over ' + str(-1) + ' trials)')

