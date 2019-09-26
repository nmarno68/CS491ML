# I am so sorry for the code you are about to see

import numpy


class Node:
    def __init__(self, index, llabel, rlabel, lchild, rchild, feature_value):
        self.featureIndex = index
        self.leftLabel = llabel
        self.rightLabel = rlabel
        self.children = {'left': lchild, 'right': rchild}
        self.featureValue = 0

    def height(self):

        if self.children['left'] == 0 and self.children['right'] == 0:
            return 1
        if self.children['left'] == 0:
            return self.children['right'].height() + 1
        if self.children['right'] == 0:
            return self.children['left'].height() + 1
        return max(self.children['left'].height(), self.children['right'].height()) + 1


class TrainBinary:

    def __init__(self):
        self.featureData = numpy.array([[]])
        self.labelData = numpy.array([[]])
        self.trainingTree = 0

    def entropy(self, probability):
        if probability <= 0:
            return 0
        else:
            return -probability * numpy.log2(probability)

    def recursive_train_binary(self, samples, features, depth):

        if depth != 0 and len(features) != 0:
            total_samples = len(samples)
            feature_data = {'information_gain': 0}
            set_entropy = 0

            for feature in features:
                feature_true_label_true = 0
                feature_false_label_true = 0
                left_samples = numpy.array([])
                right_samples = numpy.array([])
                # Looking at feature for each sample, collect data
                for sample in samples:

                    if self.featureData[int(sample)][int(feature)]:
                        # Track samples on right of split
                        right_samples = numpy.append(right_samples, sample)

                        if self.labelData[int(sample)][0]:
                            # Count number of true labels on right of split
                            feature_true_label_true += 1
                    else:

                        # Track samples on left of split
                        left_samples = numpy.append(left_samples, sample)
                        if self.labelData[int(sample)][0]:
                            # Count number of true labels on right of split
                            feature_false_label_true += 1

                # If either side has zero samples, information gain will be 0
                if len(left_samples) != 0 and len(right_samples != 0):
                    # Calculate entropy on the left side of the split on feature
                    p_label_true = feature_false_label_true / len(left_samples)
                    p_label_false = 1 - p_label_true

                    label_left = 1 if p_label_true > .5 else 0

                    entropy_left = self.entropy(p_label_true) + self.entropy(p_label_false)

                    # Calculate entropy on the right side of the split on feature
                    p_label_true = feature_true_label_true / len(right_samples)
                    p_label_false = 1 - p_label_true

                    # DO NOT FREAK OUT IF LABELS ON BOTH SIDES ARE SAME THIS IS MY BIAS RIGHT HERE ITS FINE THERE
                    # WAS JUST A 50/50 SPLIT
                    label_right = 1 if p_label_true > .5 else 0

                    entropy_right = self.entropy(p_label_true) + self.entropy(p_label_false)

                    # Calculate entropy of whole set
                    p_label_true = (feature_false_label_true + feature_true_label_true) / total_samples
                    p_label_false = 1 - p_label_true
                    set_entropy = self.entropy(p_label_true) + self.entropy(p_label_false)

                    # Calculate information gain of feature and save all necessary information of feature (IG,
                    # samples left, samples right, label left, label right)

                    information_gain = set_entropy - (len(left_samples) / total_samples) * entropy_left - \
                                       (len(right_samples) / total_samples) * entropy_right

                    # If this is the best feature to split on so far update feature info
                    if information_gain > feature_data['information_gain']:
                        feature_data['information_gain'] = information_gain
                        feature_data['feature_index'] = feature
                        feature_data['left_samples'] = left_samples
                        feature_data['right_samples'] = right_samples
                        feature_data['label_left'] = label_left
                        feature_data['label_right'] = label_right

            if 'feature_index' in feature_data:
                # Remove chosen feature from list of possible features
                features = features[features != feature_data['feature_index']]

                # Create new tree node and call function on left and right samples if we haven't hit max IG
                if feature_data['information_gain'] != set_entropy:
                    # index, llabel, rlabel, lchild, rchild
                    # print("Best choice keep going:")
                    # print(feature_data)
                    return Node(feature_data['feature_index'], feature_data['label_left'], feature_data['label_right'],
                                self.recursive_train_binary(feature_data['left_samples'], features, depth - 1),
                                self.recursive_train_binary(feature_data['right_samples'], features, depth - 1), 0)
                else:
                    # print("Best choice we're done:")
                    # print(feature_data)
                    return Node(feature_data['feature_index'], feature_data['label_left'], feature_data['label_right'],
                                0, 0, 0)
            else:
                # No feature to split on (None of the features improve entropy ~ IG = 0)
                return 0

        else:
            # No feature to split on (max depth reached or no more features)
            return 0


def DT_train_binary(X, Y, max_depth):
    trainer = TrainBinary()
    trainer.featureData = X
    trainer.labelData = Y
    available_features = numpy.arange(0, len(X[0]))
    available_samples = numpy.arange(0, len(X))

    return trainer.recursive_train_binary(available_samples, available_features, max_depth)


def test_binary_helper(samples, labels, node, all_samples):
    left_samples = numpy.array([])
    right_samples = numpy.array([])
    left_correct = 0
    right_correct = 0

    for sample in samples:
        if all_samples[int(sample)][int(node.featureIndex)] == 0:
            left_samples = numpy.append(left_samples, sample)
        else:
            right_samples = numpy.append(right_samples, sample)

    if node.children['left'] != 0:
        left_correct = test_binary_helper(left_samples, labels, node.children['left'], all_samples)
    else:
        for sample in left_samples:
            if labels[int(sample)] == node.leftLabel:
                left_correct += 1

    if node.children['right'] != 0:
        right_correct = test_binary_helper(right_samples, labels, node.children['right'], all_samples)
    else:
        for sample in right_samples:
            if labels[int(sample)] == node.rightLabel:
                right_correct += 1
    return right_correct + left_correct


def DT_test_binary(X, Y, DT):
    all_samples = numpy.arange(0, len(X))
    number_correct = test_binary_helper(all_samples, Y, DT, X)
    return number_correct / len(Y)


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):
    i = 3
    forest = numpy.array([])
    forest = numpy.append(forest, DT_train_binary(X_train, Y_train, 1))
    forest = numpy.append(forest, DT_train_binary(X_train, Y_train, 2))

    while forest[len(forest) - 1].height() != forest[len(forest) - 2].height():
        forest = numpy.append(forest, DT_train_binary(X_train, Y_train, i))
        i += 1

    accuracy = numpy.array([])
    for trees in forest:
        accuracy = numpy.append(accuracy, DT_test_binary(X_val, Y_val, trees))

    return forest[numpy.argmax(accuracy)]


def DT_make_prediction(x, DT):

    feature = x[DT.featureIndex]
    prediction = -1
    if feature == 0:
        if DT.children['left'] != 0:
            prediction = DT_make_prediction(x, DT.children['left'])
        else:
            prediction = DT.leftLabel
    else:
        if DT.children['right'] != 0:
            prediction = DT_make_prediction(x, DT.children['right'])
        else:
            prediction = DT.rightLabel

    return prediction


def DT_train_real(X, Y, max_depth):
    all_samples = numpy.arange(0, len(X))
    return train_real_helper(X, all_samples, Y, max_depth)


def entropy(probability):
    if probability <= 0:
        return 0
    else:
        return -probability * numpy.log2(probability)


def train_real_helper(samples, sample_indices, labels, max_depth):

    total_samples = len(sample_indices)
    feature_data = {'information_gain': 0}
    set_entropy = 0

    # if we haven't reached the max depth
    if max_depth != 0:

        # for each sample
        for sample in sample_indices:

            i = -1
            # and each feature in the sample
            for feature in samples[int(sample)]:
                i += 1
                # loop through all the samples splitting them to left and right based on current feature
                # and sample value
                left_samples = numpy.array([])
                right_samples = numpy.array([])
                feature_left_label_true = 0
                feature_right_label_true = 0

                for single_sample in sample_indices:

                    if samples[int(single_sample)][i] < feature:
                        left_samples = numpy.append(left_samples, single_sample)

                        if labels[int(single_sample)] == 1:
                            feature_left_label_true += 1
                    else:
                        right_samples = numpy.append(right_samples, single_sample)

                        if labels[int(single_sample)] == 1:
                            feature_right_label_true += 1

                if len(left_samples) != 0 and len(right_samples != 0):
                    # calculate entropy on each side

                    # Calculate entropy on the left side of the split on feature
                    p_label_true = feature_left_label_true / len(left_samples)
                    p_label_false = 1 - p_label_true

                    label_left = 1 if p_label_true > .5 else 0

                    entropy_left = entropy(p_label_true) + entropy(p_label_false)

                    # Calculate entropy on the right side of the split on feature
                    p_label_true = feature_right_label_true / len(right_samples)
                    p_label_false = 1 - p_label_true

                    # DO NOT FREAK OUT IF LABELS ON BOTH SIDES ARE SAME THIS IS MY BIAS RIGHT HERE ITS FINE THERE
                    # WAS JUST A 50/50 SPLIT
                    label_right = 1 if p_label_true > .5 else 0

                    entropy_right = entropy(p_label_true) + entropy(p_label_false)

                    # Calculate entropy of whole set
                    p_label_true = (feature_left_label_true + feature_right_label_true) / total_samples
                    p_label_false = 1 - p_label_true
                    set_entropy = entropy(p_label_true) + entropy(p_label_false)

                    # Calculate information gain of feature and save all necessary information of feature (IG,
                    # samples left, samples right, label left, label right)

                    information_gain = set_entropy - (len(left_samples) / total_samples) * entropy_left - \
                                       (len(right_samples) / total_samples) * entropy_right

                    # calculate IG, store, IG, feature, sample, value, left, and right
                    if information_gain > feature_data['information_gain']:
                        feature_data['information_gain'] = information_gain
                        feature_data['feature_index'] = i
                        feature_data['feature_value'] = feature
                        feature_data['labelLeft'] = label_left
                        feature_data['labelRight'] = label_right
                        feature_data['samplesLeft'] = left_samples
                        feature_data['samplesRight'] = right_samples

        if 'feature_index' in feature_data:

            # Create new tree node and call function on left and right samples if we haven't hit max IG
            if feature_data['information_gain'] != set_entropy:

                return Node(feature_data['feature_index'],
                            feature_data['labelLeft'],
                            feature_data['labelRight'],
                            train_real_helper(samples, feature_data['samplesLeft'], labels, max_depth - 1),
                            train_real_helper(samples, feature_data['samplesRight'], labels, max_depth - 1),
                            feature_data['feature_value'])
            else:

                return Node(feature_data['feature_index'],
                            feature_data['labelLeft'],
                            feature_data['labelRight'],
                            0, 0, feature_data['feature_value'])
        else:

            # No feature to split on (None of the features improve entropy ~ IG = 0)
            return 0

    else:

        # No feature to split on (max depth reached)
        return 0


def DT_test_real(X, Y, DT):

    all_samples = numpy.arange(0, len(X))
    number_correct = test_real_helper(all_samples, Y, DT, X)
    return number_correct / len(Y)


def test_real_helper(samples, labels, node, all_samples):

    left_samples = numpy.array([])
    right_samples = numpy.array([])
    left_correct = 0
    right_correct = 0

    for sample in samples:
        if all_samples[int(sample)][int(node.featureIndex)] < node.featureValue:
            left_samples = numpy.append(left_samples, sample)
        else:
            right_samples = numpy.append(right_samples, sample)

    if node.children['left'] != 0:
        left_correct = test_real_helper(left_samples, labels, node.children['left'], all_samples)
    else:
        for sample in left_samples:
            if labels[int(sample)] == node.leftLabel:
                left_correct += 1

    if node.children['right'] != 0:
        right_correct = test_real_helper(right_samples, labels, node.children['right'], all_samples)
    else:
        for sample in right_samples:
            if labels[int(sample)] == node.rightLabel:
                right_correct += 1
    return right_correct + left_correct


def DT_train_real_best(X_train, Y_train, X_val, Y_val):
    i = 3
    forest = numpy.array([])
    forest = numpy.append(forest, DT_train_real(X_train, Y_train, 1))
    forest = numpy.append(forest, DT_train_real(X_train, Y_train, 2))

    while forest[len(forest) - 1].height() != forest[len(forest) - 2].height():
        forest = numpy.append(forest, DT_train_real(X_train, Y_train, i))
        i += 1

    accuracy = numpy.array([])
    for trees in forest:
        accuracy = numpy.append(accuracy, DT_test_real(X_val, Y_val, trees))

    return forest[numpy.argmax(accuracy)]
