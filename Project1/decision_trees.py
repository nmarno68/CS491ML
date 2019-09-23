import numpy


class Node:
    def __init__(self, index, llabel, rlabel, lchild, rchild):
        self.featureIndex = index
        self.leftLabel = llabel
        self.rightLabel = rlabel
        self.children = {'left': lchild, 'right': rchild}

    def printNode(self):
        if self.featureIndex != -1:
            if self.children['left'] != 0:
                print(self.children['left'].featureIndex)
                self.children['left'].printNode()
            if self.children['right'] != 0:
                print(self.children['right'].featureIndex)
                self.children['right'].printNode()


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
            left_samples = numpy.array([])
            right_samples = numpy.array([])
            feature_data = {'information_gain': 0}
            set_entropy = 0

            for feature in features:
                feature_true_label_true = 0
                feature_false_label_true = 0
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
                    return Node(feature_data['feature_index'], feature_data['label_left'], feature_data['label_right'],
                                self.recursive_train_binary(feature_data['left_samples'], features, depth - 1),
                                self.recursive_train_binary(feature_data['right_samples'], features, depth - 1))
                else:
                    return Node(feature_data['feature_index'], feature_data['label_left'], feature_data['label_right'],
                                0, 0)
            else:
                # No feature to split on (None of the features improve entropy ~ IG = 0)
                return Node(-1, 0, 0, 0, 0)

        else:
            # No feature to split on (max depth reached or no more features)
            return Node(-1, 0, 0, 0, 0)


def DT_train_binary(X, Y, max_depth):
    trainer = TrainBinary()
    trainer.featureData = X
    trainer.labelData = Y
    available_features = numpy.arange(0, len(X[0]))
    available_samples = numpy.arange(0, len(X))

    return trainer.recursive_train_binary(available_samples, available_features, max_depth)


def DT_test_binary(X, Y, DT):
    print("I am test binary")


if __name__ == "__main__":
    x = numpy.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
    y = numpy.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
    tree = DT_train_binary(x, y, -1)
    print(tree.featureIndex)
    tree.printNode()
