import numpy


class Node:
    def __init__(self, index, llabel, rlabel, lchild, rchild):
        self.featureIndex = index
        self.leftLabel = llabel
        self.rightLabel = rlabel
        self.children = {'left': lchild, 'right': rchild}

    def printNode(self):
        print(self.featureIndex)
        print(self.children['left'])
        print(self.children['right'])


class TrainBinary:

    def __init__(self):
        self.featureData = 0
        self.labelData = 0
        self.trainingTree = 0

    def recursive_train_binary(self, samples, features, depth):
        total_samples = len(samples)
        left_samples = numpy.array([])
        right_samples = numpy.array([])
        feature_true_label_true = 0
        feature_false_label_true = 0

        print(self.featureData)

        for feature in features:

            # Looking at feature for each sample, collect data
            for sample in samples:
                if self.featureData[sample][feature]:

                    # Track samples on right of split
                    numpy.append(right_samples, sample)
                    if self.labelData[sample][0]:

                        # Count number of true labels on right of split
                        feature_true_label_true += 1
                else:

                    # Track samples on left of split
                    numpy.append(left_samples, sample)
                    if self.labelData[sample][0]:

                        # Count number of true labels on right of split
                        feature_false_label_true += 1

            # Calculate entropy on the left side of the split on feature

            # Calculate entropy on the right side of the split on feature

            # Calculate entropy of whole set

            # Calculate information gain of feature and save all necessary information of feature

        # Determine best feature to split on

        # Create new tree node and call function on left and right samples (if IG != H())

        return Node(0, 0, 0, 0, 0)


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

    x = numpy.array([[0, 1], [0, 0], [1, 0], [0, 0],  [1, 1]])
    y = numpy.array([[1], [0], [0], [0], [1]])
    DT_train_binary(x, y, -1).printNode()
