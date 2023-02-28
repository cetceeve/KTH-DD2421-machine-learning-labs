import random
import matplotlib.pyplot as plt
import monkdata as m
import dtree as t
from drawtree_qt5 import drawTree

## Assignment 0
## Which dataset is most difficult to learn
txt = """
One property that will be hard to learn is that some of the properties
have no influence on the result. Also the noise will cause trouble.

Monk 1 should be easiest to learn because there should be a clear winner
in terms of which attribute results in the highest information gain.

Monk 2 will be very hard to learn, because no single attribute
contributes to a class. It is always to attributes together which the
decision tree cannot capture because it is always looking at just one attribute.

Monk 3 will also be hard to learn because of the noise. A decision tree
is very sensitive to noise, because a wrong decision in the tree
influences all following nodes.
"""
print(txt)

## Assignment 1
print("\nAssingment 1:")
txt = "Entropy for dataset {dataset} is: {entropy}"
print(txt.format(dataset = "monk 1", entropy = t.entropy(m.monk1)))
print(txt.format(dataset = "monk 2", entropy = t.entropy(m.monk2)))
print(txt.format(dataset = "monk 3", entropy = t.entropy(m.monk3)))

## Assignment 2
print("\nAssingment 2:")
txt = """
In a uniform distributions all outcomes are equally likely. 
An example would be rolling a dice or throwing a coin.
The entropy of a uniform distribution is always log2(P).
This is also the upper bound for entropy on a dataset.
Monk 1 has entropy of 1 which meands there is a uniform distribution.

A non-uniform distribution means that some labels are more likely
to be seen in the dataset than othe labels. This sonstitutes
an information gain and therefore the entropy must be lower then
on a uniformly distributed dataset.
An example with low entropy is a heavily skewed dice.
"""
print(txt)

## Assigment 3
def getInfoGain(dataset):
    return [round(t.averageGain(dataset, attr), 3) for attr in m.attributes]
txt = "Use attribute a{x} for splitting at the root node"

print("\nAssignment 3:")
print("Information gain for dataset monk 1:")
print(getInfoGain(m.monk1))
print(txt.format(x = 5))
print("\nInformation gain for dataset monk 2:")
print(getInfoGain(m.monk2))
print(txt.format(x = 5))
print("\nInformation gain for dataset monk 3:")
print(getInfoGain(m.monk3))
print(txt.format(x = 2))

## Assignment 4
print("\nAssignment 4:")
txt = """
Information gain is maximised when the Entropy is lowest.
Choosing the attribute with the highest information gain
therefore equals choosing the attribute which will result
in the greatest reduction of entropy.

Entropy is essentially a measure of uncertainty.
Reducing entropy means reducing our uncertainty about the data.
When building a classification tree our final goals is to have one
class in every leaf of the tree. We want to be certain about which
class to put in each leaf. Therefore reducing uncertainty in
every split brings us closer to that goal.
"""
print(txt)

## Assingment 5
print("\nAssignment 5:")

def testTree(dataset):
    tree = t.buildTree(dataset, m.attributes)
    return (1.0 - t.check(tree, m.monk1), 1.0 - t.check(tree, m.monk1test))

print("Errors for dataset {dataset} is: {errors}".format(dataset="monk 1", errors=testTree(m.monk1)))
print("Errors for dataset {dataset} is: {errors}".format(dataset="monk 2", errors=testTree(m.monk2)))
print("Errors for dataset {dataset} is: {errors}".format(dataset="monk 3", errors=testTree(m.monk3)))

## Assignment 6
print("\nAssignment 6:")
txt = """
Pruning is a method to aviod overfitting, which trees are prone to do.
The idea is to build up a large tree and then remove leafes 
that do not significantly contribute to the overall outcome.

A small tree has a lower amount of splits. This is equivalent to
a lower variance. Of course this decrease in flexibility introduces
a smaller amount of bias. This trade-off is very likely worth it.
"""
print(txt)

## Assignment 7
print("\nAssignment 7:")

def partition(dataset, fraction):
    ldata = list(dataset)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruneTree(valSet, tree, error = 1.0):
    alternatives = t.allPruned(tree)
    validationErrors = {tree:1.0 - t.check(tree, valSet) for tree in alternatives}
    # select best alternative
    bestTree = min(validationErrors, key=validationErrors.get)
    newError = validationErrors.get(bestTree)
    if newError >= error:
        return tree
    else:
        return pruneTree(valSet, bestTree, newError)
    

def createPrunedTree(dataset, fraction):
    trainSet, valSet = partition(dataset, fraction)
    tree = t.buildTree(trainSet, m.attributes)
    pruned = pruneTree(valSet, tree)
    return pruned

def testPrunedTree(dataset, fraction, testset):
    tree = createPrunedTree(dataset, fraction)
    return round(1.0 - t.check(tree, testset), 3)
    
def testFraction(dataset, fraction, testset, repetitions = 100):
    return [testPrunedTree(dataset, fraction, testset) for _ in range(repetitions)] 

def allFractions(dataset, testset, fractions):
    return [testFraction(dataset, fraction, testset) for fraction in fractions]

def plot(results, name, xLabels):
    # x-axis labels
    fig, ax = plt.subplots()
    ax.set_xlabel("Fractions", fontweight ='bold')
    ax.set_ylabel("Classification Error", fontweight ='bold')
    
    # Adding title
    plt.title("Pruning results for test set " + name)
    bp_dict = plt.boxplot(results, labels=xLabels, showmeans=True)

    for line in bp_dict['means']:
        # get position data for median line
        x, y = line.get_xydata()[0] # top of median line
        # overlay median value
        plt.text(x + 0.01, y, str(round(y, 3))) # draw above, centered

        
    # show plot
    plt.show()


fractions = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
print("Plotting...")
print("Pruning siginificantly improved the error for the monk 3 graph!!!")
plot(allFractions(m.monk1, m.monk1test, fractions), "monk 1", fractions)
plot(allFractions(m.monk3, m.monk3test, fractions), "monk 3", fractions)

## pruning flow (fraction)
# partition into training and validation
# build tree on training set
# use allPruned for first pruning step
# test each alternative on the validation set
# select best alternative
# repeat pruning procedure until no improvment is made -> return pruned tree

# test flow
# compute tree error on test set

## fraction flow
# repeat test flow for same fraction X (15) times
# store results in np.array
# compute average error for tree
# compute variance for error
# return raw data, average and variance

## test flow
# test different fractions for pruning of tree with fraction flow
# return measurements in a list

## dataset flow
# perform test flow for monk1 and monk3 dataset

# plot flow
# take measurements and plot results


