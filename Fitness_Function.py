from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def fitnessFunction(k, n, trainDataset, testDataset):
    model = SVC(kernel=k, C=1/n, gamma='scale')
    model.fit(trainDataset.iloc[:, :-1], trainDataset.iloc[:, -1])
    predictions = model.predict(testDataset.iloc[:, :-1])
    accuracy = accuracy_score(testDataset.iloc[:, -1], predictions)  # Calculate accuracy
    return accuracy