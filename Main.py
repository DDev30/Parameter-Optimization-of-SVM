import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from Fitness_Function import fitnessFunction

data = datasets.load_iris()  
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

bestAccuracy = 0  
bestKernel = ""
bestNu = 0
iterations = 100
kernelList = ['rbf', 'poly', 'linear', 'sigmoid']

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

trainDataset = pd.DataFrame(X_train)
trainDataset['target'] = y_train

testDataset = pd.DataFrame(X_test)
testDataset['target'] = y_test

convergence_data = []

for i in range(iterations):
    k = random.choice(kernelList)  
    n = random.uniform(0.01, 1)  
    
    accuracy = fitnessFunction(k, n, trainDataset, testDataset)
    
    if accuracy > bestAccuracy:
        bestAccuracy = accuracy  
        bestKernel = k
        bestNu = n
        
    convergence_data.append((i, bestAccuracy, bestKernel, bestNu))

iterations, fitness = zip(*[(x[0], x[1]) for x in convergence_data])  
plt.plot(iterations, fitness)
plt.xlabel('Iteration')
plt.ylabel('Best Accuracy')
plt.title('Convergence Graph')
plt.grid(True)
plt.savefig('convergence_graph.png')
plt.close()

results_df = pd.DataFrame(convergence_data, columns=['Iteration', 'Best Accuracy', 'Best Kernel', 'Best Nu'])
results_df.to_csv('SVM_Convergence_Data.csv', index=False)

print(f'Best Accuracy: {bestAccuracy:.4f}')
print(f'Best SVM Parameters: Kernel={bestKernel}, Nu={bestNu:.4f}')