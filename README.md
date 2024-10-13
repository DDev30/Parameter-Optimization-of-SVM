
---

# SVM Optimization Project

This project implements an SVM (Support Vector Machine) model optimization process using a given dataset. The goal is to find the best SVM parameters for classification tasks.

## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
- [Sample Input](#sample-input)
- [Sample Output](#sample-output)
- [License](#license)

## Requirements

Before running the project, ensure you have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage

1. Place the `main.py` and `Fitness_Function.py` files in the same directory.
2. You can use the built-in Iris dataset from `sklearn` as it is included in the code, or modify the code to load your own dataset.

### Running the Program

Open your command line interface and navigate to the directory containing the files. Then run the following command:

```bash
python main.py
```

This will execute the program using the Iris dataset and output the results to a CSV file named `SVM_Convergence_Data.csv` and save the convergence graph as `convergence_graph.png`.

## Sample Input

The code uses the Iris dataset as a default input. If you want to use your dataset, you can replace the dataset loading section in `main.py` with the code to load your CSV file:

```python
df = pd.read_csv('your_dataset.csv')  # Replace with your dataset file
```

Ensure your dataset has a target column named `target`.

## Sample Output

Upon successful execution, the program will generate:
- **CSV File**: `SVM_Convergence_Data.csv` containing the following columns:
    - **Iteration**: The iteration number.
    - **Best Accuracy**: The best accuracy achieved so far.
    - **Best Kernel**: The kernel that achieved the best accuracy.
    - **Best Nu**: The value of Nu that achieved the best accuracy.

- **Graph**: `convergence_graph.png` showing the convergence of best accuracy over iterations.

### Example of CSV Output

```
Iteration, Best Accuracy, Best Kernel, Best Nu
0, 0.90, rbf, 0.45
1, 0.92, poly, 0.30
...
```

### Example of Graph

![Convergence Graph](convergence_graph.png)

## License

This project is licensed under the MIT License.

---
