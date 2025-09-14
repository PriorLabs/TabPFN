import pandas as pd
from tabpfn import TabPFNClassifier
from sklearn.model_selection import train_test_split

data = {
    'feature1': ['a', 'b', pd.NA, 'd'],
    'feature2': [1, 2, 3, 4],
    'target': [0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['feature1', 'feature2']]
y = df['target']

# Fix NA handling using .loc to avoid SettingWithCopyWarning
X.loc[:, 'feature1'] = X['feature1'].fillna('missing').astype(str)
X.loc[:, 'feature2'] = X['feature2'].astype(float)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = TabPFNClassifier(device='cpu')
clf.fit(X_train, y_train)

print("Test passed! Model trained without NA errors.")
