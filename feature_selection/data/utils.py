from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path

def get_data():
    filename = Path('../data/data.csv').resolve()
    df = pd.read_csv(filename)

    y_start = df['charges']
    X_start = df.loc[:, df.columns != 'charges']

    X, y, X_test, y_test = train_test_split(X_start, y_start, random_state=42, train_size=.85, stratify=X_start['smoker'], shuffle=True)

    X.sex = X.sex.astype('category')
    X.smoker = X.smoker.astype('category')
    X.is_southwest = X.is_southwest.astype('category')
    X.is_southeast = X.is_southeast.astype('category')
    X.is_northwest = X.is_northwest.astype('category')
    X.is_northeast = X.is_northeast.astype('category')

    return X, y, X_test, y_test
