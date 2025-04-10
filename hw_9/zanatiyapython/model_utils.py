import pandas as pd
import joblib
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV

def load_data():
    df = pd.read_csv("realty_data.csv")
    df['rooms'] = df['rooms'].fillna(df['rooms'].median())
    df['city'] = df['city'].fillna('unknown')
    df['district'] = df['district'].fillna('unknown')
    df['lat'] = df['lat'].fillna(df['lat'].mean())
    df['lon'] = df['lon'].fillna(df['lon'].mean())
    return df[['total_square', 'rooms', 'floor', 'lat', 'lon', 'city', 'district', 'price']].dropna()

def train_and_save_model(filepath="model.pkl"):
    df = load_data()
    X = df.drop(columns=['price'])
    y = df['price']
    cat_features = ['city', 'district']
    cat_feature_indices = [X.columns.get_loc(col) for col in cat_features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = CatBoostRegressor(verbose=0, random_state=42)
    param_grid = {
        'depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [300, 500, 700],
        'l2_leaf_reg': [1, 3, 5, 7]
    }

    search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='neg_mean_absolute_error',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train, cat_features=cat_feature_indices)
    best_model = search.best_estimator_
    joblib.dump(best_model, filepath)
    print(f"✅ Модель сохранена в {filepath}")

def load_model(filepath="model.pkl"):
    return joblib.load(filepath)