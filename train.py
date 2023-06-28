import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Obtenez l'objet Run actuel
run = Run.get_context()

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from joblib import dump
from sklearn.metrics import r2_score

from azureml.core import Workspace, Dataset, Run, Model


# Workspace details
subscription_id = '111aaa69-41b9-4dfd-b6af-2ada039dd1ae'
resource_group = 'RG_LEBIHAN_2'
workspace_name = 'car_price_estimation_2'

# workspace object
ws = Workspace(subscription_id, resource_group, workspace_name)

# Retrieve the dataset by name
dataset = Dataset.get_by_name(workspace=ws, name='car_price_estimation_clean')

df = dataset.to_pandas_dataframe()

# converting symboling to categorical
df['symboling'] = df['symboling'].astype('object')

#Fuel economy
df['fueleconomy'] = (0.55 * df['citympg']) + (0.45 * df['highwaympg'])

# split into X and y
X = df.loc[:, ['symboling', 'fueltype', 'aspiration', 'doornumber',
       'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength',
       'carwidth', 'curbweight', 'enginetype', 'cylindernumber',
       'enginesize', 'fuelsystem', 'boreratio',
       'horsepower', 'fueleconomy',
       'car_company']]

y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)

#Pipeline de préprocessing pour les variables numériques
numeric_features = ['wheelbase', 'carlength', 'carwidth', 'curbweight',
                    'enginesize', 'boreratio', 'horsepower','fueleconomy']

numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('stdscaler', MinMaxScaler()),  # moyenne nulle et écart type = 1 -> Reg, SVM, PCA
        ])

# Pipeline de pre-processing pour les variables catégorielles
categorial_features = [ "symboling", "fueltype", "aspiration", "doornumber", "carbody", 'drivewheel',
                       "enginelocation", "enginetype",	"cylindernumber", "fuelsystem", "car_company"]

categorical_transformer = OneHotEncoder(sparse=True, handle_unknown='ignore')

# a l'aide de la classe ColumnTransformer, 
# on déclare à quelles variables on applique quel transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorial_features)
    ], remainder="passthrough"
)


# Créer un pipeline qui combine le preprocessor, le transformateur polynomial et le modèle à entraîner
pipe_ridge_poly = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('poly', PolynomialFeatures(degree=2)),
                                  ('model', Ridge())])

# Définir une grille de paramètres pour la recherche des hyperparamètres
param_grid = {
    'poly__degree': [2],
    'model__alpha': [0.1, 1, 10, 100]
}

# Effectuer une recherche sur la grille des hyperparamètres
grid_search = GridSearchCV(pipe_ridge_poly, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Obtenir les meilleurs hyperparamètres et le score R2 associé
best_params_ridge_poly = grid_search.best_params_
best_score_ridge_poly = grid_search.best_score_

# On obtient un pipeline de preprocessing et de transformation polynomial qu'on peut utiliser dans un pipeline d'entraînement
ridge_poly = Ridge(alpha=best_params_ridge_poly['model__alpha'])

pipe = Pipeline([
     ('prep', preprocessor),
     ('poly', PolynomialFeatures(degree=2)),
     ('model', ridge_poly)
])

trained_model = pipe.fit(X_train,y_train)
score = trained_model.score(X_test, y_test)
y_pred = trained_model.predict(X_test)
r2 = r2_score(y_test, y_pred)


dump(pipe,'model.joblib')

run = Run.get_context()
run.log("R-squared", r2)

# Chemin du fichier modèle
model_path = 'model.joblib'

# Nom de la sortie pour le modèle
output_model_name = 'car_price_prediction_ridge'

# Enregistrez le modèle avec la sortie nommée
registered_model = Model.register(workspace=ws,
                                 model_path=model_path,
                                 model_name=output_model_name)

run.log("Score", score)
run.complete()

print(score)