import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from category_encoders import TargetEncoder
import featuretools as ft
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

def preprocess_data(X, categorical_threshold):
    def categorical_encod(uniq, categorical_threshold, column_data):
        encoding_threshold = int(0.4 * categorical_threshold)
        if uniq <= encoding_threshold:
            one_hot_encoder = OneHotEncoder()
            one_hot_encoded = one_hot_encoder.fit_transform(column_data)
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out())
            return one_hot_encoded_df
        else:
            frequency_map = column_data.iloc[:, 0].value_counts().to_dict()
            column_data_encoded = column_data.iloc[:, 0].map(frequency_map)
            return pd.DataFrame(column_data_encoded)

    # Encoding X
    columns = X.columns
    for column in columns:
        uniq = X[column].nunique()
        if uniq <= categorical_threshold:
            df = categorical_encod(uniq, categorical_threshold, X[[column]])
            X = X.drop(column, axis=1)
            X = pd.concat([X, df], axis=1)
        elif uniq > categorical_threshold and (X[column].dtype == int or X[column].dtype == float):
            scaler = StandardScaler()
            X[column] = scaler.fit_transform(X[[column]])
        elif X[column].dtype == object:
            tfidf_vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_features = tfidf_vectorizer.fit_transform(X[column])
            tfidf_df = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
            X = pd.concat([X.drop(columns=column), tfidf_df], axis=1)

    return X


def automated_feature_engineering(X, task):
    # Creating an EntitySet
    es = ft.EntitySet()

    # Defining an Entity using the DataFrame
    es = es.add_dataframe(dataframe_name="User_data_transformed", dataframe=X, index='index')

    if (task == "Regression"):
        features, feature_defs = ft.dfs(entityset=es, target_dataframe_name="User_data_transformed",
                                        agg_primitives=["count", "sum", "mean", "median", "std", "min", "max"],
                                        trans_primitives=["add_numeric", "subtract_numeric", "multiply_numeric",
                                                          "divide_numeric", "equal", "not_equal", "greater_than",
                                                          "less_than", "greater_than_equal_to", "less_than_equal_to",
                                                          "is_null", "cum_sum", "cum_min", "cum_max"])
    elif (len(X.columns) > 10 or len(X) > 150):
        features, feature_defs = ft.dfs(entityset=es, target_dataframe_name="User_data_transformed",
                                        agg_primitives=["count", "sum", "mean", "median", "std", "min", "max",
                                                        "percent_true", "num_unique", "mode"],
                                        trans_primitives=["absolute", "add_numeric", "subtract_numeric",
                                                          "multiply_numeric", "divide_numeric", "modulo_numeric",
                                                          "equal", "not_equal", "greater_than", "less_than",
                                                          "greater_than_equal_to", "less_than_equal_to", "and", "or",
                                                          "is_null", "cum_sum", "cum_min", "cum_max"])
    else:
        # Fetching aggregation and transformation primitives
        all_primitives_df = ft.primitives.list_primitives()
        aggregation_primitives = all_primitives_df[all_primitives_df['type'] == 'aggregation']
        transformation_primitives = all_primitives_df[all_primitives_df['type'] == 'transform']

        # Running Deep Feature Synthesis
        features, feature_defs = ft.dfs(entityset=es, target_dataframe_name="User_data_transformed",
                                        agg_primitives=aggregation_primitives['name'].tolist(),
                                        trans_primitives=transformation_primitives['name'].tolist())

    X = features
    return X

def perform_feature_selection(X, Y, task):
    # Drop columns with NaN values
    nan_columns = X.columns[X.isnull().any()]
    X = X.drop(nan_columns, axis=1)

    # Replace infinite values with a large finite value
    large_finite_value = 1e10
    X.replace([np.inf, -np.inf], large_finite_value, inplace=True)

    # Calculate correlations (NOT WORKING)
    # correlations = X.corrwith(Y)
    # print("Correlations with target variable:")
    # print(correlations)
    # print(Y)

    # Calculate mutual information
    if task == "Classification":
        mi_scores = mutual_info_classif(X, Y)

        # Creating a dataframe to display the scores
        mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Getting the indices of features with MI score greater than 0
        non_zero_mi_indices = [i for i, score in enumerate(mi_scores) if score > 0]

        # Creating a DataFrame with non-zero MI features
        X = X.iloc[:, non_zero_mi_indices]
        # print(mi_df, X)

    # Feature selection using Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X, Y)

    # Getting feature importances from the trained Random Forest model
    feature_importances_rf = rf_model.feature_importances_

    # Setting a threshold for feature importance
    threshold = 0.01  # You can adjust this threshold as needed

    # Selecting features with importance scores above the threshold
    selected_features_rf = X.columns[feature_importances_rf > threshold]

    # Reducing the dataframe X to include only the selected features
    X = X[selected_features_rf]

    # #printing the reduced dataframe
    #print("Reduced X:")
    #print(X)
    return(X)

# ML algorithms with hyperparameter tuning if task='Classification'
# Random Forest Classifier
def random_forest_classification(X_train, X_test, y_train, y_test):
    # Defining hyperparameter grid for Random Forest
    param_distributions = {
        'n_estimators': [i for i in range(50, 500, 50)],
        'max_depth': [None] + list(range(10, 50, 10)),
        'min_samples_split': [2, 3, 4]
    }

    # Initialize and fit the Random Forest model
    rf_classifier = RandomForestClassifier(random_state=42)

    # Performing Random Search with cross-validation
    random_search = RandomizedSearchCV(rf_classifier, param_distributions, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Getting best hyperparameters and model
    best_params = random_search.best_params_
    best_rf_model = random_search.best_estimator_

    # Make predictions using the best model
    y_pred_rf = best_rf_model.predict(X_test)

    # Evaluating accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    #print(f"Best Random Forest Accuracy (Random Search): {accuracy_rf}")

    return best_rf_model, y_pred_rf

# Gradient Boosting Classifier
def gradient_boosting_classification(X_train, X_test, y_train, y_test):
    # Defining hyperparameter distribution for Gradient Boosting
    param_distributions = {
        'n_estimators': [i for i in range(100, 251, 50)],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize and fit the Gradient Boosting model
    gb_classifier = GradientBoostingClassifier(random_state=42)

    # Performing Random Search with cross-validation
    random_search_gb = RandomizedSearchCV(gb_classifier, param_distributions, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    random_search_gb.fit(X_train, y_train)

    # Getting best hyperparameters and model
    best_params_gb = random_search_gb.best_params_
    best_gb_model = random_search_gb.best_estimator_

    # Making predictions using the best model
    y_pred_gb = best_gb_model.predict(X_test)

    # Evaluating accuracy
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    #print(f"Best Gradient Boosting Accuracy (Random Search): {accuracy_gb}")

    return best_gb_model, y_pred_gb

def logistic_regression(X_train, X_test, y_train, y_test):
    # Define the hyperparameter ranges
    param_dist = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs'],
        'multi_class': ['ovr', 'multinomial']
    }

    # Create a random search object with early stopping
    random_search = RandomizedSearchCV(
        LogisticRegression(),
        param_distributions=param_dist,
        n_iter=20,  # Maximum number of trials
        cv=3,       # Number of cross-validation folds
        scoring='accuracy',  # Use accuracy as the metric
        n_jobs=-1,   # Use all available CPU cores
        random_state=42
    )

    # Perform the random search
    random_search.fit(X_train, y_train)

    # Getting best hyperparameters and model
    best_params_logreg = random_search.best_params_
    best_logreg_model = random_search.best_estimator_

    # Making predictions using the best model
    y_pred_logreg = best_logreg_model.predict(X_test)

    # Evaluating accuracy
    accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
    #print("Accuracy Logistic Regression:", accuracy_logreg)

    return best_logreg_model, y_pred_logreg

# ML algorithms with hyperparameter tuning if task='Regression'
def random_forest_regression(X, Y):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Defining hyperparameter grid for Random Forest
    param_distributions = {
        'n_estimators': [i for i in range(50, 500, 50)],
        'max_depth': [None] + list(range(10, 50, 10)),
        'min_samples_split': [2, 3, 4]
    }

    # Initialize and fit the Random Forest model
    rf_regressor = RandomForestRegressor(random_state=42)

    # Performing Random Search with cross-validation
    random_search = RandomizedSearchCV(rf_regressor, param_distributions, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    random_search.fit(X_train, y_train)

    # Getting best hyperparameters and model
    best_params = random_search.best_params_
    best_rf_model = random_search.best_estimator_

    # Make predictions using the best model
    y_pred_rf = best_rf_model.predict(X_test)

    # Evaluating mean squared error
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    #print(f"Best Random Forest Mean Squared Error (Random Search): {mse_rf}")

    return best_rf_model, y_pred_rf, X_train, X_test, y_train, y_test


def gradient_boosting_regression(X_train, X_test, y_train, y_test):
    # Defining hyperparameter distribution for Gradient Boosting
    param_distributions = {
        'n_estimators': [i for i in range(50, 251, 50)],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9],
        'max_features': ['sqrt', 'log2']
    }

    # Initialize and fit the Gradient Boosting model
    gb_regressor = GradientBoostingRegressor(random_state=42)

    # Performing Random Search with cross-validation
    random_search_gb = RandomizedSearchCV(gb_regressor, param_distributions, n_iter=10, cv=5, random_state=42, n_jobs=-1)
    random_search_gb.fit(X_train, y_train)

    # Getting best hyperparameters and model
    best_params_gb = random_search_gb.best_params_
    best_gb_model = random_search_gb.best_estimator_

    # Making predictions using the best model
    y_pred_gb = best_gb_model.predict(X_test)

    # Evaluating mean squared error
    mse_gb = mean_squared_error(y_test, y_pred_gb)
    #print(f"Best Gradient Boosting Mean Squared Error (Random Search): {mse_gb}")

    return best_gb_model, y_pred_gb

def linear_regression(X_train, X_test, y_train, y_test):
    # Define the hyperparameter ranges
    param_dist = {
        'fit_intercept': [True, False],
    }

    # Create a random search object
    random_search = RandomizedSearchCV(
        LinearRegression(),
        param_distributions=param_dist,
        n_iter=10,   # Maximum number of trials
        cv=3,        # Number of cross-validation folds
        scoring='neg_mean_squared_error',  # Use negative mean squared error as the metric
        n_jobs=-1,   # Use all available CPU cores
        random_state=42
    )

    # Perform the random search
    random_search.fit(X_train, y_train)

    # Getting best hyperparameters and model
    best_params_linear = random_search.best_params_
    best_linear_model = random_search.best_estimator_

    # Making predictions using the best model
    y_pred_linear = best_linear_model.predict(X_test)

    # Evaluating mean squared error
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    #print("Mean Squared Error Linear Regression:", mse_linear)

    return best_linear_model, y_pred_linear

def select_best_model_classification(y_test, y_pred_rf, y_pred_gb, y_pred_logreg):
    metrics_rf = {
        "Accuracy": accuracy_score(y_test, y_pred_rf),
        "Precision": precision_score(y_test, y_pred_rf, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'),
        "Recall": recall_score(y_test, y_pred_rf, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'),
        "F1-Score": f1_score(y_test, y_pred_rf, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
    }

    metrics_gb = {
        "Accuracy": accuracy_score(y_test, y_pred_gb),
        "Precision": precision_score(y_test, y_pred_gb, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'),
        "Recall": recall_score(y_test, y_pred_gb, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'),
        "F1-Score": f1_score(y_test, y_pred_gb, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
    }

    metrics_logreg = {
        "Accuracy": accuracy_score(y_test, y_pred_logreg),
        "Precision": precision_score(y_test, y_pred_logreg, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'),
        "Recall": recall_score(y_test, y_pred_logreg, average='binary' if len(np.unique(y_test)) == 2 else 'weighted'),
        "F1-Score": f1_score(y_test, y_pred_logreg, average='binary' if len(np.unique(y_test)) == 2 else 'weighted')
    }

    # Compare metrics and select the best model
    best_model = None
    best_metrics = None
    for model_name, model_metrics in [("Random Forest", metrics_rf), ("Gradient Boosting", metrics_gb), ("Logistic Regression", metrics_logreg)]:
        if best_model is None or model_metrics["Accuracy"] > best_metrics["Accuracy"]:
            best_model = model_name
            best_metrics = model_metrics

    return best_model, best_metrics

def select_best_model_regression(y_test, y_pred_rf, y_pred_gb, y_pred_linear):
    metrics_rf = {
        "Mean Squared Error": mean_squared_error(y_test, y_pred_rf),
        "Root Mean Squared Error": mean_squared_error(y_test, y_pred_rf, squared=False),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred_rf),
        "R-squared": r2_score(y_test, y_pred_rf)
    }

    metrics_gb = {
        "Mean Squared Error": mean_squared_error(y_test, y_pred_gb),
        "Root Mean Squared Error": mean_squared_error(y_test, y_pred_gb, squared=False),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred_gb),
        "R-squared": r2_score(y_test, y_pred_gb)
    }

    metrics_lr = {
        "Mean Squared Error": mean_squared_error(y_test, y_pred_linear),
        "Root Mean Squared Error": mean_squared_error(y_test, y_pred_linear, squared=False),
        "Mean Absolute Error": mean_absolute_error(y_test, y_pred_linear),
        "R-squared": r2_score(y_test, y_pred_linear)
    }

    # Compare metrics and select the best model
    best_model = None
    best_metrics = None
    for model_name, model_metrics in [("Random Forest", metrics_rf), ("Gradient Boosting", metrics_gb), ("Linear Regression", metrics_lr)]:
        if best_model is None or model_metrics["R-squared"] > best_metrics["R-squared"]:
            best_model = model_name
            best_metrics = model_metrics

    return best_model, best_metrics


# Main Streamlit app code
def main():
    st.title("Automated Machine Learning Project")
    uploaded_file_original = st.file_uploader("Upload a CSV file for training", type=["csv"])
    uploaded_file_predictor = st.file_uploader("Upload a CSV file for predicting", type=["csv"])
    task = st.radio("Specify task", ["Classification", "Regression"])
    submit_button = st.button("Submit")
    if submit_button and uploaded_file_original is not None and uploaded_file_predictor is not None:
        df = pd.read_csv(uploaded_file_original)
        st.write("Uploaded training DataFrame:")
        st.write(df)
        # Preprocessing steps
        df = df.fillna(method='ffill')
        X = df.iloc[:, :-1]
        Y = df.iloc[:, -1]
        X_predict = pd.read_csv(uploaded_file_predictor)
        X_predictor = X_predict.fillna(method='ffill')

        categorical_threshold = Y.nunique()
        if(task == "Regression"):
            categorical_threshold = 10
        X = preprocess_data(X, categorical_threshold)
        X_predictor = preprocess_data(X_predictor, categorical_threshold)
        if(task=="Classification" and (Y.dtype != int and Y.dtype != float)):
            label_encoder = LabelEncoder()
            Y = label_encoder.fit_transform(Y)
            Y = pd.DataFrame(Y)
        st.write("Preprocessing done")
        st.write(X)
        st.write(X_predictor)
        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        y_train = y_train.values.ravel()
        X_train = automated_feature_engineering(X_train, task)
        X_test = automated_feature_engineering(X_test, task)
        X_predictor = automated_feature_engineering(X_predictor, task)
        st.write("Feature Engineering done")
        X_train = perform_feature_selection(X_train, y_train, task)

        Xtrain_columns=X_train.columns
        mask_test = X_test.columns.isin(Xtrain_columns)
        mask_predictor = X_predictor.columns.isin(Xtrain_columns)
        X_test = X_test.loc[:, mask_test]
        X_predictor = X_predictor.loc[:, mask_predictor]
        st.write("Feature Selection done")
        # Classification models
        if(task=="Classification"):
            best_rf_model, y_pred_rf = random_forest_classification(X_train, X_test, y_train, y_test)
            best_gb_model, y_pred_gb = gradient_boosting_classification(X_train, X_test, y_train, y_test)
            best_logreg_model, y_pred_logreg = logistic_regression(X_train, X_test, y_train, y_test)

            best_model, best_metrics = select_best_model_classification(y_test, y_pred_rf, y_pred_gb, y_pred_logreg)
        else:
            best_rf_model, y_pred_rf = random_forest_regression(X_train, X_test, y_train, y_test)
            best_gb_model, y_pred_gb = gradient_boosting_regression(X_train, X_test, y_train, y_test)
            best_linear_model, y_pred_linear = linear_regression(X_train, X_test, y_train, y_test)

            best_model, best_metrics = select_best_model_regression(y_test, y_pred_rf, y_pred_gb, y_pred_linear)

        st.write("Best Model:", best_model)
        st.write("Metrics:")
        for metric, value in best_metrics.items():
            st.write(f"{metric}: {value}")

        if best_model == 'Random Forest':
            model = best_rf_model
        elif best_model == 'Gradient Boosting':
            model = best_gb_model
        elif best_model == 'Logistic Regression':
            model = best_logreg_model
        elif best_model == 'Linear Regression':
            model = best_linear_model
        else:
            st.write("Unknown best model:", best_model)
            model = None
        if model is not None:
            predictions = model.predict(X_predictor)
            if (task == "Classification" and (predictions.dtype != int and predictions.dtype != float)) :
                #accuracy = accuracy_score(Y.iloc[-50:], predictions)
                predictions = label_encoder.inverse_transform(predictions)
                #st.write(f"Accuracy on last 50 samples: {accuracy:.2f}")
                st.write("Predictions:")
            else:
                #mse = mean_absolute_error(Y.iloc[-50:], predictions)
                #st.write(f"Mean Absolute Error on last 50 samples: {mse:.2f}")
                st.write("Predictions:")

            combined_data = pd.concat([X_predict.reset_index(drop=True), pd.DataFrame(predictions, columns=["Predictions"])], axis=1)
            styled_combined_data = combined_data.style.set_properties(subset=['Predictions'],**{'background-color': 'yellow'})
            st.dataframe(styled_combined_data, width=800, height=400)


if __name__ == "__main__":
    main()
