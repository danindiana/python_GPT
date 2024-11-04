from data_processing import load_data, preprocess_data, list_csv_files
from model_building import build_model
from training_model import train_model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    # Ensure that both y_test and predictions have the same shape
    if y_test.ndim == 2:
        y_test = y_test[:, 0]
    
    if predictions.ndim == 2:
        predictions = predictions[:, 0]
    
    # Evaluate the model
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        model = build_model(input_shape)
        
        # Train the model
        train_model(model, X, y)
        
        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)

def main():
    directory = '.'  # Default directory is the current working directory
    file_path = list_csv_files(directory)
    
    if file_path:
        df = load_data(file_path)
        X, y, scaler = preprocess_data(df)
        
        input_shape = (X.shape[1], X.shape[2])
        model = build_model(input_shape)
        
        # Train the model
        train_model(model, X, y)
        
        # Split data into training and testing sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Evaluate the model
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
