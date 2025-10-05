import argparse
import json
import joblib
import numpy as np
import pandas as pd

# Load the trained model and preprocessing objects
model = joblib.load('foraging_habitat_model_intensive.pkl')
scaler = joblib.load('scaler_intensive.pkl')
poly = joblib.load('poly_features.pkl')

def main():
    parser = argparse.ArgumentParser(description='Predict shark foraging habitat probabilities.')
    parser.add_argument('--data', type=str, required=True, help='JSON string containing the input data. Should be a list of dictionaries with keys: nflh, chl, a_412, bb_412, pic, kd_490, rrs_412, sst')
    
    args = parser.parse_args()
    
    try:
        # Parse the JSON data
        input_data = json.loads(args.data)
        
        # Convert input data to DataFrame
        features_df = pd.DataFrame(input_data)
        
        # Ensure required columns are present
        required_columns = ['nflh', 'chl', 'a_412', 'bb_412', 'pic', 'kd_490', 'rrs_412', 'sst']
        for col in required_columns:
            if col not in features_df.columns:
                raise ValueError(f'Missing required column: {col}')
        
        # Apply polynomial features
        X_poly = poly.transform(features_df[required_columns])
        
        # Pad X_poly to match scaler input (if necessary)
        if X_poly.shape[1] < scaler.n_features_in_:
            padding = np.zeros((X_poly.shape[0], scaler.n_features_in_ - X_poly.shape[1]))
            X_poly_padded = np.hstack([X_poly, padding])
        else:
            X_poly_padded = X_poly
        
        # Apply scaling
        X_scaled = scaler.transform(X_poly_padded)
        
        # Predict probabilities
        probabilities = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class (foraging)
        
        # Print the results
        print(json.dumps({'probabilities': probabilities.tolist()}))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))

if __name__ == '__main__':
    main()