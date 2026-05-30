import pandas as pd
def SFE_dataset(train_slices, test_slices, val_slices):
    
    '''
    Extract Statistical Feature from train, test, validation set
    '''
    
    def process_stat_dataset(df):
        X = df.iloc[:, :-1].values
        Y = df.iloc[:, -1].values
       
        # Parameters
        m = 50  # Number of timesteps
        k = 9   # Number of features
        n = len(X)  # Number of samples
       
        # Reshape X into (n, k, m)
        X = X.reshape(n, k, m)
        df_features = pd.DataFrame()
       
        for i in range(k):
            x = X[:, i, :]
           
            # Basic Descriptive Statistics
            mean = np.mean(x, axis=1)
            std = np.std(x, axis=1)
            median = np.median(x, axis=1)
            maximum = np.max(x, axis=1)
            minimum = np.min(x, axis=1)
            rng = maximum - minimum
            iqr = np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)
            q1 = np.percentile(x, 25, axis=1)
            q3 = np.percentile(x, 75, axis=1)
            kt = np.nan_to_num(kurtosis(x, axis=1), nan=0)
            sk = np.nan_to_num(skew(x, axis=1), nan=0)
         

            # Construct a DataFrame for the extracted features
            df_i = pd.DataFrame({
                'Mean': mean,
                'Standard deviation': std,
                'Median': median,
                'Maximum value': maximum,
                'Minimum value': minimum,
                'Range': rng,
                'Interquartile range': iqr,
                'Interquartile first quarter (Q1)': q1,
                'Interquartile third quarter (Q3)': q3,
                'Kurtosis': kt,
                'Skewness': sk,
            })
           
            # Concatenate this DataFrame with the previous ones
            df_features = pd.concat([df_features, df_i], axis=1)
       
        # Concatenate df_features with Y (the labels)
        result = pd.concat([df_features, pd.DataFrame(Y, columns=['Label'])], axis=1)
        return result
   
    # Convert to DataFrames
    train_df = pd.DataFrame(train_slices)
    test_df = pd.DataFrame(test_slices)
    val_df = pd.DataFrame(val_slices)

    # Process each dataset
    train_stat = process_stat_dataset(train_df)
    test_stat = process_stat_dataset(test_df)
    val_stat = process_stat_dataset(val_df)
   
    return train_stat, test_stat, val_stat