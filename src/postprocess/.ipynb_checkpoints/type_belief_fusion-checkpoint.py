from src.config import *

# Belief Mapping
def belief_mapping_type(SVMs_prob_distribution, SVMs_Accuracy, FFNNs_prob_distribution, FFNNs_Accuracy, LSTMs_prob_distribution, LSTMs_Accuracy):
    def mapping(Accuracy, Distribution):
        n_rows, n_cols = Distribution.shape
        L = np.zeros((n_rows, n_cols + 1))
        
        # Populate the L matrix
        L[:, :-1] = Distribution * Accuracy
        L[:, -1] = 1 - Accuracy
        # Initialize an array to store the results
        results = np.zeros((n_rows, n_cols + 3))
        
        # Loop through each row in the L matrix
        for i in range(n_rows):
            x = L[i, :]
            
            # Define Theta_1 and Theta_2 based on current row
            # 2 : Civilian Helicopter
            # 3 : Military Helicopter
            # 0 : Bomber
            # 1 : Civilian Commercial Airliner 

            
            FOD_1 = np.array([x[2], x[4]])  
            FOD_2 = np.array([x[0], x[1]])
            
            # Calculate alpha values
            if np.sum(FOD_1) != 0:
                alpha1 = np.abs(x[2] - x[4]) / np.sum(FOD_1)
            else:
                alpha1 = 0
                
            
            if np.sum(FOD_2) != 0:
                alpha2 = np.abs(x[0] - x[1]) / np.sum(FOD_2)
            else:
                alpha2 = 0
            
            # Adjust values in x based on alpha values
            x[2] = x[2] * alpha1
            x[4] = x[4] * alpha1
            x[0] = x[0] * alpha2
            x[1] = x[1] * alpha2
            
            # Create the result row with adjusted values and the new sums
            A = x[:-1]  # All elements except the last two
            B = np.array([(1 - alpha1) * np.sum(FOD_1)])  # Single element array
            C = np.array([(1 - alpha2) * np.sum(FOD_2)])  # Single element array
            D = np.array([x[-1]])  # Single element array
            y = np.concatenate([A, B, C, D])
            
            # Store the result in the results array
            results[i, :] = y
        
        return results
    
    # Convert to DataFrame
    SVMs_prob_distribution = pd.DataFrame(SVMs_prob_distribution)
    FFNNs_prob_distribution = pd.DataFrame(FFNNs_prob_distribution)
    LSTMs_prob_distribution = pd.DataFrame(LSTMs_prob_distribution)
   
    # Process each dataset
    Belief_SVMs = mapping(SVMs_Accuracy, SVMs_prob_distribution)
    Belief_FFNNs = mapping(FFNNs_Accuracy, FFNNs_prob_distribution)
    Belief_LSTMs = mapping(LSTMs_Accuracy, LSTMs_prob_distribution)
    
    return Belief_SVMs, Belief_FFNNs, Belief_LSTMs
    
# --------------------------- FFNN + LSTM Combination ---------------------------- # 
def fuse_ffnn_lstm(FFNNs_Belief, LSTMs_Belief, batch_size=20):
    def combine_predictions(FFNNs_Belief, LSTMs_Belief):
        m_values = []
        for i in range(len(FFNNs_Belief)):
            m1 = FFNNs_Belief[i]
            m2 = LSTMs_Belief[i]
            length = len(m1)
            
            # Initialize m with the combined belief values
            m = [m1[j] * m2[j] + m1[j] * m2[length-1] + m1[length-1] * m2[j] for j in range(length)]
            
            # Update specific indices based on the given rules
            m[0] += m1[0] * m2[9] + m1[9] * m2[0]
            m[1] += m2[1] * m1[9] + m1[1] * m2[9]
            m[2] += m2[2] * m1[8] + m1[2] * m2[8]
            m[4] += m1[4] * m2[8] + m2[4] * m1[8]
            m[10] = m1[10] * m2[10]
            
            # Handle conflicts
            m[9] += (m1[0] * m2[1] + m1[1] * m2[0])
            m[8] += (m1[2] * m2[4] + m1[4] * m2[2])
            
            # Normalize and append to results
            m_values.append(np.array(m) / np.sum(m))
            if np.sum(m) == 0:
                print("Error")
        return np.array(m_values)

    # Process in batches and combine results
    all_combined_results = []
    for i in range(0, len(FFNNs_Belief), batch_size):
        FFNN_batch = FFNNs_Belief[i:i+batch_size]
        LSTM_batch = LSTMs_Belief[i:i+batch_size]
        combined_results = combine_predictions(FFNN_batch, LSTM_batch)
        all_combined_results.append(combined_results)

    # Return final concatenated results
    return np.concatenate(all_combined_results, axis=0)

# -------------------- (FFNNs + LSTMs) + SVMs Combination ------------------------------------------ #
def fuse_ffnn_lstm_svm(LSTM_FFNN_Combine, SVMs_belief, batch_size=20):
    def combine_predictions(LSTM_FFNN_Combine, SVMs_belief):
        m_values = []
        for i in range(len(LSTM_FFNN_Combine)):
            # Get prediction rows from each model
            m1 = LSTM_FFNN_Combine[i, :]
            m2 = SVMs_belief[i, :]
            
            # Number of columns in predictions
            length = len(m1)
            m = np.zeros(length)
            
            # Modified Yager Rule of Combination
            for j in range(length):
                temp = m1[j] * m2[j] + m1[j] * m2[length-1] + m1[length-1] * m2[j]
                m[j] = temp
            
            # Specific updates to the combined result
            if length > 11:
                m[0] += m1[0] * m2[9] + m1[9] * m2[0]
                m[1] += m2[1] * m1[9] + m1[1] * m2[9]
                m[2] += m2[2] * m1[8] + m1[2] * m2[8]
                m[4] += m1[4] * m2[8] + m2[4] * m1[8]
                m[10] = m1[10] * m2[10]
            
                # Handle conflicts and update combined result
                conflict_1 = (m1[0] * m2[1] + m1[1] * m2[0])
                m[9] += conflict_1
                
                conflict_2 = m1[2] * m2[4] + m1[4] * m2[2]
                m[8] += conflict_2
            
            # Normalize the belief
            sum_m = np.sum(m)
            if sum_m != 0:
                m /= sum_m
            else:
                m = np.ones(length) / length  # Avoid division by zero by evenly distributing if all zero
            
            m_values.append(m)
        
        return np.array(m_values)
    
    # Initialize an empty list to store all combined results
    all_combined_results = []

    # Perform combination for each batch of rows
    for i in range(0, len(LSTM_FFNN_Combine), batch_size):
        LSTM_FFNN_batch = LSTM_FFNN_Combine[i:i+batch_size]
        SVM_batch = SVMs_belief[i:i+batch_size]
        
        # Combine predictions
        combined_results = combine_predictions(LSTM_FFNN_batch, SVM_batch)
        
        # Append the combined results to the list
        all_combined_results.append(combined_results)

    # Concatenate all the combined results into a single array
    final_results = np.concatenate(all_combined_results, axis=0)
    
    return final_results

# ------------------------ Combination Across Time ------------------------
def temporal_fusion_type(data,batch_size=20):
    combined_results = []
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        m1 = None
        for j in range(start, end):
            m2 = data[j] 
            if m1 is None:
                m1 = m2
            else:
                m = []
                length = len(m2)
                for k in range(length):
                    temp = m1[k] * m2[k] + m1[k] * m2[length-1] + m1[length-1] * m2[k]
                    m.append(temp)
                # Specific updates to the combined result
                m[0] = m[0] + m1[0] * m2[9] + m1[9] * m2[0]
                m[1] = m[1] + m2[1] * m1[9] + m1[1] * m2[9]
                m[2] = m[2] + m2[2] * m1[8] + m1[2] * m2[8]
                m[4] = m[4] + m1[4] * m2[8] + m2[4] * m1[8]
                m[10] = m1[10] * m2[10]
                # Handle conflicts and update combined result
                conflict_1 = m1[0] * m2[1] + m1[1] * m2[0] 
                m[9] = m[9] + conflict_1
                conflict_2 = m1[2] * m2[4] + m1[4] * m2[2]
                m[8] = m[8] + conflict_2
                # Normalize the belief
                m = np.array(m) / np.sum(m)
                m2 = m

            combined_results.append(m2)
    
    return combined_results