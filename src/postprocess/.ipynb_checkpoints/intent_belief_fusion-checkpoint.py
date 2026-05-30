from src.config import *
# Belief Mapping 
def belief_mapping_intent(Distribution,Accuracy):
        n_rows, n_cols = Distribution.shape
        L = np.zeros((n_rows, n_cols + 1))
        # Populate the L matrix
        L[:, :-1] = Distribution * Accuracy
        L[:, -1] = 1 - Accuracy
        results = L
        return results

# Jousselme Distance Calculation
def jousselme_distance(m1, m2):
    """
    Calculate Jousselme's distance between two mass functions.
    """
    D = np.array([
        [1, 0, 0, 0.333],
        [0, 1, 0, 0.333],
        [0, 0, 1, 0.333],
        [0.333, 0.333, 0.333, 1]
    ]) 
    diff = m1 - m2   
    # Compute Jousselme's distance
    quadratic_form = diff.T @ D @ diff
    distance = np.sqrt(0.5 * quadratic_form)
    return distance
    
# Define a function to combine predictions for a given range of rows
def temporal_fusion_intent(data,d, batch_size=20):
    """
    Combine predictions in batches of a given size, using Jousselme distance to handle transitions.
    """
    combined_results = []

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch_mass_function = None

        for j in range(start, end):
            m_temp = data.iloc[j].values

            if batch_mass_function is None:
                batch_mass_function = m_temp
            else:
                d = jousselme_distance(batch_mass_function, m_temp)
                # ------------------ optimal distance for maximum accuracy ------------------- 
                if d <= 0.736:  
                    m = []
                    length = len(m_temp)
                    for k in range(length - 1):
                        temp = (batch_mass_function[k] * m_temp[k] +
                                batch_mass_function[k] * m_temp[length - 1] +
                                batch_mass_function[length - 1] * m_temp[k])
                        m.append(temp)
                    m.append(batch_mass_function[length - 1] * m_temp[length - 1])

                    denom = np.sum(m)
                    m_normalized = np.array(m) / denom
                    batch_mass_function = m_normalized
                
                else:
            
                    batch_mass_function = m_temp
    

            combined_results.append(batch_mass_function)

    return pd.DataFrame(combined_results)

