from scipy.optimize import minimize
from src.postprocess.intent_belief_fusion import *

# Define the optimization objective function
def objective(x, data, test_label):
    """
    Objective function that computes the accuracy for a given threshold x.
    """
    # Use the provided threshold x to combine the predictions
    combined_mass_functions = combine_predictions_with_threshold(data, x)
    
    # Get predicted class labels from combined mass functions
    predicted_indices = np.argmax(combined_mass_functions.values, axis=1)
    column = ['Hostile', 'NonHostile', 'Suspicious']
    predicted_labels = [column[i] for i in predicted_indices]

    # Calculate the accuracy
    test_accuracy = accuracy_score(test_label, predicted_labels)
    
    # Return the negative accuracy as we want to maximize the accuracy (minimization problem)
    return -test_accuracy


def combine_predictions_with_threshold(data, x, batch_size=20):
    """
    Combine predictions in batches of a given size, using the threshold x for Jousselme distance.
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
                if d <= x:  # Use the threshold x for comparison
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