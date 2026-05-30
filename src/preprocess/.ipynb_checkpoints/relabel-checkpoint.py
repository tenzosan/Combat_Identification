from src.config import *
# Relabel 
def labelling(data):
    # Loop through the rows of the DataFrame
    for i in range(len(data)):
        # Check if the first condition matches for "Theta_2"
        if (np.argmax(data.iloc[i,:-1]) == 9) and (np.argmax(data.iloc[i, [0, 1, 9]]) == 2):  # Adjusted the logical operator
            data.iloc[i, -1] = 'Θ2'
        
        # Check if the second condition matches for "Theta_1"
        if (np.argmax(data.iloc[i,:-1]) == 8) and (np.argmax(data.iloc[i, [2, 4, 8]]) == 2):  # Adjusted the logical operator
            data.iloc[i, -1] = 'Θ1'
    
    return data