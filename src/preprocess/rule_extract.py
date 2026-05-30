from src.config import *

# --------------------------------- Feature Extraction (Rule Base) ---------------------------------------- #

def extract_rule(data_slices,k = 4,m = 50):
    data_slices = pd.DataFrame(data_slices)
    x = data_slices.iloc[:, :-1].values
    y = data_slices.iloc[:, -1].values

    # Parameters
    n = len(x)

    # Reshape data
    X = x.reshape(n, k, m)

    # Initialize result list
    results = []

    # Iterate over each sample
    for i in range(n):

        # Extract features
        dRange = X[i, 0, :]
        Attacker_deviation_Angle = X[i, 1, :]
        Aspect_Angle = X[i, 2, :]
        Off_angle = X[i, 3, :]

        f_epsilon = np.where(
            Aspect_Angle < 90,
            Aspect_Angle,
            180 - Aspect_Angle
        )

        # Rule 1
        rule1_satisfied = (
            Attacker_deviation_Angle < f_epsilon
        )

        # Rule 2
        rule2_satisfied = np.where(
            Aspect_Angle < 90,
            Off_angle < 90,
            rule1_satisfied
        )

        # Rule 3
        rule3_satisfied = dRange < 0

        # Fractions
        rule1_fraction = np.mean(rule1_satisfied)
        rule2_fraction = np.mean(rule2_satisfied)
        rule3_fraction = np.mean(rule3_satisfied)

        # Store results
        results.append({
            'Attacker_Deviation_Angle': rule1_fraction,
            'Off_Angle': rule2_fraction,
            'dRange': rule3_fraction,
            'Target': y[i]
        })

    return pd.DataFrame(results)