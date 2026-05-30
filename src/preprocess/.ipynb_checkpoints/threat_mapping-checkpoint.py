from src.config import *
# ------------------------------ Redefine the label with new target type information ----------------------
def map_threat(Const, Weight):
    # Define conditions for replacements
    conditions = [
        (Const.iloc[:, -1] == "Attacking") & (Weight.iloc[:, -1] == "Level_I"),
        (Const.iloc[:, -1] == "Attacking") & (Weight.iloc[:, -1] == "Level_II"),
        (Const.iloc[:, -1] == "Attacking") & (Weight.iloc[:, -1] == "Level_III"),
        (Const.iloc[:, -1] == "Non Attacking") & (Weight.iloc[:, -1] == "Level_I"),
        (Const.iloc[:, -1] == "Non Attacking") & (Weight.iloc[:, -1] == "Level_II"),
        (Const.iloc[:, -1] == "Non Attacking") & (Weight.iloc[:, -1] == "Level_III")
    ]
    
    # Corresponding choices
    choices = [
        "Hostile",
        "Hostile",
        "Suspicious",
        "Suspicious",
        "NonHostile",
        "NonHostile"
    ]
    
    # Apply conditions and choices using np.select
    Const.iloc[:, -1] = np.select(conditions, choices, default=Const.iloc[:, -1])
    
    # Combine the relevant columns into a new DataFrame
    New_Data = pd.concat([Const.iloc[:, :-1], Weight.iloc[:, :-1], Const.iloc[:, -1]], axis=1)

    return New_Data 