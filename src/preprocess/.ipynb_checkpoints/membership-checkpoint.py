from src.config import *
# Membership value 
def membership_function(data):
    
    # 0 ==  Bomber 1 == CivilianCommercialAirliner 2 == Civilian Helicopter 3 == CruiseMissile
    # 4 == MilitaryHelicopter 5 == MultiroleFighter 6 == SmallCivilian 7 == SmallUnmanned 8 == Theta_1
    # 9 == Theta_2 10 == Omega 
    
    # Calculate levels
    Level_I = data.iloc[:, 0] + data.iloc[:, 3] + data.iloc[:, 5]
    Level_II = data.iloc[:, 4] + data.iloc[:, 7] + data.iloc[:, 8] + data.iloc[:, 9] + data.iloc[:, 10] 
    Level_III = data.iloc[:, 1] + data.iloc[:, 2] + data.iloc[:, 6]
    

    # Replace the last column based on conditions
    conditions = [
        data.iloc[:, -1].isin(["Bomber", "MultiroleFighter", "CruiseMissile"]),
        data.iloc[:, -1].isin(["MilitaryHelicopter", "SmallUnmanned",'Θ1','Θ2']),
        data.iloc[:, -1].isin(["CivilianCommercialAirliner", "SmallCivilian", "CivilianHelicopter"])
    ]
    
    choices = ["Level_I", "Level_II", "Level_III"]
    
    # Apply conditions to create a new category column
    data.iloc[:, -1] = np.select(conditions, choices, default=data.iloc[:, -1])
    Membership_value = pd.concat([ Level_I, Level_II, Level_III, data.iloc[:, -1]], axis=1)
    
    return Membership_value