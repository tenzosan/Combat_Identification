from src.config import *
def bm_arrange(final_fused,y_test):
    final_fused = pd.DataFrame(final_fused)
    y_test = pd.DataFrame(y_test)
    data = pd.concat([final_fused,y_test], axis=1)
    # Rearrange
    data1 = data[data.iloc[:, -1] == 'Bomber']
    data2 = data[data.iloc[:, -1] == 'CivilianCommercialAirliner']
    data3 = data[data.iloc[:, -1] == 'CivilianHelicopter']
    data4 = data[data.iloc[:, -1] == 'CruiseMissile']
    data5 = data[data.iloc[:, -1] == 'MilitaryHelicopter']
    data6 = data[data.iloc[:, -1] == 'MultiroleFighter']
    data7 = data[data.iloc[:, -1] == 'SmallCivilian']
    data8 = data[data.iloc[:, -1] == 'SmallUnmanned']
    rearranged_data = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8], axis=0)
    rearranged_data = pd.DataFrame(rearranged_data.iloc[:,:].values)
    return rearranged_data