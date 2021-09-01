import numpy as np
from scipy.stats import norm
import pandas as pd
import experiments_helper

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_small_dataset(size, seed = 0):
    def generate_A():
        np.random.seed(seed+1)
        return pd.Series(np.random.normal(loc=6, scale=1, size=size), name="A")

    def generate_B():
        return pd.Series(experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], size, seed=seed+2), name="A")

    def generate_C(evidence):
        np.random.seed(seed+3)
        return pd.Series(evidence['A'] * evidence['B'] + np.random.normal(size=size), name="C")
    
    def generate_D(evidence):
        np.random.seed(seed+4)
        return pd.Series(10 + 0.8*evidence['C'] + np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="D")

    def generate_E(evidence):
        np.random.seed(seed+5)
        return pd.Series(sigmoid(evidence['D']) + np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="E")

    df = pd.DataFrame()
    df['A'] = generate_A()
    df['B'] = generate_B()
    df['C'] = generate_C(df)
    df['D'] = generate_D(df)
    df['E'] = generate_E(df)

    return df


def generate_medium_dataset(size, seed = 0):
    def generate_X1():
        return pd.Series(experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], size, seed=seed+1), name="X1")

    def generate_X2(evidence):
        np.random.seed(seed+2)
        return pd.Series(10 + 0.8*evidence['X1'] + np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="X2")

    def generate_X3(evidence):
        np.random.seed(seed+3)
        return pd.Series(sigmoid(evidence['X1']) + np.random.normal(size=size), name="X3")
    
    def generate_X4(evidence):
        np.random.seed(seed+4)
        return pd.Series(evidence['X2'] * evidence['X3'] + np.random.normal(size=size), name="X4")

    def generate_X5(evidence):
        np.random.seed(seed+5)
        return pd.Series(2.3 - 1.8*evidence['X2'] + 0.7*evidence['X4'] +
                            np.random.normal(loc=0, scale=np.sqrt(2), size=size), name="X5")

    def generate_X6(evidence):
        np.random.seed(seed+6)
        return pd.Series(3*sigmoid(0.5*evidence['X4']) + np.random.normal(loc=0, scale=np.sqrt(1), size=size), name="X6")

    def generate_X7(evidence):
        np.random.seed(seed+7)
        return pd.Series(5.7 - 1.5*evidence['X4'] + 0.5*evidence['X3'] +
                            np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="X7")

    def generate_X8(evidence):
        np.random.seed(seed+8)
        return pd.Series(1. - 0.3*evidence['X5'] + 0.3*evidence['X6'] +
                            np.random.normal(loc=0, scale=np.sqrt(0.25), size=size), name="X8")

    def generate_X9(evidence):
        np.random.seed(seed+9)
        return pd.Series(sigmoid(evidence['X6'] * evidence['X7']) +
                            np.random.normal(loc=0, scale=np.sqrt(1.5), size=size), name="X9")

    def generate_X10(evidence):
        np.random.seed(seed+10)
        return pd.Series(-3. + 2*evidence['X8'] - 3.5*evidence['X9'] +
                            np.random.normal(loc=0, scale=np.sqrt(1), size=size), name="X10")

    df = pd.DataFrame()
    df['X1'] = generate_X1()
    df['X2'] = generate_X2(df)
    df['X3'] = generate_X3(df)
    df['X4'] = generate_X4(df)
    df['X5'] = generate_X5(df)
    df['X6'] = generate_X6(df)
    df['X7'] = generate_X7(df)
    df['X8'] = generate_X8(df)
    df['X9'] = generate_X9(df)
    df['X10'] = generate_X10(df)

    return df

def generate_large_dataset(size, seed = 0):
    def generate_X1():
        return pd.Series(experiments_helper.sample_mixture([0.5, 0.5], [-2, 2], [1, 1], size, seed=seed+1), name="X1")

    def generate_X11():
        np.random.seed(seed+11)
        return pd.Series(np.random.normal(loc=6, scale=1, size=size), name="X11")
    
    def generate_X12():
        return pd.Series(
                experiments_helper.sample_mixture([0.3, 0.4, 0.3], [-2, 0, 2], [0.5, 1, 0.5], size, seed=seed+12),
                name="X12")

    def generate_X2(evidence):
        np.random.seed(seed+2)
        return pd.Series(10 + 0.8*evidence['X1'] + np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="X2")

    def generate_X13(evidence):
        np.random.seed(seed+13)
        return pd.Series(evidence['X11'] / evidence['X2'] + 
                np.random.normal(loc=0, scale=np.sqrt(2), size=size), name="X13")

    def generate_X3(evidence):
        np.random.seed(seed+3)
        return pd.Series(sigmoid(evidence['X1']) + np.random.normal(size=size), name="X3")
    
    def generate_X14(evidence):
        np.random.seed(seed+14)
        return pd.Series(-2 + 3.2*evidence['X3'] - 5*evidence['X12'] +
                            np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="X14")

    def generate_X4(evidence):
        np.random.seed(seed+4)
        return pd.Series(evidence['X2'] * evidence['X3'] + np.random.normal(size=size), name="X4")

    def generate_X5(evidence):
        np.random.seed(seed+5)
        return pd.Series(2.3 - 1.8*evidence['X2'] + 0.7*evidence['X4'] +
                            np.random.normal(loc=0, scale=np.sqrt(2), size=size), name="X5")

    def generate_X15(evidence):
        np.random.seed(seed+15)
        return pd.Series(1 + 0.5*evidence['X5'] + np.random.normal(loc=0, scale=np.sqrt(2), size=size), name="X15")

    def generate_X17(evidence):
        np.random.seed(seed+17)
        return pd.Series(-3 + evidence['X13'] - 0.5*evidence['X15'] +
                            np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="X17")

    def generate_X6(evidence):
        np.random.seed(seed+6)
        return pd.Series(3*sigmoid(0.5*evidence['X4']) + np.random.normal(loc=0, scale=np.sqrt(1), size=size), name="X6")

    def generate_X7(evidence):
        np.random.seed(seed+7)
        return pd.Series(5.7 - 1.5*evidence['X4'] + 0.5*evidence['X3'] +
                            np.random.normal(loc=0, scale=np.sqrt(0.5), size=size), name="X7")

    def generate_X16(evidence):
        np.random.seed(seed+16)
        return pd.Series(sigmoid(0.2*evidence['X7']) + np.random.normal(loc=0, scale=np.sqrt(1), size=size), name="X16")

    def generate_X18(evidence):
        np.random.seed(seed+18)
        return pd.Series(4 - 2*evidence['X16'] + 1.5*evidence['X14'] +
                            np.random.normal(loc=0, scale=np.sqrt(1), size=size), name="X18")

    def generate_X8(evidence):
        np.random.seed(seed+8)
        return pd.Series(1. - 0.3*evidence['X5'] + 0.3*evidence['X6'] +
                            np.random.normal(loc=0, scale=np.sqrt(0.25), size=size), name="X8")

    def generate_X9(evidence):
        np.random.seed(seed+9)
        return pd.Series(sigmoid(evidence['X6'] * evidence['X7']) +
                            np.random.normal(loc=0, scale=np.sqrt(1.5), size=size), name="X9")

    def generate_X10(evidence):
        np.random.seed(seed+10)
        return pd.Series(-3. + 2*evidence['X8'] - 3.5*evidence['X9'] +
                            np.random.normal(loc=0, scale=np.sqrt(1), size=size), name="X10")

    def generate_X19(evidence):
        np.random.seed(seed+19)
        return pd.Series(sigmoid(evidence['X17'] * evidence['X10']) +
                            np.random.normal(loc=0, scale=np.sqrt(4), size=size), name="X19")

    def generate_X20(evidence):
        np.random.seed(seed+20)
        return pd.Series(np.sin(evidence['X18'] / evidence['X10']) +
                            np.random.normal(loc=0, scale=np.sqrt(5), size=size), name="X20")

    df = pd.DataFrame()
    df['X1'] = generate_X1()
    df['X11'] = generate_X11()
    df['X12'] = generate_X12()
    df['X2'] = generate_X2(df)
    df['X13'] = generate_X13(df)
    df['X3'] = generate_X3(df)
    df['X14'] = generate_X14(df)
    df['X4'] = generate_X4(df)
    df['X5'] = generate_X5(df)
    df['X15'] = generate_X15(df)
    df['X17'] = generate_X17(df)
    df['X6'] = generate_X6(df)
    df['X7'] = generate_X7(df)
    df['X16'] = generate_X16(df)
    df['X18'] = generate_X18(df)
    df['X8'] = generate_X8(df)
    df['X9'] = generate_X9(df)
    df['X10'] = generate_X10(df)
    df['X19'] = generate_X19(df)
    df['X20'] = generate_X20(df)

    correct_order = ["X" + str(i) for i in range(1, 21)]

    return df[correct_order]

if __name__ == "__main__":

    for i,n in enumerate(experiments_helper.INSTANCES):
        small_df = generate_small_dataset(n, seed=i)
        medium_df = generate_medium_dataset(n, seed=i+100)
        large_df = generate_large_dataset(n, seed=i+200)

        small_df.to_csv('data/small_' + str(n) + '.csv', index=False)
        medium_df.to_csv('data/medium_' + str(n) + '.csv', index=False)
        large_df.to_csv('data/large_' + str(n) + '.csv', index=False)