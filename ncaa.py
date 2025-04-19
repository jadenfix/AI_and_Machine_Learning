import pandas as pd
import numpy as np
name = ['jpfix']
file_path = "ncaa.pkl"
df = pd.read_pickle(file_path)
print(df.head(10))
print(df.columns)
#A = np.zeros((10,10))
#b = np.zeros((10,1))
#print(A)
#print(b)
def cfb_rank(df):
    N = len(df)
    A = np.zeros((N,N))
    b = np.zeros(N)
    for i in range(N):
        A[i,i]= len(df.iloc[i,4]) # 4th col is opponents
        b[i] = df.iloc[i,3] #3rd is point diff 
        for opponents in df.iloc[i,4]:
            A[i,opponents] -=1
    
    x = np.linalg.solve(A,b) #solves linear system, Ax=b and returns x
    x = x - np.mean(x) #normalization
    results_df = pd.DataFrame({
        'team': df.iloc[:,0],
        'score': x})
    results_df = results_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    print("This is Results DF:",results_df,A,b)
    return [results_df, A, b] 
# Call the function and get results
rankings, A_matrix, b_vector = cfb_rank(df)

# Display the rankings DataFrame, A matrix, and b vector
print(rankings)
print("A matrix:\n", A_matrix)
print("b vector:\n", b_vector)