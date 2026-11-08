import numpy as np
import pandas as pd

def auto_align_up_axis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the covariance matrix to find the flattest plane of a 3D point cloud.
    Dynamically swaps axes so the flat plane rests on Ursina's X-Z ground plane,
    making the Y-axis represent height.

    Args:
        df (pd.DataFrame): DataFrame containing 3D point coordinates in 'x', 'y', 'z' columns.

    Returns:
        pd.DataFrame: Aligned copy of the input DataFrame with reoriented axes.
    """
    # 1. Extract just the coordinates as a numpy matrix
    points = df[['x', 'y', 'z']].values
    
    # 2. Calculate Covariance Matrix
    covariance_matrix = np.cov(points.T)
    
    # 3. Get eigenvalues and force them to be real numbers (drops the +0.00j)
    eigenvalues, _ = np.linalg.eig(covariance_matrix)
    eigenvalues = np.real(eigenvalues)
    
    # 4. Find the index of the smallest spread (0=X, 1=Y, 2=Z)
    min_axis = np.argmin(eigenvalues)
    
    # Create a copy of the dataframe to avoid warnings when modifying data
    aligned_df = df.copy()
    
    if min_axis == 2:
        print(f"Detected Z as height (variance: {eigenvalues[2]:.2f}). Swapping Y and Z for Ursina.")
        aligned_df['x'] = df['x']
        aligned_df['y'] = df['z']
        aligned_df['z'] = df['y']
        
    elif min_axis == 0:
        print(f"Detected X as height (variance: {eigenvalues[0]:.2f}). Swapping X and Y for Ursina.")
        aligned_df['x'] = df['y']
        aligned_df['y'] = df['x']
        aligned_df['z'] = df['z']
        
    else:
        print(f"Detected Y as height (variance: {eigenvalues[1]:.2f}). No swap needed.")
        # Data is already perfectly oriented for Ursina
        pass 
        
    return aligned_df
