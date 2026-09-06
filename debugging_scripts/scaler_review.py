import joblib
import numpy as np
from config import GLOBAL_CONFIG


def analyze_scaler():
    scaler = joblib.load(GLOBAL_CONFIG.pipeline.scaler_path)

    features = GLOBAL_CONFIG.pipeline.input_features

    # For standard scaler
    for name, mean, scale in zip(features, scaler.mean_, scaler.scale_):
        print(f"{name:<15} | {mean:<15.5f} | {scale:<15.5f}")

    # For RobustScaler
    #print("Scale (IQR):", scaler.scale_)
    #print("Center (Median):", scaler.center_)

    # Test on random data
    #sample_input = np.array([[12.0223, -0.6695, 3.4839, 1.9046, -0.0631]])
    #scaled_output = scaler.transform(sample_input)

    #print(f"Raw values:   {sample_input[0]}")
   # print(f"Scaled values: {scaled_output[0]}")


if __name__ == "__main__":
    analyze_scaler()


### Output:
#speed           | 5.03642         | 5.63638        
#acceleration    | 1.19932         | 2.26579        
#distance        | 22.20481        | 16.31129       
#lane_offset     | -0.00099        | 0.08926        
#heading_error   | -0.00006        | 0.02639 

#Raw values:   [12.0223 -0.6695  3.4839  1.9046 -0.0631]
#Scaled values: [ 1.23942648 -0.82479761 -1.1477273  21.34980309 -2.38899688]

# We can see that scaled values on the lane offset are 21+ and this lead the nn to behave strangly, since one step is very small.
# For first fix, we will multiply it by 0.1 but we need to find better solution for it

# After changing the scaler to RobustScaler we hit even bigger problem. Due to really small amount of data in curves we baasicaly divided by 0.
# We create dataset_review.py to see where does out data stands