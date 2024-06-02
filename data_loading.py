"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import pandas as pd
import pickle

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data
    

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy', 'ECG', 'SWANSF']
  
  if data_name == 'stock':
    ori_data = np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'energy':
    ori_data = np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)
  elif data_name == 'SWANSF':
    with open('data/swansf.pkl', 'rb') as f:
        ori_data = pickle.load(f)
  elif data_name == 'ECG':
    parsed_data = []
    with open('data/ECG5000_data.txt', 'r') as file:
        for line in file:
            # Splitting each line by space and filtering out empty strings
            parsed_line = [float(x) for x in line.strip().split() if x]
            parsed_data.append(parsed_line)

    # Converting the list of lists into a DataFrame
    df = pd.DataFrame(parsed_data)

    # Dropping the first column (Label)
    df = df.drop(df.columns[0], axis=1)

    # Convert the DataFrame into a 3D NumPy array with the shape (num_samples, num_timestamps, num_attributes = 1)
    np_array_3d = df.values.reshape((4500, 140, 1))
    
  data = []      
  if data_name == 'ECG':
    data_reshaped = np_array_3d.reshape(np_array_3d.shape[0], np_array_3d.shape[1])
    data_transposed = data_reshaped.T
    scaled_data_transposed = MinMaxScaler(data_transposed)
    data = scaled_data_transposed.T.reshape(np_array_3d.shape[0], np_array_3d.shape[1], 1)
        
  elif data_name == 'SWANSF': 
    data = ori_data
    
  else:
    # Flip the data to make chronological data
      ori_data = ori_data[::-1]
      # Normalize the data
      ori_data = MinMaxScaler(ori_data)

      # Preprocess the dataset
      temp_data = []    
      # Cut data by sequence length
      for i in range(0, len(ori_data) - seq_len):
        _x = ori_data[i:i + seq_len]
        temp_data.append(_x)

      # Mix the datasets (to make it similar to i.i.d)
      idx = np.random.permutation(len(temp_data))    
      
      for i in range(len(temp_data)):
        data.append(temp_data[idx[i]])
    

    
    
  return data