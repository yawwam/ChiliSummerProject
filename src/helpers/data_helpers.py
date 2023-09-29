import pandas as pd
import numpy as np
import glob
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


"""
***********************************************************************************************************************
Data generation functions
***********************************************************************************************************************
"""

def header_sep_check(file_path, sniffer, first_col):
    """
    Check if the given CSV file has a header row and determine the delimiter used in the file.
    Args:
        file_path (str): The path to the CSV file.
        sniffer (csv.Sniffer): A CSV sniffer object used to determine the delimiter.
        first_col (str): The name of the expected first column in the CSV file.

    Returns:
        A tuple containing the header row index (0 if present, None otherwise) and the delimiter used in the file.
    """
    header=None
    with open(file_path, 'r') as csvfile:
        first_line = csvfile.readline()
        separator = sniffer.sniff(first_line)
        if first_col in first_line:
            header=0
    return header,separator.delimiter

def read_cap_img(file_path, sniffer, default_columns):
    """
    Read a CSV file containing capacitive images data

    Args:
        file_path (str): The path to the CSV file.
        sniffer (csv.Sniffer): A CSV sniffer object used to determine the delimiter.
        default_columns (list): A list of column names to use if the file does not have a header row.

    Returns:
        A pandas DataFrame containing the capacitive images data.
    """
    #look for file delimiter, check if the file comes with a header
    header, separator = header_sep_check(file_path, sniffer, 'TimeStamps')                
    #read and assign columns if necessary   
    if header==None:
        cap_img=pd.read_csv(file_path, sep=separator, low_memory=False, on_bad_lines='skip', header=None)
        cap_img.columns=default_columns
    else :
        cap_img=pd.read_csv(file_path,sep= separator,low_memory=False,on_bad_lines='skip')
        #check if cap_img is as expected
        if cap_img.columns.values.tolist() != default_columns:
            raise KeyError('Columns are not as expected for file: ' + file_path)
    cap_img = cap_img.rename(columns={'TimeStamps': 'Timestamp'})
    return cap_img

def read_skeleton(file_path,sniffer,default_columns):
    """
    Read a CSV file containing skeleton data

    Args:
        file_path (str): The path to the CSV file.
        sniffer (csv.Sniffer): A CSV sniffer object used to determine the delimiter.
        default_columns (list): A list of column names to use if the file does not have a header row.

    Returns:
        A pandas DataFrame containing the skeleton data.
    """
    _ , separator = header_sep_check(file_path, sniffer, 'Timestamp')                
    skeleton=pd.read_csv(file_path,sep= separator,index_col=0)
        
    if skeleton.columns.values.tolist() != default_columns:
        raise KeyError('Columns are not as expected for file: ' + file_path)

    return skeleton

def process_cap_img(cap_img,trunc=False):
    """
    Process the capacitive images data

    Args:
        cap_img (DataFrame): The capacitive images dataframe.
        trunc (bool): Whether to truncate the capacitive images to 41 rows instead of 42.

    Returns:
        A pandas DataFrame containing the processed capacitive images data.
    """
    numeric_cols =  list(map(str,range(3024)))
    pixels= cap_img[numeric_cols]
    img = pixels.values.reshape(-1,42,72).astype(int)
    #we keep only 41 rows instead of 42 to match touchpose
    cap_img['ScanSizeY'] = 41 if trunc else 42
    img = img[:,:41,:] if trunc else img
    #merge pixels columns into one array
    cap_img['cap_img']= list(img)
    cap_img = cap_img.drop(pixels.columns, axis=1)
    cap_img[cap_img.columns.drop('cap_img')] = cap_img[cap_img.columns.drop('cap_img')].apply(pd.to_numeric, errors='coerce')
    cap_img.dropna(inplace=True)
    return cap_img

def process_skeleton(skeleton,last_fingers_cols,trunc=False):
    """
    Process the skeleton data

    Args:
        skeleton (DataFrame): The skeleton dataframe.
        last_fingers_cols (list): A list of columns to drop from the skeleton dataframe (last measurement of each finger)
        trunc (bool): Whether to remove last_finger_cols or not.

    Returns:
        A pandas DataFrame containing the processed skeleton data.
    """
    #we keep only four first measurements for each finger instead of 5
    if trunc : 
        skeleton=skeleton.drop(last_fingers_cols,axis=1)
    hand_cords = skeleton.drop("Timestamp",axis=1)
    skeleton['skeleton'] = list(hand_cords.values.reshape(-1,21,3).astype(float))
    skeleton = skeleton.drop(hand_cords.columns, axis=1)
    skeleton[skeleton.columns.drop('skeleton')] = skeleton[skeleton.columns.drop('skeleton')].apply(pd.to_numeric, errors='coerce')
    skeleton.dropna(inplace=True)
    skeleton.Timestamp = skeleton.Timestamp*1000
    return skeleton

def map_cap_img_to_skeleton(skeleton,cap_img,threshhold=40,average=True):
    """
    Map the capacitive images to the skeleton data using bins

    Args:
        skeleton (DataFrame): The skeleton dataframe.
        cap_img (DataFrame): The capacitive images dataframe.
        threshhold (int): The maximum time difference between a capacitive image and a skeleton data point.
        average (bool): Whether to average the capacitive images or not.

    Returns:
        A pandas DataFrame containing the mapped capacitive images data.
    """
    #we map the skeleton to the cap_img
    # we will make both df overlap
    max_cap, min_cap = max(cap_img.Timestamp), min(cap_img.Timestamp)
    mask_skel = (skeleton.Timestamp >= min_cap) & (skeleton.Timestamp <= max_cap)
    if not mask_skel.any():
        return pd.DataFrame()
    skeleton=skeleton[mask_skel].reset_index(drop=True)
    bins = skeleton.Timestamp
    x = cap_img.Timestamp
    cap_img['where_to'] = np.digitize(x, bins)
    cap_img.where_to=cap_img.where_to.map(lambda x: x-1)
    #remove skeletons that hold too much cap_img because of the jumps
    ranking=skeleton.Timestamp.diff().sort_values(ascending=False)
    ranking=ranking[ranking > threshhold]
    ranking.index=ranking.index-1
    cols = cap_img.columns.drop(['Timestamp','MergeTimeStamps','cap_img','where_to'])
    dicts = {x: 'first' for x in cols}
    dicts['cap_img'] = np.mean
    #need also to remove first and last mapping since all out of time cap_img would map there
    to_remove = list(ranking.index)+[-1,len(skeleton)-1]
    if not average :
        duplicates = cap_img.where_to.value_counts()[cap_img.where_to.value_counts()>1].index
        to_remove = to_remove + list(duplicates)
    cap_img=cap_img[~cap_img.where_to.isin(to_remove)]
    res=cap_img.groupby('where_to').agg(dicts)
    data=skeleton.iloc[res.index]
    data = pd.concat([data,res],axis=1).reset_index(drop=True)
    return data

def perfect_map(skeleton,cap_img,time_tolerance = 7) :
    """
    Map the capacitive images to the skeleton data using the nearest timestamp

    Args:
        skeleton (DataFrame): The skeleton dataframe.
        cap_img (DataFrame): The capacitive images dataframe.
        time_tolerance (int): The maximum time difference between a capacitive image and a skeleton data point.

    Returns:    
        A pandas DataFrame containing the mapped capacitive images data.
    """
    merged_df = pd.merge_asof(
        skeleton, cap_img,
        left_on='Timestamp', right_on='Timestamp',
        direction='nearest', tolerance=time_tolerance 
    ).dropna().reset_index(drop=True)
    return merged_df.drop(columns=['ScanSizeX','ScanSizeY','MergeTimeStamps'])

"""
********************************************************************************************************************** 
Data reading and processing functions
**********************************************************************************************************************
"""

def get_touch_model(sub_folder, name, model_path) :
    """
    Load a touchpose model from a given path

    Args:
        sub_folder (str): The sub folder containing the model.
        name (str): The name of the model.
        model_path (str): The path to the model.

    Returns:
        A gien touchpose model.
    """
    model = load_model(model_path+"/"+str(sub_folder)+"/"+str(name)+".hdf5")
    return model

def joint_norm(chili_data, keep_wrist, norm='all', dim2=False):
    """
    Normalize the skeleton data in chili_data Dataframe

    Args:
        chili_data (DataFrame): The skeleton dataframe.
        keep_wrist (bool): Whether to keep the wrist or not.
        norm (str): The type of normalization to apply.
        dim2 (bool): Whether to use 2D or 3D skeleton data.

    Returns:
        A tuple containing the normalized chili_data data, the mean and the standard deviation.
    """
    dim = 2 if dim2 else 3
    if keep_wrist :
        #scale everything besides wrist
        chili_data['skeleton']= chili_data['skeleton'].apply(lambda x : np.vstack((x[0],x[1:]-x[0])))
    else :
        #wrist is now 0
        chili_data['skeleton']= chili_data['skeleton'].apply(lambda x :x-x[0])

    if norm=='joint':
        #shape (21,3)/(21,2)
        mean = np.mean(chili_data['skeleton'].values)
        std = np.std(chili_data['skeleton'].values)
    elif norm=='axis' :
        #shape (1,3)/(1,2), repeat it 21 times to multiply it correctly for each joint in loss function
        mean = np.repeat(np.mean(chili_data.skeleton.apply(lambda x : np.mean(x,axis=0)).values).reshape(-1,dim),21,axis=0)
        std = np.repeat(np.std(chili_data.skeleton.apply(lambda x : np.std(x,axis=0)).values).reshape(-1,dim),21,axis=0)
    elif norm=='all' :
        #mean and std of 63/42 joints
        mean=np.mean(np.concatenate(chili_data['skeleton'].values))
        std=np.std(np.concatenate(chili_data['skeleton'].values))

    chili_data['skeleton'] = chili_data['skeleton'].apply(lambda x : (x - mean)/std)
    
    if ((not keep_wrist) & (not norm=='all')) :
        #put back 0 in wrist cords
        chili_data['skeleton'] = chili_data['skeleton'].apply(lambda x : np.vstack((np.zeros([1, dim]),x[1:])))
        #std 1 and mean 0 to keep 0 after rescaling in loss function
        std[0][:] = 1
        mean[0][:] = 0

    if not norm=='all' :
        mean,std=mean.reshape(-1,21*dim),std.reshape(-1,21*dim)

    return chili_data, mean, std
  
def filter_std(chili_data, m, lower=False) :
    """
    Filter the capacitive images data based on the standard deviation

    Args:
        chili_data (DataFrame): The capacitive images dataframe.
        m (int): The number of standard deviations to keep.
        lower (bool): Whether to only apply a lower bound.

    Returns:
        A pandas DataFrame containing the filtered capacitive images data.
    """
    caps = chili_data.cap_img.apply(lambda x : len(x[x!=0]))
    mean_area, std_area = caps.mean(), caps.std()
    caps=caps[caps>(mean_area - m*std_area)] if lower else caps[(caps>(mean_area - m*std_area)) & (caps<(mean_area + m*std_area)) ]
    chili_data=chili_data.loc[caps.index].reset_index(drop=True)
    return chili_data

def filter_val(chili_data, val) :
    """
    Filter the capacitive images data based on a given value

    Args:
        chili_data (DataFrame): The capacitive images dataframe.
        val (int): The value to use as a threshold.

    Returns:
        A pandas DataFrame containing the filtered capacitive images data.
    """
    caps = chili_data.cap_img.apply(lambda x : len(x[x!=0]))
    mean_area = caps.mean()
    caps=caps[(caps>(mean_area - val))]
    chili_data=chili_data.loc[caps.index].reset_index(drop=True)
    return chili_data
  
def read_data(keep_wrist, data_path, dim2=False , dim_to_remove='x', filter_cap_img=None, norm_skel=None, norm_cap_img=True) :
    """
    Read and process the data from a given path

    Args:
        keep_wrist (bool): Whether to keep the wrist or not.
        data_path (str): The path to the data.
        dim2 (bool): Whether to use 2D or 3D skeleton data.
        dim_to_remove (str): The dimension to remove from the capacitive images.
        filter_cap_img (function): A function to use to filter the capacitive images.
        norm_skel (str): The type of normalization to apply to the skeleton data.
        norm_cap_img (bool): Whether to normalize the capacitive images or not.

    Returns:
        A tuple containing chili_data Dataframe, the mean, the standard deviation and the skeleton columns
        or chili_data Dataframe and the skeleton columns if norm_skel is None.
    """
    dims = {'x','y','z'}
    dims.remove(dim_to_remove)
    wrist_cols = [f'wrist_{d}' for d in dims] if dim2 else ['wrist_x','wrist_y','wrist_z']
    #create skeleton_cols depending on chosen dimension
    if dim2 :
        skeleton_cols = wrist_cols + sum([[f'{x}_{y}_{d}' for d in dims] for x in ['Thumb','Index','Middle','Ring','Pinky'] for y in range(4)],[])
    else :
        skeleton_cols = wrist_cols + sum([[f'{x}_{y}_x', f'{x}_{y}_y', f'{x}_{y}_z'] for x in ['Thumb','Index','Middle','Ring','Pinky'] for y in range(4)],[])

    files = glob.glob(data_path+'*.pkl')
    chili_data = pd.concat([pd.read_pickle(fp) for fp in files], ignore_index=True)

    if dim2 :
        if dim_to_remove == 'x' :
            chili_data['skeleton']= chili_data['skeleton'].apply(lambda x : x[:,1:].astype(np.float32))
        elif dim_to_remove == 'y' :
            chili_data['skeleton']= chili_data['skeleton'].apply(lambda x : x[:,::2].astype(np.float32))
        elif dim_to_remove == 'z' :
            chili_data['skeleton']= chili_data['skeleton'].apply(lambda x : x[:,:2].astype(np.float32))
        
        print(f'Dimension removed from capacitive images : {dim_to_remove}')
        
    l = len(chili_data)
    #remove data points with empty capacitive images
    chili_data=chili_data.loc[~chili_data.cap_img.apply(lambda x :not x.any())].reset_index(drop=True)
    print('Before removing empty cap_img : ',l ,' now : ', len(chili_data))

    if filter_cap_img is not None :
        l = len(chili_data)
        chili_data = filter_cap_img(chili_data)
        print('before filtering cap_img : ',l ,' now : ', len(chili_data))
    
    #Normalize cap_img
    if norm_cap_img :
        meancap = np.mean(np.concatenate(chili_data['cap_img'].values))
        stdcap = np.std(np.concatenate(chili_data['cap_img'].values))
        chili_data['cap_img']= chili_data['cap_img'].apply(lambda x : (x-meancap)/stdcap)

    print('Unique Timestamps : ' , chili_data.Timestamp.is_unique)
    chili_data = chili_data.sort_values(by=['Timestamp']).reset_index(drop=True)
  
   #Normalize skeleton data
    if norm_skel is not None:
        chili_data, mean, std = joint_norm(chili_data, keep_wrist, norm=norm_skel, dim2=dim2)
        return chili_data, mean, std, skeleton_cols
    else : 
        return chili_data, skeleton_cols



"""
********************************************************************************************************************** 
Data loading functions for model
**********************************************************************************************************************
"""

def data_multiload(df, dim2=False):
    """
    Load the data from a given dataframe

    Args:
        df (DataFrame): chili_data dataframe.
        dim2 (bool): Whether to use 2D or 3D skeleton data.
    
    Returns:
        A tuple containing the capacitive images, the skeleton and the gesture data.
    """
    x_cap_img = np.array(df["cap_img"].tolist())
    y_skeleton = np.array(df["skeleton"].tolist())
    y_skeleton = y_skeleton.reshape(-1,42) if dim2 else y_skeleton.reshape(-1,63)
    y_gesture = np.array(pd.get_dummies(df["gesture"],dtype=int).values.tolist())
    return x_cap_img, y_skeleton, y_gesture

def load_data(chili_data, train_type, train_size, seed, dim2=False, fold = None) :
    """
    Load the needed data from a given dataframe for a given training type

    Args:
        chili_data (DataFrame): The skeleton dataframe.
        train_type (str): The type of training to use.
        train_size (float): The size of the training set.
        seed (int): The random seed to use.
        dim2 (bool): Whether to use 2D or 3D skeleton data.
        fold (str): The fold to use for cross validation.
    
    Returns:
        A tuple containing the training, validation and test data.
    """
    train_size, val_size, test_size = (0.8, 0.1, 0.1) if train_size == 0.8 else (0.6, 0.2, 0.2)

    if train_type == 'tr_test_split' :
        train, test_valid = train_test_split(chili_data, test_size=test_size+val_size, random_state=seed, stratify = chili_data['gesture'])
        valid, test = train_test_split(test_valid, test_size=0.5, random_state=seed, stratify = test_valid['gesture'])
    elif train_type == 'cross_gesture' :
        train = chili_data[chili_data.gesture!=fold].copy()
        test = chili_data[chili_data.gesture==fold].copy()
    elif train_type == 'cross_subject' :
        train = chili_data[chili_data.subject!=fold].copy()
        test = chili_data[chili_data.subject==fold].copy()
    elif train_type == 'time' :
        temp = train_size + val_size
        train = chili_data.groupby(['subject','trial']).apply(lambda x: x.head(round(temp*len(x)))).reset_index(drop=True)
        valid = train.groupby(['subject','trial']).apply(lambda x: x.tail(round( (val_size/temp)*len(x) ))).reset_index(drop=True)
        train = train.groupby(['subject','trial']).apply(lambda x: x.head(round( (train_size/temp)*len(x) ))).reset_index(drop=True)
        test = chili_data.groupby(['subject','trial']).apply(lambda x: x.tail(round(test_size*len(x)))).reset_index(drop=True)

    data = {'train':train, 'test':test} if train_type in ['cross_gesture','cross_subject'] else {'train':train, 'valid': valid, 'test':test}
    for d in data :
        data[d].gesture = pd.Categorical(data[d].gesture, categories=chili_data.gesture.unique())
        data[d] = shuffle(data[d],random_state=seed)
        X, y_skeleton, y_gesture = data_multiload(data[d], dim2=dim2)
        y = tf.concat([y_skeleton,y_gesture],axis=1)
        data[d] = {'X' : X, 'y' : y}
    return data
  
