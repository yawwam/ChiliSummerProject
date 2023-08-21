import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.models import load_model



def header_sep_check(file_path,sniffer, first_col):
    header=None
    with open(file_path, 'r') as csvfile:
        first_line = csvfile.readline()
        separator = sniffer.sniff(first_line)
        if first_col in first_line:
            header=0
    return header,separator.delimiter

def read_cap_img(file_path,sniffer,default_columns):
    #look for file delimiter, check if the file comes with a header
    header, separator = header_sep_check(file_path, sniffer, 'TimeStamps')                
    #read and assign columns if necessary   
    if header==None:
        cap_img=pd.read_csv(file_path, sep= separator, low_memory=False, on_bad_lines='skip', header=None)
        cap_img.columns=default_columns
    else :
        cap_img=pd.read_csv(file_path,sep= separator,low_memory=False,on_bad_lines='skip')
        #check if cap_img is as expected
        if cap_img.columns.values.tolist() != default_columns:
            raise KeyError('Columns are not as expected for file: ' + file_path)
    cap_img = cap_img.rename(columns={'TimeStamps': 'Timestamp'})
    return cap_img

def read_skeleton(file_path,sniffer,default_columns):
    _ , separator = header_sep_check(file_path, sniffer, 'Timestamp')                
    skeleton=pd.read_csv(file_path,sep= separator,index_col=0)
        
    if skeleton.columns.values.tolist() != default_columns:
        raise KeyError('Columns are not as expected for file: ' + file_path)
    
    return skeleton

def process_cap_img(cap_img,trunc=False):
    numeric_cols =  list(map(str,range(3024)))
    pixels= cap_img[numeric_cols]
    img = pixels.values.reshape(-1,42,72).astype(int)
    #we keep only 41 rows instead of 42
    cap_img['ScanSizeY'] = 41 if trunc else 42
    img = img[:,:41,:] if trunc else img
    cap_img['cap_img']= list(img)
    cap_img = cap_img.drop(pixels.columns, axis=1)
    cap_img[cap_img.columns.drop('cap_img')] = cap_img[cap_img.columns.drop('cap_img')].apply(pd.to_numeric, errors='coerce')
    cap_img.dropna(inplace=True)
    return cap_img

def process_skeleton(skeleton,last_fingers_cols,trunc=False):
    #we keep only four firt measurements for each finger instead of 5
    if trunc : 
        skeleton=skeleton.drop(last_fingers_cols,axis=1)
    hand_cords = skeleton.drop("Timestamp",axis=1)
    skeleton['skeleton'] = list(hand_cords.values.reshape(-1,21,3).astype(int))
    skeleton = skeleton.drop(hand_cords.columns, axis=1)
    skeleton[skeleton.columns.drop('skeleton')] = skeleton[skeleton.columns.drop('skeleton')].apply(pd.to_numeric, errors='coerce')
    skeleton.dropna(inplace=True)
    skeleton.Timestamp = skeleton.Timestamp*1000
    return skeleton

def map_cap_img_to_skeleton(skeleton,cap_img,threshhold=80):
    #we map the skeleton to the cap_img
    # we will make both df overlap
    max_cap, min_cap = max(cap_img.Timestamp), min(cap_img.Timestamp)
    mask_skel = (skeleton.Timestamp >= min_cap) & (skeleton.Timestamp <= max_cap)
    if not mask_skel.any():
        #TODO LOOK FOR DRIFT AND REPORT IT
        return pd.DataFrame()
    #add one frame before and after if possible for bins
    loc1=mask_skel[mask_skel].first_valid_index()
    locn=mask_skel[mask_skel].last_valid_index()
    if loc1>0:
        mask_skel.loc[loc1-1]=True
    if locn<len(skeleton)-1:
        mask_skel.loc[locn+1]=True
    skeleton=skeleton[mask_skel].reset_index(drop=True)
    bins = skeleton.Timestamp
    x = cap_img.Timestamp
    cap_img['where_to'] = np.digitize(x, bins)
    cap_img.where_to=cap_img.where_to.map(lambda x: x-1)
    cap_img=cap_img[~cap_img.where_to.isin([-1,len(skeleton)-1])]
    #remove skeletons that hold too much cap_img because of the junmps
    ranking=skeleton.Timestamp.diff().sort_values(ascending=False)
    ranking=ranking[ranking > threshhold]
    ranking.index=ranking.index-1
    cols = cap_img.columns.drop(['cap_img','where_to', 'MergeTimeStamps', 'Timestamp'])
    dict = {x: lambda x: x.value_counts().index[0] for x in cols}
    dict['cap_img'] = np.mean
    cap_img=cap_img[~cap_img.where_to.isin(ranking.index)]
    res=cap_img.groupby('where_to').agg(dict)
    data=skeleton.iloc[res.index]
    data = pd.concat([data,res],axis=1).reset_index(drop=True)
    return data


def data_splits(df, id_arr):
    X_cap, Y_skeleton = None, None
    for p_id in id_arr:
        df_extract = df.get_group(p_id)
        x_cap_img = np.array(df_extract["cap_img"].tolist())
        y_skeleton = np.array(df_extract["skeleton"].tolist()).reshape(-1,63)

        x_cap_img[x_cap_img < 0] = 0
        maxVal = 3023
        x_cap_img = x_cap_img/maxVal

        if X_cap is None:
            X_cap = x_cap_img
            Y_skeleton = y_skeleton
        else:
            X_cap = np.vstack((X_cap, x_cap_img))
            Y_skeleton = np.vstack((Y_skeleton, y_skeleton))
    return X_cap, Y_skeleton

def get_eth_model(sub_folder,name, model_path) :
    model = load_model(model_path+"/"+str(sub_folder)+"/"+str(name)+".hdf5")
    return model

def data_load(df):
    x_cap_img = np.array(df["cap_img"].tolist())
    y_skeleton = np.array(df["skeleton"].tolist()).reshape(-1,63)
    return x_cap_img, y_skeleton

def data_multiload(df):
    x_cap_img = np.array(df["cap_img"].tolist())
    y_skeleton = np.array(df["skeleton"].tolist()).reshape(-1,63)
    y_gesture = np.array(pd.get_dummies(df["gesture"],dtype=int).values.tolist())
    return x_cap_img, y_skeleton, y_gesture

def skeleton_loss(y_true, y_pred):
    J = tf.constant(21.0)  # number of predicted joints
    return tf.reduce_sum(tf.square(y_true - y_pred)) / (J * 3)

def auc_pck(y_true, y_pred):
    thresholds = np.arange(20, 51, 5)  # Thresholds from 20 mm to 50 mm
    pck_values = []


    # Calculate the Euclidean distance between predicted and true joint coordinates
    distances = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

    # Calculate the percentage of correct keypoints for each threshold
    for threshold in thresholds:
        correct_keypoints = tf.cast(distances < threshold, tf.float32)
        pck = tf.reduce_mean(correct_keypoints) * 100.0
        pck_values.append(pck)

    return tf.convert_to_tensor(pck_values, dtype=tf.float32)


def end_point_error(y_true, y_pred):
    # Calculate the Euclidean distance between predicted and true joint coordinates
    distances = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))
    # Calculate the mean EPE over all joints
    mean_epe = tf.reduce_mean(distances)
    return mean_epe

def plot_hand(hand_coordinates, ax, title=''):

    # Plot the hand coordinates
    for i in range(len(hand_coordinates)):
        x = hand_coordinates[i][0]
        y = hand_coordinates[i][1]
        z = hand_coordinates[i][2]
        
        # Plot the coordinates
        ax.scatter(x, y, z, color='red', marker='o')

    # Connect the points to form hand segments
    hand_segments = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]

    for segment in hand_segments:
        x_segment = [hand_coordinates[segment[0]][0], hand_coordinates[segment[1]][0]]
        y_segment = [hand_coordinates[segment[0]][1], hand_coordinates[segment[1]][1]]
        z_segment = [hand_coordinates[segment[0]][2], hand_coordinates[segment[1]][2]]
        ax.plot(x_segment, y_segment, z_segment, color='blue')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

def plot_hand_compare(y_true, y_out, num_samples=5):
    # Create a 3D plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 20), subplot_kw={'projection': '3d'})
    y_true= y_true.reshape(-1,21,3)
    y_out = y_out.reshape(-1,21,3)
    for i in range(num_samples):
        # Plot hand coordinates for y_test
        plot_hand(y_true[i], axes[i, 0], title='True skeleton {}'.format(i + 1))

        # Plot hand coordinates for y_out
        plot_hand(y_out[i], axes[i, 1], title='Predicted skeleton {}'.format(i + 1))

    # Adjust layout and display the figure
    plt.tight_layout()
    plt.show()


