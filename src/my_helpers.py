import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv


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
    return cap_img

def process_skeleton(file_name,finger_cols):
    #we keep only four firt measurements for each finger instead of 5
    skeleton = pd.read_csv(file_name,sep=',', index_col=0)
    s=skeleton.copy()
    s=s.drop(finger_cols,axis=1)
    return s


def plot_hand(hand_coordinates):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

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
    ax.set_title('Hand Coordinates')

    # Show the plot
    plt.show()
