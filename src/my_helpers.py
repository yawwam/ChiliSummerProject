import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_cap_img(file_name):
    cap_img = pd.read_csv(file_name,sep=' ')
    pixels= cap_img.iloc[:, -3024:]
    cap_img['img'] = pixels.values.tolist()
    cap_img = cap_img.drop(pixels.columns, axis=1)
    return cap_img

def process_cap_img(cap_img):
    numeric_cols =  list(map(str,range(3024)))
    pixels= cap_img[numeric_cols]
    cap_img['cap_img'] = np.array(pixels.values.tolist()).astype(int)
    cap_img = cap_img.drop(pixels.columns, axis=1)
    return cap_img

def read_skeleton(file_name):
    #we keep only four firt measurements for each finger instead of 5
    skeleton = pd.read_csv(file_name,sep=',', index_col=0)
    s=skeleton.copy()
    for x in ['Thumb','Index','Middle','Ring','Pinky'] :
        s=s.drop([f'{x}_4_x', f'{x}_4_y', f'{x}_4_z'],axis=1)
    return s

def prepare_input(df,maxVal=1501):
    input=np.array(df['img'].map(lambda x: np.array(x).reshape(42,72)).tolist())
    input=input[:,:41,:]
    input = input/maxVal
    return input


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
