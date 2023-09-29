import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from helpers.metrics_helpers import single_end_point_error

  
def plot_skel_joints_distribs(y_train_skeleton,y_test_skeleton,joints_idx,joints_epe):
    """
    Plots the end point error distribution for each joint of the skeleton over all samples in the train and test set.

    Args:
        y_train_skeleton (np.array): The skeleton coordinates of the train set.
        y_test_skeleton (np.array): The skeleton coordinates of the test set.
        joints_idx (list): The indices of the joints to plot.
        joints_epe (list): The end point error of each joint.   
    """
    plt.figure(figsize=(15, 60))
    for i in range(len(joints_idx)) :
        y_train_skel = y_train_skeleton[:,joints_idx[i]]
        y_test_skel = y_test_skeleton[:,joints_idx[i]]
        plt.subplot(21, 3, i+1)
        sns.kdeplot(y_train_skel, label='Train')
        sns.kdeplot(y_test_skel, label='Test')
        plt.title(f'{skeleton_cols[joints_idx[i]]} mean epe : {round(float(joints_epe[i]),2)}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_skel_cap_distribs(y_train_skeleton,y_test_skeleton,X_train,X_test):
    """
    Plots the skeleton and capacitive images distributions for each axis.

    Args:
        y_train_skeleton (np.array): The skeleton coordinates of the train set.
        y_test_skeleton (np.array): The skeleton coordinates of the test set.
        X_train (np.array): The capacitive images of the train set.
        X_test (np.array): The capacitive images of the test set.
    """
    #skeleton distributions
    y_train_skel = y_train_skeleton.reshape(-1,21,3)
    y_test_skel = y_test_skeleton.reshape(-1,21,3)
    x_train,y_train,z_train = y_train_skel[:,:,0], y_train_skel[:,:,1], y_train_skel[:,:,2]
    x_test,y_test,z_test = y_test_skel[:,:,0], y_test_skel[:,:,1], y_test_skel[:,:,2]
    plt.figure(figsize=(15, 4))
    train_axes = [x_train,y_train,z_train]
    test_axes = [x_test,y_test,z_test]
    axes = ['X','Y','Z']
    for i in range(1,4):
        plt.subplot(1, 3, i)
        sns.kdeplot(train_axes[i-1].flatten(), label='Train')
        sns.kdeplot(test_axes[i-1].flatten(), label='Test')
        plt.title(axes[i-1]+'-axis Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()

    plt.suptitle('Skeleton Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()

    #capacitive images distributions
    plt.figure(figsize=(4, 4))
    train_cap_data = X_train.flatten()
    index = np.argwhere(train_cap_data==0)
    train_zeroes= (len(index)/len(train_cap_data))*100
    train_cap_data = np.delete(train_cap_data,index)
    test_cap_data = X_test.flatten()
    index = np.argwhere(test_cap_data==0)
    test_zeroes= (len(index)/len(test_cap_data))*100
    test_cap_data = np.delete(test_cap_data,index)
    sns.kdeplot(train_cap_data,label='Train')
    sns.kdeplot(test_cap_data,label='Test')
    plt.title('Capacitive Images Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(f'Percentage of zeroes in capactive train data',train_zeroes)
    print(f'Percentage of zeroes in capacitive test data',test_zeroes)
    

def plot_hand_2d(hand_coordinates, ax, title=''):
    """
    Plots one hand coordinates in 2D.

    Args:
        hand_coordinates (np.array): The hand coordinates.
        ax (matplotlib.axes): The axes to plot on.
        title (str, optional): The title of the plot. 
    """
    # Plot the hand coordinates in 2D
    for i in range(len(hand_coordinates)):
        x = hand_coordinates[i][0]
        y = hand_coordinates[i][1]
        ax.scatter(x, y, color='red', marker='o')

    # Connect the points to form hand segments in 2D
    fingers = {'thumb': ([(0, 1), (1, 2), (2, 3), (3, 4)], 'blue'),
               'index': ([(0, 5), (5, 6), (6, 7), (7, 8)], 'green'),
               'middle': ([(0, 9), (9, 10), (10, 11), (11, 12)], 'red'),
               'ring': ([(0, 13), (13, 14), (14, 15), (15, 16)], 'magenta'),
               'pinky': ([(0, 17), (17, 18), (18, 19), (19, 20)], 'cyan')}

    for f in fingers:
        for segment in fingers[f][0]:
            x_segment = [hand_coordinates[segment[0]][0], hand_coordinates[segment[1]][0]]
            y_segment = [hand_coordinates[segment[0]][1], hand_coordinates[segment[1]][1]]
            ax.plot(x_segment, y_segment, color=fingers[f][1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend(handles=[mpatches.Patch(color=fingers[f][1], label=f) for f in fingers.keys()], loc='upper left')

def plot_hand_3d(hand_coordinates, ax, title='', camera_coords=False):
    """
    Plots one hand coordinates in 3D.

    Args:
        hand_coordinates (np.array): The hand coordinates.
        ax (matplotlib.axes): The axes to plot on.
        title (str, optional): The title of the plot.
        camera_coords (bool, optional): Whether to plot the camera coordinates. 
    """
    # Plot the hand coordinates
    for i in range(len(hand_coordinates)):
        x = hand_coordinates[i][0]
        y = hand_coordinates[i][1]
        z = hand_coordinates[i][2]
        ax.scatter(x, y, z, color='red', marker='o')

    # Connect the points to form hand segments
    fingers = {'thumb': ([(0, 1), (1, 2), (2, 3), (3, 4)], 'blue'),
               'index': ([(0, 5), (5, 6), (6, 7), (7, 8)], 'green'),
               'middle': ([(0, 9), (9, 10), (10, 11), (11, 12)], 'red'),
               'ring': ([(0, 13), (13, 14), (14, 15), (15, 16)], 'magenta'),
               'pinky': ([(0, 17), (17, 18), (18, 19), (19, 20)], 'cyan')}

    for f in fingers:
        for segment in fingers[f][0]:
            x_segment = [hand_coordinates[segment[0]][0], hand_coordinates[segment[1]][0]]
            y_segment = [hand_coordinates[segment[0]][1], hand_coordinates[segment[1]][1]]
            z_segment = [hand_coordinates[segment[0]][2], hand_coordinates[segment[1]][2]]
            ax.plot(x_segment, y_segment, z_segment, color=fingers[f][1])
    
    if camera_coords:
        # Add a marker at the origin
        ax.scatter(0, 0, 0, color='black', s=100, label='Origin', marker='x')

        

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    legend_handles = [mpatches.Patch(color=fingers[f][1], label=f) for f in fingers.keys()]
    if camera_coords:
        legend_handles.append(mpatches.Patch(color='black',  label='Origin'))

    ax.legend(handles=legend_handles, loc='upper left')

def plot_hand_compare(y_first, y_second, num_samples=5, dim2=False, title1=None, title2=None, same_axis=False):
    """
    Plots a comparison between two sets of hand coordinates.

    Args:
        y_first (np.array): The first set of hand coordinates.
        y_second (np.array): The second set of hand coordinates.
        num_samples (int, optional): The number of samples to plot. 
        dim2 (bool, optional): Whether to plot in 2D. Defaults to False.
        title1 (str, optional): The title of the first set plots. 
        title2 (str, optional): The title of the second set plots.
        same_axis (bool, optional): Whether to set the same axis limits for both plots. 
    """
    if dim2:
        _, axes = plt.subplots(num_samples, 2, figsize=(12, 20))
    else:
        _, axes = plt.subplots(num_samples, 2, figsize=(12, 20), subplot_kw={'projection': '3d'})

    dim = 2 if dim2 else 3

    y_first = y_first.reshape(-1, 21, dim)
    y_second = y_second.reshape(-1, 21, dim)

    true_vs_pred = title1 is None

    # Calculate the minimum and maximum values from both sets of hand coordinates

    if same_axis :
        x_min = min(y_first.min(axis=(0,1)).min(), y_second.min(axis=(0,1)).min())
        x_max = max(y_first.max(axis=(0,1)).max(), y_second.max(axis=(0,1)).max())
        y_min = min(y_first.min(axis=(0,1)).min(), y_second.min(axis=(0,1)).min())
        y_max = max(y_first.max(axis=(0,1)).max(), y_second.max(axis=(0,1)).max())
        z_min = min(y_first.min(axis=(0,1)).min(), y_second.min(axis=(0,1)).min())
        z_max = max(y_first.max(axis=(0,1)).max(), y_second.max(axis=(0,1)).max())

    for i in range(num_samples):
        if true_vs_pred: 
            epe = round(single_end_point_error(y_first[i], y_second[i]).numpy(), 2)
            title1 ='True skeleton {}'.format(i + 1)
            title2 = 'Predicted skeleton {}, epe {} : '.format((i + 1), epe)

        if dim2:
            plot_hand_2d(y_first[i], axes[i, 0], title=title1)
            plot_hand_2d(y_second[i], axes[i, 1], title=title2)
        else:
            plot_hand_3d(y_first[i], axes[i, 0], title=title1, camera_coords=True)
            plot_hand_3d(y_second[i], axes[i, 1], title=title2, camera_coords=True)
        
        # Set the same axis limits for both plots
        if same_axis :
            axes[i, 0].set_xlim(x_min, x_max)
            axes[i, 0].set_ylim(y_min, y_max)
            axes[i, 0].set_zlim(z_min, z_max)
            
            axes[i, 1].set_xlim(x_min, x_max)
            axes[i, 1].set_ylim(y_min, y_max)
            axes[i, 1].set_zlim(z_min, z_max)

        # Add a marker at the origin for 3D plots
        if not dim2:
            axes[i, 0].scatter(0, 0, 0, color='black', s=100, marker='x')
            axes[i, 1].scatter(0, 0, 0, color='black', s=100, marker='x')

    plt.tight_layout()
    plt.show()


    
def plot_cap_img(cap_image, ax=None):
    """
    Plots a capacitive image as heatmap.

    Args:
        cap_image (np.array): The capacitive image.
        ax (matplotlib.axes, optional): The axes to plot on. 
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(cap_image, cmap='viridis', cbar=True, ax=ax)
    ax.set_title('Capacitive Image Heatmap')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    plt.tight_layout()

def plot_skel_cap_img(skeleton, cap_image, title='', camera_coords=False, dim2=False):
    """
    Plots a skeleton and a capacitive image as a pair

    Args:
        skeleton (np.array): The skeleton coordinates.
        cap_image (np.array): The capacitive image.
        title (str, optional): The title of the plot. 
        camera_coords (bool, optional): Whether to plot the camera coordinates. 
        dim2 (bool, optional): Whether to plot in 2D. 
    """
    if dim2:
        _, axs = plt.subplots(1, 2, figsize=(12, 5))
        plot_hand_2d(skeleton.reshape(21, 2), axs[0], title='Skeleton: ' + title)
        plot_cap_img(cap_image, axs[1])
    else:
        fig = plt.figure(figsize=(12, 5))
        axs = [fig.add_subplot(121, projection='3d'), fig.add_subplot(122)]
        plot_hand_3d(skeleton.reshape(21, 3), axs[0], title='Skeleton: ' + title, camera_coords=camera_coords)
        plot_cap_img(cap_image, axs[1])
    
    plt.tight_layout()
    plt.show()