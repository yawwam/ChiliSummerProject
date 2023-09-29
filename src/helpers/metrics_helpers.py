import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


"""
********************************************************************************************************************** 
Model loss and metric functions
**********************************************************************************************************************
"""

def skeleton_loss(y_skeleton_true, y_skeleton_pred):
    """
    Calculates the weighted mean squared error between the predicted and true joint coordinates.

    Args:
        y_skeleton_true (tensor): True joint coordinates of shape (batch_size, 63 or 42)
        y_skeleton_pred (tensor): Predicted joint coordinates of shape (batch_size, 63 or 42)

    Returns:
        mse (tensor): Mean squared error between the predicted and true joint coordinates
        
    """
    J = tf.constant(21.0)  # number of predicted joints
    mse = tf.reduce_sum(tf.square(y_skeleton_true - y_skeleton_pred),axis=1) / (J * 3)
    return mse


def un_skeloss(mean, std):
    """
    Normalized skeleton loss by scaling the predicted and true joint coordinates with the mean and standard deviation.
    """
    
    def skeleton_loss(y_skeleton_true, y_skeleton_pred) :
        y_skeleton_true = y_skeleton_true * std + mean
        y_skeleton_pred = y_skeleton_pred * std + mean
        J = tf.constant(21.0)  # number of predicted joints
        mse = tf.reduce_sum(tf.square(y_skeleton_true - y_skeleton_pred),axis=1) / (J * 3)
        return mse

    return skeleton_loss

    
def NCE(y_gesture_true, y_gesture_pred):
    """
    Attempt to compute the normalized cross entropy loss between the predicted and true gesture labels.
    """
    epsilon = 1e-7
    y_gesture_pred = tf.clip_by_value(y_gesture_pred, epsilon, 1 - epsilon)
    y_gesture_true = tf.clip_by_value(y_gesture_true, epsilon, 1 - epsilon)
    cross_entropy_loss = -tf.reduce_sum(y_gesture_true * tf.math.log(y_gesture_pred), axis=-1)
    min_loss = tf.reduce_min(cross_entropy_loss)
    max_loss = tf.reduce_max(cross_entropy_loss)
    # Normalize the loss to the range [0, 1]
    normalized_loss = (cross_entropy_loss - min_loss+ epsilon) / (max_loss - min_loss+ epsilon)
    print(normalized_loss)
    return tf.reduce_mean(normalized_loss)
        
def unw_loss(mean, std, alpha, beta, dim2=False):
    """
    Normalized weighted loss by scaling the predicted and true joint coordinates with the mean and standard deviation.

    Args:
        mean (np array or scalar): Mean of the training data
        std (np array or scalar): Standard deviation of the training data
        alpha (float): Weight for the gesture loss
        beta (float): Weight for the skeleton loss
        dim2 (bool): Whether to use the 2D or 3D skeleton loss
    
    Returns:
        w_loss (tensor): Normalized weighted loss taking into account gesture and skeleton loss
    """
    
    
    def w_loss(y_true,y_pred):
        edge = 42 if dim2 else 63
        true_gesture, pred_gesture = y_true[:,edge:], y_pred[:,edge:] 
        true_skeleton, pred_skeleton = y_true[:,:edge], y_pred[:,:edge]
        true_skeleton, pred_skeleton = true_skeleton * std + mean, pred_skeleton * std + mean
        sk_loss_scaled = tf.reduce_mean(skeleton_loss(true_skeleton, pred_skeleton))
        #sk_loss_scaled = tf.reduce_mean((sk_loss - tf.reduce_min(sk_loss)) / (tf.reduce_max(sk_loss) - tf.reduce_min(sk_loss) ))
        ccr = tf.keras.losses.CategoricalCrossentropy()
        gesture_loss = ccr(true_gesture, pred_gesture)
        loss = alpha*gesture_loss+ (beta*sk_loss_scaled)
        return loss
        
    return w_loss

def end_point_error(y_true, y_pred, dim2=False):
    """
    Calculates the mean end point error between the predicted and true joint coordinates.

    Args:
        y_true (tensor): True output (batch_size, 67 or 46)
        y_pred (tensor): Predicted output (batch_size, 67 or 46)

    Returns:
        mean_epe (tensor): Mean end point error between the predicted and true joint coordinates
    """
    # Calculate the Euclidean distance between predicted and true joint coordinates
    edge = 42 if dim2 else 63
    y_skeleton_true, y_skeleton_pred = y_true[:,:edge], y_pred[:,:edge]
    euclidian_distances = tf.norm(y_skeleton_true - y_skeleton_pred, axis=1)
    mean_epe = tf.reduce_mean(euclidian_distances)
    return mean_epe

def single_end_point_error(y_true,y_pred, dim2=False):
    """
    Calculates the end point error between the predicted and true joint coordinates for one sample.
    """
    edge = 42 if dim2 else 63
    y_skeleton_true, y_skeleton_pred = y_true[:,:edge], y_pred[:,:edge]
    return tf.norm(y_skeleton_true - y_skeleton_pred)

def un_epe(mean, std, dim2=False):
    """
    Normalized end point error by scaling the predicted and true joint coordinates with the mean and standard deviation.
    """

    def end_point_error(y_true, y_pred):
        edge = 42 if dim2 else 63
        y_skeleton_true, y_skeleton_pred = y_true[:,:edge], y_pred[:,:edge]
        y_skeleton_true = y_skeleton_true * std + mean
        y_skeleton_pred = y_skeleton_pred * std + mean
        euclidian_distances = tf.norm(y_skeleton_true - y_skeleton_pred, axis=1)
        mean_epe = tf.reduce_mean(euclidian_distances)
        return mean_epe

    return end_point_error


def auc_pck(y_true, y_pred, dim2=False):
    """
    Calculates the area under the curve of the percentage of correct keypoints (PCK) for different thresholds.

    Args:
        y_true (tensor): True output (batch_size, 67 or 46)
        y_pred (tensor): Predicted output (batch_size, 67 or 46)

    Returns:
        pck_values (tensor): Area under the curve of the percentage of correct keypoints (PCK) for different thresholds
    """
    edge = 42 if dim2 else 63
    y_skeleton_true, y_skeleton_pred = y_true[:,:edge], y_pred[:,:edge]
    thresholds = np.arange(20, 51, 5)  # Thresholds from 20 mm to 50 mm
    pck_values = []


    # Calculate the Euclidean distance between predicted and true joint coordinates
    distances = tf.sqrt(tf.reduce_sum(tf.square(y_skeleton_true - y_skeleton_pred), axis=-1))

    # Calculate the percentage of correct keypoints for each threshold
    for threshold in thresholds:
        correct_keypoints = tf.cast(distances < threshold, tf.float32)
        pck = tf.reduce_mean(correct_keypoints) * 100.0
        pck_values.append(pck)

    return tf.convert_to_tensor(pck_values, dtype=tf.float32)


def un_pck(mean, std, dim2=False):
    """
    Normalized area under the curve of the percentage of correct keypoints (PCK) for different thresholds by scaling the predicted and true joint coordinates with the mean and standard deviation.
    """

    def auc_pck(y_true, y_pred):
        edge = 42 if dim2 else 63
        y_skeleton_true, y_skeleton_pred = y_true[:,:edge], y_pred[:,:edge]
        y_skeleton_true = y_skeleton_true * std + mean
        y_skeleton_pred = y_skeleton_pred * std + mean
        thresholds = np.arange(20, 51, 5) 
        pck_values = []

        distances = tf.sqrt(tf.reduce_sum(tf.square(y_skeleton_true - y_skeleton_pred), axis=-1))

        for threshold in thresholds:
            correct_keypoints = tf.cast(distances < threshold, tf.float32)
            pck = tf.reduce_mean(correct_keypoints) * 100.0
            pck_values.append(pck)

        return tf.convert_to_tensor(pck_values, dtype=tf.float32)
    
    return auc_pck
    
def compute_acc(dim2=False) :
    """
    Computes the accuracy of the gesture classification.
    """
    
    def gesture_acc(y_true,y_pred) :
        edge = 42 if dim2 else 63
        y_gesture_true, y_gesture_pred = y_true[:,edge:], y_pred[:,edge:]
        correct_predictions = tf.equal(tf.argmax(y_gesture_true, axis=-1), tf.argmax(y_gesture_pred, axis=-1))
        
        # Convert the boolean values to floats and calculate the mean
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy
        
    return gesture_acc

def compute_f1_score(dim2=False) :
    """
    Computes the f1 score of the gesture classification.
    """

    def gesture_f1_score(y_true,y_pred):
        edge = 42 if dim2 else 63
        y_gesture_true, y_gesture_pred = y_true[:,edge:], y_pred[:,edge:]
        # Define the true positives, false positives and false negatives
        tp = K.sum(K.round(K.clip(y_gesture_true * y_gesture_pred, 0, 1)))
        fp = K.sum(K.round(K.clip(y_gesture_pred - y_gesture_true, 0, 1)))
        fn = K.sum(K.round(K.clip(y_gesture_true - y_gesture_pred, 0, 1)))
    
        # Calculate the precision and recall
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
    
        # Calculate the F1 score
        f1_score = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
        return f1_score
    
    return gesture_f1_score

"""
********************************************************************************************************************** 
Model evaluation functions
**********************************************************************************************************************
"""
    
    
def sorted_joint_end_point_error(y_skeleton_true,y_skeleton_pred, mean, std) :
    """
    Computes the mean end point error for each joint and sorts them in descending order.

    Args:
        y_skeleton_true (tensor): True joint coordinates of shape (batch_size, 63 or 42)
        y_skeleton_pred (tensor): Predicted joint coordinates of shape (batch_size, 63 or 42)
        mean (np array or scalar): Mean of the data
        std (np array or scalar): Standard deviation of data

    Returns:
        max_index (tensor): Values and indices of the sorted mean end point error for each joint
    """
    y_skeleton_true = y_skeleton_true*std +mean
    y_skeleton_pred = y_skeleton_pred*std +mean
    #take mean root squared error for all joints
    euclidian_distances= tf.reduce_mean(tf.sqrt(tf.square(y_skeleton_true-y_skeleton_pred)),axis=0)
    max_index = tf.math.top_k(euclidian_distances,k=len(euclidian_distances))
    return max_index

def sorted_sample_end_point_error(y_skeleton_true, y_skeleton_pred, mean, std) :
    """
    Computes the mean end point error for each sample and sorts them in descending order.

    Args:
        y_skeleton_true (tensor): True joint coordinates of shape (batch_size, 63 or 42)
        y_skeleton_pred (tensor): Predicted joint coordinates of shape (batch_size, 63 or 42)
        mean (np array or scalar): Mean of the data
        std (np array or scalar): Standard deviation of the data

    Returns:    
        max_index (tensor): Values and indices of the sorted mean end point error for each sample
    """
    y_skeleton_true = y_skeleton_true*std +mean
    y_skeleton_pred = y_skeleton_pred*std +mean
    #equivalent to tf.sqrt(tf.reduce_sum(tf.square(y_skeleton_true-y_skeleton_pred),axis=1))
    euclidian_distances = tf.norm(y_skeleton_true - y_skeleton_pred,axis=1)
    max_index = tf.math.top_k(euclidian_distances,k=len(euclidian_distances))
    return max_index

def evaluate(model, data, mean, std, dim2=False) :
    """
    Evaluates the model on the test and validation set.

    Args:
        model (keras model): Trained model
        data (dict): Dictionary containing the training, validation and test data
        mean (np array or scalar): Mean of the training data
        std (np array or scalar): Standard deviation of the training data
        dim2 (bool): Whether to use the 2D or 3D skeleton loss
    
    Returns:
        epe_max (tensor): Mean end point error and indices for each joint sorted in descending order
    """
    edge = 42 if dim2 else 63
    X_test, X_valid, y_test, y_valid = data['test']['X'], data['valid']['X'], data['test']['y'], data['valid']['y']
    y_out = model.predict(X_test, batch_size=32)
    y_vout = model.predict(X_valid, batch_size=32)
    epe_max=0
    for name,y_pred,y_true in [('Testing',y_out,y_test),('Validation',y_vout,y_valid)] :
        print(f'{name} Evaluation')
        auc = un_pck(mean, std, dim2=dim2)(y_true, y_pred).numpy()
        epe = un_epe(mean, std, dim2=dim2)(y_true, y_pred).numpy()
        if name == 'Testing':
            y_test_skeleton, y_out_skeleton = y_true[:,:edge], y_pred[:,:edge]
            epe_max = sorted_joint_end_point_error(y_test_skeleton,y_out_skeleton, mean, std)
        gesture_accuracy = compute_acc(dim2=dim2)(y_true, y_pred).numpy()
        gesture_f1 = compute_f1_score(dim2=dim2)(y_true, y_pred).numpy()
        print("Gesture Accuracy: ", gesture_accuracy)
        print("Gesture F1 Score: ", gesture_f1)
        print("AucPCK : ", auc)
        print("End point error ", epe)
        print('*'*50)
    return epe_max

    
