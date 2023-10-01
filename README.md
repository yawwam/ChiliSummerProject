## **CHILI Lab Summer project**

## **Purpose**
This project was developed as part of summer project at the CHILI Lab, EPFL. It evolves around hand skeleton reconstituion and gesture classification from standalone capactive images data. For this task, we implement a multi-task deep network, alongside visualization and evaluation tools.


## **Project Structure**

├── README.md 

├── Project_Report.pdf

├── src <br />
*  ├── data_exploration.ipynb <br />
*  ├── skel_cap_img_map_logic.ipynb <br />
*  ├── generate_data.ipynb <br />
*  ├── data_stats.ipynb <br />
*  ├── define_model.py <br />
*  ├── model_train.ipynb <br />
*  ├── model_eval.ipynb <br />
*  ├── helpers <br />
    * ├── touch_helpers.py
    * ├── data_helpers.py
    * ├── metrics_helpers.py
    * ├── plot_helpers.py

├── data <br />
*  ├── DATA2803 <br />
*  .... <br />
*  ├── DATA0304 <br />
*  ├── StudentList_isl.csv <br />
*  ├── eth_data <br />
    *  ├── P01.pkl <br />
    *  .... <br />
    *  ├── P10.pkl <br />
*  ├── processed <br />
    *  ├── native_mapped <br />
        *  ├── map_7ms <br />
            *  ├── day2803.pkl <br />
            *  .... <br />
            *  ├── day0304.pkl <br />
  
├── models <br />
*  ├── TouchPose_crossGestures <br />
*  ├── chili_models <br />
    *  ├── partialModels <br />
    *  ├── 3d_models <br />
    *  ├── 2d_models <br />
  
## **Data :**
- Chili data : Collected during a 5 days experiment involving 32 subjects. The capactive image data was recorded from a Wacom Tablet, while the hand skeleton data was collected from a leapmotion controller. The preprocessing conducted in this project results in about 80k data samples.
- TouchPose data : https://github.com/eth-siplab/TouchPose.


## **File description::**
- data_exploration.ipynb : this notbeook explores the different Chili data files.<br />
- skel_cap_img_map_logic.ipynb : this notebook explains the mapping logic applied for capactive images and skeleton data using Timestamps. <br />
- generate_data.ipynb : this script generates the preprocessed data files (day2803.pkl...day0304.pkl) to be used. 
- data_stats.ipynb : this notebook explores the data at hand distribution, and compares it to the TouchPose project data.
- define_model.py : the model definition file.
- model_train.ipynb : all the models described in the report were trained in this file.
- model_eval.ipynb : a notebook that evaluates trained models, as well as specific samples of interest using visualization tools.
- touch_helpers.py : helpers used in the TouchPose project.
- data_helpers.py : some helpers for data preprocessing and loading.
- metrics_helpers.py : file that define model metrics and evaluation tools.
- plot_helpers.py : some visualization helpers.

 

## **Set up :**
Clone the repository and download data/models folders. 


## **Authors :**
- Aouame Mohamed Yassine

## **Credits :**
eth-siplab/TouchPose : https://github.com/eth-siplab/TouchPose.
