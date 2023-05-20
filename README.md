# FLUIDMechatronix™: Connectivity, Identification, and Advanced Control via Python (Updated: April 30, 2023)

Python control and connectivity algorithms, simulations, and system identifiers for the [FLUIDMechatronix™](https://www.turbinetechnologies.com/educational-lab-products/pump-lab-with-automation) process automation and pumps teaching system, developed by [_Khush Harman Singh Sekhon_](https://github.com/ginni0002), Purushottama Dasari, Om Prakash, and and [_Oguzhan Dogru_](https://github.com/oguzhan-dogru). Contact [dogru@ualberta.ca](mailto:dogru@ualberta.ca?subject=[GitHub]%20Fluid%20Mechatronix) for the details/datasets.

## Codes: 

### OPC_connection:
  - **OPCWrapper.py** : Wrapper class for the basic opc methods read/write/close for OPC_direct_readwrite.py to work with OPC-UA.
  - **OPC_direct_readwrite.py** : Helper functions used to perform read/write operations with OpenOPC, store and plot data for Fluid Mechatronix experiment.
  - **OPC_expert_readwrite.py** : Helper functions used to perform read/write with OPC Expert and requests module, store and plot data for Fluid Mechatronix experiment.
  - **opcda-to-opcua.py** : Modified proxy to convert opc-da tag data (in 32-bit python env) to opc-ua tag data (in 64-bit python env). The [**original proxy**](http://courses.compute.dtu.dk/02619/software/opcda_to_opcua.py) was created by [**@Eskild Schroll-Fleischer**](https://github.com/eskildsf)
  - **ua_client.py** : A working example for an opc-ua client in python-64-bit.

### State Space Modelling:
  - **State_space_modelling_ex.ipynb** : Working example of state space model implemented in python
  - **gif_plot.ipynb** : A plotting function to create a gif plot from state space data saved as a csv.
  - **animated_plot.py** : A plotting function to create a gif plot with matlab animation module. (faster than gif_plot.py)
  - **pid_control.py** : PID controller (position and velocity) classes for controlling a simulated state space environment.
  - **plotter.py** : Online plotting function to plot data in real-time. Works with both simulated and experimental environments.
  - **online_plot.py** : stand-alone online plotter function (older version of plotter.py), redundant since plotter.py was made.

### System Identification:
  - **sysID_modellign.m** : A compilation of multiple system models for Fluid Mechatronix in matlab.
    1. FOPTD and SOPTD 
    2. SOPTD with custom parameters
    3. State Space 
    4. Linear ARX
    5. Non-Linear ARX
    6. First Principles (static)
  - **pysindy.ipynb** : pySINDy (Sparse Indetification of Non-Linear Dynamics) based SISO and SIMO models for Fluid Mechatronix in python.
  - **rgs_signal.m** : Function to generate RGS signal in matlab.

### Automation [In Progress]:
  - **constants.py** : Stores all global constants used by fsm_states.py
  - **fsm_states.py** : Finite state machine based model, will be used in conjunction with RL based agents to automate Fluid Mechatronix lab machine.
  - **plotter.py** : Almost identical to the pre-existing plotter.py in State_space_modelling/ directory with minor changes.

### RL Agents [In Progress]:
  - **Qlearning.ipynb** : Q learning based agent for simulated environment (based on Fluid Mechatronix lab machine).
  - **A2C.py** : Advantage Actor Critic agent trained on simulated environment.
  - **A3C.py** : Asynchronous Advantage Actor Critic agent trained on simulated environment for 3000 episodes.
  - **A2C/A3C Agents** : Actor and Critic agent weights (as .h5 files)
## Data:

### Plots and Data:

Numerical data is saved as 'csv/mat' files and plots as 'pdf'/'png/gif' files.

### sysID_data:
  - **0SysID_trainingData.mat**: training data for system identification. (Originally created by Bowen Liu)
  - **0SysID_testingData.mat**: testing data for system identification. (Originally created by Bowen Liu)
  - **0SysID_SignalData.mat**: RGS signal data used to excite the system. 
  - train_data, test_data and simo_data are csv files with identical data as previously mentioned sysID .mat files.
  
### State Space Modelling:
  - **Sim_data.csv** : Simulation data generated from State_space_modelling_ex.ipynb.
  - **eg.gif** : Gif plot generated from either gif_plot.ipynb or animated_plot.py.
  
### img_refs:
Contains images and figures used in the github repo.

### Miscellaneous 
  - **state_machine_diagram.png** : Details the finite state machine structure used in fsm_states.py. (subject to change)
  - **state_transition_table_automation.xlsx** : State transitions in detail. (subject to change)
  - **FM_tags.xlsx**: All tag data for Fluid Mechatronix obtained using OpenOPC.
