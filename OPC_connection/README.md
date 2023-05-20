# **OPC Connection using Python with: OpenOPC / FreeOpcUA / OPC Expert**
Connecting with Fluid Mechatronix system, read/write and store data, made by:

[**Khush Harman**](https://github.com/ginni0002), [**Oguzhan Dogru**](https://github.com/oguzhan-dogru), [**Om Prakash**](https://github.com/prakashoms), [**Purushottama Rao Dasari**](https://github.com/Purush-IITMadras)

## Files:
1. **OPCWrapper.py** : Wrapper class for the basic opc methods read/write/close.
2. **OPC_direct_readwrite.py** : Helper functions used to perform read/write with OpenOPC.
3. **OPC_expert_readwrite.py** : Helper functions used to perform read/write with OPC Expert and requests module.
4. **OpenOPC.py** : OpenOPC library module for python, used to connect to OPC-DA server using python.
5. **README.md** : Documentation on how to use these files.
6. **opcda-to-opcua.py** : Proxy to convert opc-da tag data (in 32-bit python env) to opc-ua tag data (in 64-bit python env).
7. **requirements.txt** : all packages required for this set-up.
8. **ua_client.py** : A working example for an opc-ua client running on python-64-bit.

## Description:
The original [OpenOPC library](https://openopc.sourceforge.net/about.html) developed by @**Barry Barnreiter**(barry_b@users.sourceforge.net) used to connect any OPC DA Server via win32 COM connection.
The original library only worked for **python 2.7** but many independent developers like [**@Josea Maita**](https://github.com/joseamaita/openopc120)(2017) and [**@Prof. Anton D. Kachalov**](https://github.com/ya-mouse)(2014)
built wrapper modules to port it to **python 3.7** and **python 3.4+** respectively.


The major issue even with these libraries is that only certain combinations of python versions and windows work:
|**Combination**|**Description**|**Disadvantages**|
|:------------------:|:------------------------------------------------------------------------------------------------------------------------:|:----------------:|
|**[Windows 10] + [Python 2.7]**|This is the [original scheme](https://openopc.sourceforge.net/) that can be run directly with OpenOPC library without any additional software but python2.7 doesn’t have LTS – long term support (ended in 2020) and most of the libraries needed for RL modelling are not compatible with version 2.7|No TF2 support and no LTS support for python|
|**[Windows 10] + [Python 3.7(32-bit)]**|An [updated implementation](https://github.com/j3mg/openopc/) of OPC for python 3.7+ 32-bit. Since it is python 3.x, newer libraries can be used for coding RL models in the future. The only issue being the 32-bit version as it decreases performance compared to 64-bit version and tensorflow does not support the 32-bit version.|No TF2 support, computationally slower and takes more memory|
|**[Windows 10] + [Python 3.4(32-bit)]**|Similar to the previous combination with the same fallbacks.|No TF2 support, computationally slower and takes more memory|
|**[Windows 10] + [Python 3.7(64-bit)]**|There is not an official library for this combination but a DLL wrapper by [Graybox](http://gray-box.net/daawrapper.php) (who are responsible for many other OPC software products). This combination does not allow read/write of tag values.|None of the above|

\* (TF2 - TensorFlow 2)

Our intended combination is **[Win10] + [Python 3.7 64-bit]**, the workaround to achieve this is either using **OPC Expert** to create a tunneler and
pass all the opc-da server data through a proxy to opc-ua client **OR** by creating a **custom** proxy (in python in our case).

The [**original proxy**](http://courses.compute.dtu.dk/02619/software/opcda_to_opcua.py) was created by [**@Eskild Schroll-Fleischer**](https://github.com/eskildsf) and modified by us to work with our server implementation.

### Proxy functions and their brief description:
|**Function Name**|**Description**|
|------------------------------------|--------------------------------------------------------------------------------------------------------|
|[`class SubscriptionHandler(object):`](https://github.com/ginni0002/Khush_Main/blob/60769826edbe6ae11326820a9dac6c3c68adecb7/OPC_connection/opcda-to-opcua.py#L77)|As specified in the [OPC UA specification](https://opclabs.doc-that.com/files/onlinedocs/QuickOpc/Latest/User%27s%20Guide%20and%20Reference-QuickOPC/Subscribing%20to%20OPC%20UA%20Data%20Changes.html), using a subscription handler we can subscribe to data changes and catch datachange notifications|
|[`def read_value(value):`](https://github.com/ginni0002/Khush_Main/blob/60769826edbe6ae11326820a9dac6c3c68adecb7/OPC_connection/opcda-to-opcua.py#L106)|Reads the tag value and returns `float(value)` if in decimal format, `value = None` otherwise|
|[`for node in tqdm.tqdm(nodes):`](https://github.com/ginni0002/Khush_Main/blob/60769826edbe6ae11326820a9dac6c3c68adecb7/OPC_connection/opcda-to-opcua.py#L129)|Loops over all nodes i.e. list of tag-path+tag-name for all tags, splits the tag-path and tag-name and creates a folder tree for each tag-path i.e. `tree = {folder1: tag1, folder2: tag2 ...}`. Then it seperates readable and writable variables i.e. tags according to the read/write property. |
<br>
<p align="center">
  <img alt="proxy server layout" src="https://github.com/ginni0002/Khush_Main/blob/a3a6f09dd8993cac1f7b6bdf826c482ae60c2544/img_refs/opc_proxy_congif.png">
</p>


The basic principle is that the proxy does not convert COM/DCOM communication to adhere to TCP/IP standard but instead, instantiates a UA server on the client side and transfers the namespace from the DA server to the newly created UA server which requires 5-10 mins to parse the ~5000 tags in the namespace.
The modifications to the proxy involve the **.properties()** method for OpenOPC client as currently this method returns invalid data.<br>

### Original:
```python
for id, description_of_id, value in c.properties(node):
  if id is ITEM_ACCESS_RIGHTS:
     if value == 'Read':
      value = ACCESS_READ
    elif value == 'Write':
      value = ACCESS_WRITE
    elif value == 'Read/Write':
      value = ACCESS_READ_WRITE
```
### Modified:
```python
value = c.properties(node, id=5)

    if value == 'Read':
        value = ACCESS_READ
    elif value == 'Write':
        value = ACCESS_WRITE
    elif value == 'Read/Write':
        value = ACCESS_READ_WRITE

    node_obj[ITEM_ACCESS_RIGHTS] = value

    node_obj[ITEM_VALUE], node_obj[ITEM_QUALITY], node_obj[ITEM_TIMESTAMP] = c.read(node)

    node_obj[ITEM_CANONICAL_DATATYPE] = c.properties(node, id=1)
```


There are some other changes as well like changing the following on line 101:
```
path_as_string = node.get_path_as_string()
```
with:
```
path_as_string = node.get_path(as_string=True)
```

More information on common issues encountered while setting up the proxy are mentioned [here](https://github.com/FreeOpcUa/python-opcua/issues/489)

## Enviornment Set-up/Installation

## 1. OpenOPC set-up

Download the original OpenOPC installer from [here](https://sourceforge.net/projects/openopc/files/). Note that all the options should be **ticked on**.
<p align="left">
  <img alt="OpenOPC installer screenshot" src="https://github.com/ginni0002/Khush_Main/blob/main/img_refs/installer.png">
</p>

After the installation is complete, install the **Gateway-Service** which handles all the COM/DCOMM calls used by OpenOPC library.
```
C:\OpenOPC\bin> OpenOPCService.exe -install
Installing service OpenOpcService
Service installed
```

To run the service, type the following:
```
C:\OpenOPC\bin> net start zzzOpenOpcService
```
Similiarly to stop the service:
```
C:\OpenOPC\bin> net stop zzzOpenOpcService
```
To manually check if the service is running, use task manager:
<p align="left">
  <img alt="taskmanager service screenshot" src="https://github.com/ginni0002/Khush_Main/blob/main/img_refs/service_taskmngr.png">
</p>

All the opc cli commands as well as how to set-up the enviornment variables is mentioned in the original docs (by @Barry Barnreiter)[here](https://github.com/ginni0002/Khush_Main/blob/main/docs/OpenOPC%20Notes%20-%20OpenOPC.pdf).

## 2. Proxy set-up


The **opcda-to-opcua.py** proxy is run on **[Windows 10] + [Python 3.7(32-bit)]** while the client uses **[Windows 10] + [Python 3.7(64-bit)]**.

The working principle has been explained in eariler sections thus without reiterating any points, here are the steps to install it.

1. Firstly, If you have Anaconda or pycharm installed (, even the python `virtualenv` package works) set-up two enviornments and install [Python 3.7 32-bit](https://www.python.org/ftp/python/3.7.0/python-3.7.0.exe) in one and [Python 3.7 64-bit](https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe) in another and run the following command in both of them.

      ```
      pip install requirements.txt
      ```

2. Run a command-line instance in each enviornment. To check if the correct python version is running, simply start the python cli by typing `python` in the cmd terminal.
  The version information is shown here:
  
  ### Python 32-bit
  
  ![python terminal 32-bit](https://github.com/ginni0002/Khush_Main/blob/main/img_refs/py-64bit.png)
  
  ### Python 64-bit
  
  ![python terminal 64-bit](https://github.com/ginni0002/Khush_Main/blob/main/img_refs/py-32bit.png)
  
3. Now in the 32-bit virtual-environment type
      ```
      python opcda-to-opcua.py
      ```
  and wait for the proxy to initialize all the tags. (make sure the Fluid Mechatronix machine is running)
  After initialization is complete, it will be listening on **port 4843** by default (port can be changed in the python file itself but not recommended). 

4. The opcua client gui can be used to look at the server namespace for testing purposes.

<p align="center">
  <img alt="opcua client screenshot" src="https://github.com/ginni0002/Khush_Main/blob/main/img_refs/opcua-client.png">
</p>

  To install the client-gui follow [these steps](https://github.com/FreeOpcUa/opcua-client-gui).

5. Now, in the 64-bit environment run your own opcua client. The existing  **ua_client.py** can be used as an example.

  **NOTE**: If the opcua client uses any methods from **OPC_direct_readwrite.py**, import only the [**ExperimentData**](https://github.com/ginni0002/Khush_Main/blob/bd79958580b7599b895fad78269baca8b293ccae/OPC_connection/OPC_direct_readwrite.py#L49) class and pass the [**OPCWrapper**](https://github.com/ginni0002/Khush_Main/blob/e6faa587d30e4281e0a775e0d00511bc375c3b4a/OPC_connection/OPCWrapper.py#L8) instance to the class. Then all OPC_direct_readwrite.py methods are accessible.

## 3. OPC-direct and OPC-Expert set-up

### OPC Direct read/write

OPC direct refers to **OPC_direct_readwrite.py** which uses the OpenOPC library in python 32-bit to connect to the remote server using opc-da communication standard while OPC-Expert i.e. **OPC_expert_readwrite.py** uses OPC Expert tunneler software and requests module to connect to the server but instead of DA standard it uses UA.
### Steps to run OPC_direct_readwrite.py:
1. Run a cmd terminal in the python 32-bit environment set-up in [step 2](https://github.com/ginni0002/Khush_Main/edit/main/OPC_connection/README.md#2-proxy-set-up).
2. run the following command (make sure the Fluid Mechatronix machine is running)
```
python OPC_direct_readwrite.py
```
3. A cli interface will be displayed.

<p align="left">
  <img alt="OPC_direct_readwrite.py screenshot" src="https://github.com/ginni0002/Khush_Main/blob/main/img_refs/OPC_direct_cli.png">
</p>

4. Manually start/stop or change the speed of the pump by commands **1** and **2** respectively.

    To collect data for plotting, type **3** and enter the **set points** referring to pump speed in Hz and then the **set-time** which refers to time between consecutive pump speeds.
    If set-time is 10 then 10 data points will be collected for each set-point.


5. Plot the collected data by typing 5 in the main window of the cli.
<p align="center">
  <img alt="OPC_direct_plotted data" src="https://github.com/ginni0002/Khush_Main/blob/main/img_refs/Pressure_Speed_plt.png">
</p>

6. The data is also stored as a csv file **OPC_direct.csv** which can be used for further analysis.
    
    **Note** that whenever OPC_direct_readwrite.py is started, any previous data in that csv is deleted.


### OPC Expert read/write

For running **OPC_expert_readwrite.py**, firstly make sure that the OPC Expert software is installed on the PC. OPC Expert can be downloaded from [this link](https://opcexpert.com/).

After installing and activating the software, run OPC Expert and then open a cmd terminal and run the following command:
```
python OPC_Expert_readwrite.py
```

**OPC_expert_readwrite.py** has almost similiar structure to **OPC_direct_readwrite.py** except for the helper function [**request_builder**](https://github.com/ginni0002/Khush_Main/blob/e6faa587d30e4281e0a775e0d00511bc375c3b4a/OPC_connection/OPC_expert_readwrite.py#L138) and the fact that most of the functions that involve read/write of data are inside the **ExperimentData** class in case of **OPC direct**.
The reason for that being the wrapper class for opc (OPCWrapper.py) wraps all the functions inside **OPC_direct_readwrite.py** to make it accessible via any opcua client.








