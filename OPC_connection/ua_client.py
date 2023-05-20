"""
python OPC-UA client 
reusing functionality
steps:
1. run opcda-to-opcua.py proxy
2. read/write data from proxy
3. first implement reading functionality

define an opc class with .read() and .write() methods that wrap opc ua methods 

"""

from OPC_direct_readwrite import pump_control_cli, ExperimentData
from opcua import Client, ua
from OPCWrapper import OPCWrapper

# TAGS
PUMP_FOLDER = "[FluidMech]PumpDrivePwrFlx525"
PSI_PIDE_FOLDER = "[FluidMech]PositivePsiTransducer"
OPER_LOCK = "OCmd_AcqLock"
PUMP_START = "OCmd_Start"
PUMP_STOP = "OCmd_Stop"
PUMP_SPEED = "OSet_SpeedRef"
PUMP_FDBK = "Val_SpeedFdbk"
IS_RUNNING = "Sts_Running"
P_OUTLET = "Val"


OPC_DA_SERVER = 'RSLinx Remote OPC Server'
client = Client("opc.tcp://admin@localhost:4843/freeopcua/server", timeout = 36000)
uri = "http://examples.freeopcua.github.io"




try:
    # connect to client, get server namespace index and root node
    client.connect()
    idx = client.get_namespace_index(uri)
    root = client.get_root_node()

    # for testing purposes
    opc = OPCWrapper(client, uri)
    # print(opc.read(cmd_string(PUMP_FOLDER, PUMP_FDBK)))
    
    # print(is_motor_on(opc))
    pump_control_cli(opc)

except Exception as e:
    print(e)

finally:
    client.close_session()
    exit()
