"""
    All global tag constants defined here
"""

TAGS = {
    '[FluidMech]PumpDrivePwrFlx525': [
        "OCmd_AcqLock", "OCmd_Start", "OCmd_Stop", "OSet_SpeedRef", "Val_SpeedFdbk", "Sts_Running", "OCmd_Unlock", "OCmd_AcqLock"
        ],

    "[FluidMech]PositivePsiTransducer": [
        "Val"
        ],

}


ENDPOINT = "opc.tcp://admin@localhost:4843/freeopcua/server"
APPLICATION_NAME = 3
SERVER_NAMESPACE_URI = 2
SERVER_NODE_ID = 'ns=2;i=1'
METHOD_ID = '2:get_status'


TAG_ID = 'ns=2;i=1075'
PUMP_FOLDER = list(TAGS.keys())[0]
PRESSURE_FOLDER = list(TAGS.keys())[1]

PRIME_PUMP = '[FluidMech]Local:1:O.Data'
PUMP_SPEED = ".".join([PUMP_FOLDER, TAGS[PUMP_FOLDER][3]])
OUTPUT_PRESSURE = ".".join([PRESSURE_FOLDER, TAGS[PRESSURE_FOLDER][0]])
OPER_MODE = ".".join([PUMP_FOLDER, TAGS[PUMP_FOLDER][7]])
PROG_MODE = ".".join([PUMP_FOLDER, TAGS[PUMP_FOLDER][6]])

MV_LOW = 0.0
MV_HIGH = 100.0

SP_LOW = 0.0
SP_HIGH = 10.

STEPS = 10