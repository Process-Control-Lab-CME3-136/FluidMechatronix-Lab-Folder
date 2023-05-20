import OpenOPC

import matplotlib.pyplot as plt

import pywintypes
import os
import csv
import tqdm
import pandas as pd
import numpy as np
from datetime import datetime
import time
from dateutil import parser
import pytz


pywintypes.datetime = pywintypes.TimeType

GATEWAY = "192.168.1.1"
OPC_HOST = "localhost"
OPC_SERVER_NAME = "RSLinx Remote OPC Server"
CSV_FILE_NAME = "OPC_direct.csv"

# IMPORTANT READ/WRITE TAGS
PUMP_FOLDER = "[FluidMech]PumpDrivePwrFlx525"
PSI_PIDE_FOLDER = "[FluidMech]PositivePsiTransducer"
OPER_LOCK = "OCmd_AcqLock"
PUMP_START = "OCmd_Start"
PUMP_STOP = "OCmd_Stop"
PUMP_SPEED = "OSet_SpeedRef"
PUMP_FDBK = "Val_SpeedFdbk"
IS_RUNNING = "Sts_Running"
P_OUTLET = "Val"


# Response returned has server timestamp with timezone set to Aftica/Accra, converting to Edmonton timezone.
old_tz = pytz.timezone("Africa/Accra")
curr_tz = pytz.timezone("America/Edmonton")


# preserving state of all time-associated variables like exp_start, exp_end, number of SPs etc.
class ExperimentData:
    def __init__(self, opc):
        # initializing the datetime types to same value
        self.exp_cur_start = datetime.now(curr_tz)
        self.exp_cur_end = datetime.now(curr_tz)
        self.exp_prev_delta = 0.0
        self.num_sp = 0     # nunmber of set points for experiment
        self.set_time = 10  # no. of iterations per experiment

        self.first_experiment = True  # if running first experiment

        self.csv_path = get_csv_path(CSV_FILE_NAME)

        self.opc = opc

    def collect_data_auto(self):
        """
        Main loop to collect data for a list of set points.
        Ask user for set-point list and set-time i.e. number of iterations per set-point.
        Set Pump speed and collect data.

        """
        # ask user for set points (SP) and set time (ST) for each run
        sp_list = input(
            f"Enter a space-seperated list of setpoints(1-5)\n>>>").strip().split(" ")


        if not all(np.char.isdigit(sp_list)):
            print("Please enter valid integer sp values")
            return

        if not (1<= len(sp_list) <= 5):
            print("Incorrect number of values")
            return

        # ask for set_time i.e. time for each experiment
        set_time = input("Enter set time per experiment\n>>>").strip()

        if not set_time.isdigit():
            print("Incorrect input, try again...")
            return

        self.set_time = int(set_time)
        sp_list = np.array(sp_list)
        sp_list = sp_list.astype(int)

        # check if motor on, if not turn it on and set speed to sp
        while not self.is_motor_on(): self.pump_toggle()

        # Currently in append mode, thus new data will be appended at the end of csv
        with open(self.csv_path, "a") as c:
            writer = csv.writer(c, dialect="excel")

            for sp in sp_list:

                self.pump_speed_change(sp)

                print(f"Collecting Data for set point: {sp}")
                self.collect_pump_data(writer)

        # Shut down the motor after all experiment runs
        if self.is_motor_on(): self.pump_toggle()

    def collect_pump_data(self, writer):
        """
        Check if collecting new data or continuing previous experiment (self.first_experiment)
        Open csv file in write mode, read Pressure, Motor RPM data and record timestamps per iteration.
        Adjust time-stamps to minimize initial delay in recording data (exp_prev_delta)
        store data in a .csv file.
        """

        # implying first experiment, write column titles to csv and set exp_start i.e. reference time.
        if self.first_experiment:
            writer.writerow(
                ["Output_Pressure", "Pump_Measured_Speed", "Time"])

        self.exp_cur_start = datetime.now(curr_tz)
        time_stamp = ""
        max_delay = 0.5

        for _ in tqdm.tqdm(range(self.set_time)):
            t1 = time.perf_counter()

            counter = 0
            while True:
                opc_response = self.opc.read(
                    [cmd_string(PUMP_FOLDER, PUMP_FDBK), cmd_string(PSI_PIDE_FOLDER, P_OUTLET), ])

                _, pressure, _, time_stamp = opc_response[1]
                
                if counter == 0:
                    time_stamp_prev = time_stamp
                    
                if time_stamp != time_stamp_prev:
                    break
                counter += 1

            t2 = time.perf_counter()
            # p_val is a list of tuples, each tuple per read.
            # Formatted as (Description, Value, Quality, TimeStamp)

            speed = opc_response[0][1]

            # To maintain almost even time delay.
            if t2-t1 < max_delay:
                time.sleep(max_delay+t1-t2)
            else:
                max_delay = t2-t1
            
            time_stamp = datetime.replace(parser.parse(time_stamp), tzinfo=None)
            time_stamp = old_tz.localize(time_stamp).astimezone(curr_tz)
            
            if self.first_experiment:
                time_stamp = (time_stamp-self.exp_cur_start).total_seconds()
            else:
                time_stamp = (time_stamp-self.exp_cur_start).total_seconds() + \
                    self.exp_prev_delta

            writer.writerow((pressure, speed, round(time_stamp, ndigits=3)))

        if self.first_experiment:
            print(f"Data saved in : {CSV_FILE_NAME}")

        # Experiment duration added to an accumulator
        self.exp_cur_end = datetime.now(curr_tz)
        self.exp_prev_delta += (self.exp_cur_end -
                                self.exp_cur_start).total_seconds()
        self.first_experiment = False

    def pump_toggle(self):
        # Starts/Stops the pump
        start_stop = self.is_motor_on()

        # set initial speed to a safe value e.g. 10 Hz if initially set to >60 Hz
        if self.opc.read(cmd_string(PUMP_FOLDER, PUMP_FDBK))[0] > 60:
            self.pump_speed_change(10)

        # Switch OCmd_Start -> 1 and OCmd_Stop -> 0 or vice versa.
        
        self.opc.write((cmd_string(PUMP_FOLDER, PUMP_START), 1-start_stop))
        self.opc.write((cmd_string(PUMP_FOLDER, PUMP_STOP), start_stop))

    def is_motor_on(self):
        # Reads current motor speed
        # Return -> 1|0

        return self.opc.read(cmd_string(PUMP_FOLDER, IS_RUNNING))[0]
    
    def pump_speed_change(self, speed_ref=None):
        # Control Pump RPM (Hz)
        # Params:
        #       speed_ref - pump speed in Hz

        if speed_ref is None:
            speed_ref = input(
                "Enter speed in RPM(Hz) [0-60] to be safe\n>>>").strip()

            if not speed_ref.isdigit():
                print("Incorrect option entered")
                return

        speed_ref = int(speed_ref)

        if speed_ref >= 0 and speed_ref <= 60:
            tag = cmd_string(PUMP_FOLDER, PUMP_SPEED)

            self.opc.write((tag, speed_ref))
            if not self.is_motor_on():
                time.sleep(0.5)
                self.pump_toggle()

        else:
            print("Please choose a valid speed...")


def get_csv_path(file_name=CSV_FILE_NAME):

    # Return -> CSV file relative path to current directory

    parent_dir = os.getcwd()
    csv_path = os.path.join(parent_dir, file_name)
    return csv_path


def cmd_string(folder_name="", tag_name=""):
    # Params: folder_name, tag_name
    # Return -> full tag-path i.e. <Folder Name>.<Tag Name>
    if folder_name and tag_name:
        return ".".join([folder_name, tag_name])
    print("Invalid folder/file name")


def plot_pump_data():
    # Import csv data and check if csv path exists.
    # If exists, read csv data with pandas, predefined columns as the usecols parameter.

    columns = ["Output_Pressure", "Pump_Measured_Speed", "Time"]
    if not os.path.exists(get_csv_path(CSV_FILE_NAME)):
        print("No experiment data to plot")
        return

    df = pd.read_csv(get_csv_path(CSV_FILE_NAME), usecols=columns)
    plt.plot(df.Time, df.Output_Pressure,
             df.Time, df.Pump_Measured_Speed, marker="+")
    plt.xlabel("Time in seconds")
    plt.title("Plot of Pump speed and Outlet Pressure v/s Time")
    plt.legend(["Outlet Pressure", "Pump Speed  "])
    plt.show()


def pump_control_cli(opc):

    exp_data = ExperimentData(opc)
    # on initialization remove all previous data
    if os.path.exists(exp_data.csv_path):
        os.remove(exp_data.csv_path)

    # Lock the motor in operator mode for open-loop testing
    exp_data.opc.write((f"{PUMP_FOLDER}.{OPER_LOCK}", 1))

    while True:

        inp = input("""
                    Enter Option:
                    1. Start/stop
                    2. Speed change 
                    3. Collect data 
                    4. Plot data
                    5. Start new experiment
                    6. Exit
                    >>>""").strip()
        if not inp.isdigit():
            print("Incorrect Option\n")
            continue

        option = int(inp)
        if option == 1:
            exp_data.pump_toggle()

        elif option == 2:
            exp_data.pump_speed_change()

        elif option == 3:

            exp_data.collect_data_auto()

        elif option == 4:
            plot_pump_data()

        elif option == 5:

            if os.path.exists(exp_data.csv_path):
                print("Deleting previous experiment results...")
                os.remove(exp_data.csv_path)
                exp_data.first_experiment = True
            else:
                print("No experiment data exitst")
            
            #reset exp_data instance
            exp_data = ExperimentData(opc)

        elif option == 6:
            # close connection and exit
            print("Exiting...")

            exp_data.opc.write((cmd_string(PUMP_FOLDER, PUMP_STOP), 1))
            exp_data.opc.close()

            exit()
        else:
            print("Incorrect Option, Try again")
            continue


if __name__ == "__main__":
    opc = OpenOPC.client()

    opc.connect(OPC_SERVER_NAME, OPC_HOST)
    print("Available Servers:")
    for serv in opc.servers():

        print(serv)
    print()

    pump_control_cli(opc)
