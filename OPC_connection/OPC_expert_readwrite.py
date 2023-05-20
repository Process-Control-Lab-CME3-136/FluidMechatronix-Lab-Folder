import requests
import json
import os
from datetime import datetime
import time
import pytz
import csv
import tqdm
import dateutil.parser
import pandas as pd
import matplotlib.pyplot as plt

CSV_FILE_NAME = "OPC_expert.csv"
CSV_COLUMNS = ["Output_Pressure", "Pump_Measured_Speed", "Time"]

# Read/Write URL
REQUEST_ENDPOINT = "https://localhost/{}?computer=192.168.1.40"
SERVER_URL = "&item=RSLinx%20Remote%20OPC%20Server->"

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

# JSON Object tag constants
JSON_DATA = "Data"
JSON_TAG_ID = 0
JSON_TAG_VALUE = "Value"
JSON_TAG_TIMESTAMP = "ServerTimestamp"

# Response returned has server timestamp with timezone set to Aftica/Accra, converting to Edmonton timezone.
old_tz = pytz.timezone("Africa/Accra")
curr_tz = pytz.timezone("America/Edmonton")

# Experiment time variables
exp_cur_start = datetime.now(curr_tz)
exp_cur_end = exp_cur_start
exp_prev_delta = 0.0


# This implementation will be simpler in design compared to OPC_direct_readwrite as reimplementing
# everything is not beneficial.

def get_set_points():
    # get set points from user
    set_points = input("Enter set points (space seperated)\n>>>").strip()
    if set_points and all(i.isdigit() for i in set_points.split(" ")):
        set_points = [int(i) for i in set_points.split(" ")]
        for i in set_points:
            if not 0 <= i <= 60:
                print("Invalid data points entered, please try again")
                return
    else:
        print("Please enter a valid list")
        return

    set_time = input("Enter number of data points per experiment\n>>>").strip()
    if set_time.isdigit():
        set_time = int(set_time)
    else:
        print("Enter valid set time")
        return

    # run the experiment, save the data
    with open(get_csv_path(), "a") as c:
        writer = csv.writer(c, dialect="excel")

        # implying first experiment, write column titles to csv and set exp_start i.e. reference_time
        writer.writerow(
            ["Output_Pressure", "Pump_Measured_Speed", "Time"])

        # Turn motor on and change pump speed
        if not is_motor_on():
            pump_toggle()

        for i, set_point in enumerate(set_points):
            print(f"Collecting pump data for sp:{set_point}")
            pump_speed_change(set_point)
            if i == 0:
                write_tag_values(writer, set_point, set_time, True)
            else:
                write_tag_values(writer, set_point, set_time)

        # shut down the motor after experiments end
        if is_motor_on():
            pump_toggle()
    return True


def write_tag_values(writer, sp=None, set_time=5, is_first_exp=False):
    # read values for sp and set_time
    if not sp:
        raise ValueError("No value supplied for set point")

    if not request_builder(PUMP_FOLDER, PUMP_SPEED, sp, 'write'):
        raise requests.RequestException(
            f"Write request for tag {PUMP_FOLDER}.{PUMP_SPEED} failed for unknown reason")

    global exp_cur_start, exp_cur_end, exp_prev_delta
    exp_cur_start = datetime.now(curr_tz)

    for _ in tqdm.tqdm(range(set_time)):
        t1 = time.perf_counter()

        # send read requests to get tag values for motor_speed and inlet_pressure
        motor_speed, inlet_pressure, time_stamp = read_tag_values()

        # to maintain almost even data collection
        t2 = time.perf_counter()
        if t2 - t1 < 0.5:
            time.sleep(0.5 + t1 - t2)

        if is_first_exp:
            time_stamp = (time_stamp - exp_cur_start).total_seconds()
        else:
            time_stamp = (
                time_stamp - exp_cur_start).total_seconds() + exp_prev_delta

        writer.writerow((inlet_pressure, motor_speed, time_stamp))

    # Update the time variables
    exp_cur_end = datetime.now(curr_tz)
    exp_prev_delta += (exp_cur_end - exp_cur_start).total_seconds()


def read_tag_values():
    # read tag values for Motor RPM and Pressure

    rpm_val = request_builder(PUMP_FOLDER, PUMP_FDBK)[JSON_DATA][JSON_TAG_ID]
    psi_val = request_builder(PSI_PIDE_FOLDER, P_OUTLET)[JSON_DATA][JSON_TAG_ID]
    try:
        motor_speed = rpm_val[JSON_TAG_VALUE]
        inlet_pressure = psi_val[JSON_TAG_VALUE]
        time_stamp = psi_val[JSON_TAG_TIMESTAMP]
    except Exception as e:
        print("An exception occurred")
        print(e)

    # Convert time_stamp into datetime objects
    time_stamp = datetime.replace(
        dateutil.parser.parse(time_stamp), tzinfo=None)
    time_stamp = old_tz.localize(time_stamp).astimezone(curr_tz)

    return motor_speed, inlet_pressure, time_stamp


def request_builder(folder="", files=[], value=0, read_write="read"):
    # request string builder

    # Format: <Server endpoint>?computer=<Host IP>&item=<Server Name>-><Tag Name>&value=<Value if write operation>
    # Restrictions:
    #       1.  Only one folder name can be passed as multiple folder + file combinations need 2D-arrays,
    #           increasing time complexity and decreasing performance.
    #       2.  Assuming only one write value is permitted

    request_string = REQUEST_ENDPOINT.format(read_write)
    if not folder.strip():
        raise Exception("Invalid tag folder")

    if isinstance(files, list) and len(files):
        # if multiple files, create a joint string of &item=<server>-><tag>&value=<value>
        for file in files:
            request_string += f"{SERVER_URL}{cmd_string(folder, file)}"

    elif files.strip():
        request_string += f"{SERVER_URL}{cmd_string(folder, files)}"

    else:
        raise Exception("Invalid request parameters, please try again")

    if value or (not value and read_write.lower() != "read"):
        # if value = 0 but writing or if value not zero implying
        # no need to use "write" explicitly if value not null.

        request_string += "&value=" + str(value)

    res = requests.get(request_string, verify=False)
    return json.loads(res.text)


def pump_toggle():
    # Starts/Stops the pump
    start_stop = is_motor_on()

    # set initial speed to a safe value e.g. 10 Hz if initially set to >60 Hz
    res = request_builder(PUMP_FOLDER, PUMP_FDBK)
    if int(res[JSON_DATA][JSON_TAG_ID][JSON_TAG_VALUE]) > 60:
        pump_speed_change(10)

    # Switch OCmd_Start -> 1 and OCmd_Stop -> 0 or vice versa.
    request_builder(PUMP_FOLDER, PUMP_START, 1 - start_stop, "write")
    request_builder(PUMP_FOLDER, PUMP_STOP, start_stop, "write")


def pump_speed_change(speed_set_point=None):
    # Control Pump RPM (Hz)
    # Params:
    #       speed_set_point - pump speed in Hz

    if speed_set_point is None:
        speed_set_point = input(
            "Enter speed in RPM(Hz) [0-60] to be safe\n>>>").strip()

        if not speed_set_point.isdigit():
            raise Exception("Enetred speed not a numeric value")

    speed_set_point = int(speed_set_point)

    if 0 <= speed_set_point <= 60:
        request_builder(PUMP_FOLDER, PUMP_SPEED, speed_set_point, "write")

    else:
        raise Exception("Entered speed not within safe limits")


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
    raise Exception("Invalid folder/file name")


def is_motor_on():
    # Reads current motor speed
    # Return -> 1|0
    res = request_builder(PUMP_FOLDER, IS_RUNNING)
    return int(res['data'][0]["Value"])


def plot_pump_data():
    # Import csv data and check if csv path exists.
    # If exists, read csv data with pandas, predefined columns as the usecols parameter.

    if not os.path.exists(get_csv_path()):
        raise Exception("No experiment data to plot")

    df = pd.read_csv(get_csv_path(), usecols=CSV_COLUMNS)
    plt.plot(df.Time, df.Output_Pressure,
             df.Time, df.Pump_Measured_Speed, marker="+")
    plt.xlabel("Time in seconds")
    plt.grid(visible=True)
    plt.title("Plot of Pump speed and Outlet Pressure v/s Time")
    plt.legend(["Outlet Pressure", "Pump Speed  "])
    plt.show()


def pump_control_cli():
    # on initialization remove all previous data
    if os.path.exists(get_csv_path()):
        os.remove(get_csv_path())

    # Lock the motor in operator mode for open-loop testing
    request_builder(PUMP_FOLDER, OPER_LOCK, 1)

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
            pump_toggle()

        elif option == 2:
            pump_speed_change()

        elif option == 3:

            if os.path.exists(get_csv_path()):
                print("Deleting previous experiment results...")
                os.remove(get_csv_path())

            get_set_points()

        elif option == 4:
            plot_pump_data()

        elif option == 5:

            if os.path.exists(get_csv_path()):
                print("Deleting previous experiment results...")
                os.remove(get_csv_path())

            else:
                print("No experiment data exists")

        elif option == 6:
            # close connection and exit
            print("Exiting...")

            request_builder(PUMP_FOLDER, PUMP_STOP, 1)

            exit()
        else:
            print("Incorrect Option, Try again")
            continue


if __name__ == "__main__":

    # Initially, set pump speed to 10 Hz and ask for set points
    if request_builder(PUMP_FOLDER, PUMP_SPEED, 10):
        print("Pump set at initial RPM of 10 Hz")

    pump_control_cli()
