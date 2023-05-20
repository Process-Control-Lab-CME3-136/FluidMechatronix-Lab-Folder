from opcua import Client, ua, common
from opcua.ua.status_codes import StatusCodes, code_to_name_doc
from opcua import Node
import multiprocessing as mp
from plotter import ExperimentPlot
import pytz
from constants import *
from pid_control import PositionPID


old_tz = pytz.timezone("Africa/Accra")
curr_tz = pytz.timezone("America/Edmonton")


""" State Interface """


class State(object):

    def name(self):
        return ''

    def enter(self, machine):
        pass

    def exit(self, machine):
        pass

    def update(self, machine):
        pass


class StateMachine():
    def __init__(self, save_gif=False):
        self.save_gif: bool = save_gif
        self.states: dict = {}
        self.tag_id: str = TAG_ID
        self.opercontrol_mode: str = OPER_MODE
        self.url: str = ENDPOINT

        self.state = None
        self.init_plotter = None
        self.client = None
        self.process = None
        self.server_obj = None
        self.server_name = None
        self.root = None
        self.uri = None
        self.idx = None
        self.pid = None

    def add_state(self, state: State):
        # add State to state name dictionary
        self.states[state.name()] = state

    def go_to_state(self, state_name):
        # change from one state to another based on some transition condition
        if self.state:
            self.state.exit(self)
        self.state: State = self.states[state_name]
        print("Current State: ", self.state.name())
        self.state.enter(self)

    def update(self):
        # update the current state's state variables

        if self.state:
            self.state.update(self)

    def plotter(self):
        ExperimentPlot(self.client, self.tag_id,
                       save_gif=self.save_gif).cont_plot()

    def get_pid(self):
        self.pid = PositionPID(Ts=0.1, K_p=0.3, K_i=0.001, K_d=0.01,
                               cutoff_H=155, cutoff_L=-20)

    def get_node(self, path: str) -> Node:
        """
        Returns node object as a string for the node specified by given path
        path format: Objects_Folder.Server_Folder.Nodes_Folder.Node

        """
        if isinstance(path, str) and path:

            folders = [f"{str(self.root.get_browse_name().NamespaceIndex)}:Objects",
                       f"{self.server_name}"]

            for folder in path.split("."):
                folder = f"{str(self.idx)}:{folder}"
                folders.append(folder)
            return self.root.get_child(folders)

    def init_params(self):
        self.client: Client  # declaration for syntax highlighting
        self.root: Node = self.client.get_root_node()
        self.uri: str = self.client.get_namespace_array()[SERVER_NAMESPACE_URI]
        self.idx: int = self.client.get_namespace_index(self.uri)
        self.server_obj: Node = self.client.get_node(SERVER_NODE_ID)
        self.server_name: str = self.server_obj.get_browse_name().to_string()

    def toggle_control_mode(self):

        # initialize toggle_value
        # toggle value = 0 --> operator mode
        # toggle value = 1 --> program mode
        toggle_value = 0
        if isinstance(self.state, AutomaticState) and self.opercontrol_mode == OPER_MODE:
            # set pump speed to 0 and switch to prog_mode
            pumpspeed_node = self.get_node(PUMP_SPEED)
            pumpspeed_node.set_value(0)
            toggle_value = 1

        prog_mode = self.get_node(PROG_MODE)
        prog_mode.set_value(toggle_value)
        prog_mode.set_value(toggle_value)

        oper_mode = self.get_node(OPER_MODE)
        oper_mode.set_value(1-toggle_value)
        # while self.server_obj.call_method(METHOD_ID) != "Success":
        #     time.sleep(0.1)
        oper_mode.set_value(1-toggle_value)

    def change_oper_mode(self):

        state_name = 'manual'
        if self.state.name() == state_name:
            state_name = 'automatic'

        try:
            oper_mode = input(f"""Enter option to switch or press enter to stay in {state_name.capitalize()} mode:
                                1. {state_name.capitalize()}
                                2. Exit 
                                >>>""").strip()
            if oper_mode == '':
                return
            elif int(oper_mode) == 1:
                self.go_to_state(state_name)
            elif int(oper_mode) == 2:
                self.go_to_state('exit')
        except:
            raise ValueError("Invalid option, try again")


class StartState(State):
    """
    enter: connect to client, check tags, reset tags, inspect manual, prime pump
    update: update tag status if not "Good"
    exit: start plotter function
    """

    def __init__(self):
        pass

    def name(self):
        return 'start'

    def enter(self, machine: StateMachine):
        try:

            print("Connecting to server...")
            # connect to opcua client
            machine.client = Client(machine.url)
            machine.client.connect()
        except Exception as e:
            print("Error connecting to the server, check if server available")
            exit()

        try:
            machine.init_params()

            print("Checking tags...")
            # check tags, reset tags, update tag status
            self.check_tags(machine)

            # ask operator to do a visual inspection
            self.inspect_manual()

            # start pump prime if not already primed
            self.prime_pump(machine)

        except Exception as e:
            print(e)
            machine.client.close_session()
            machine.client.disconnect()

            exit()

    def update(self, machine: StateMachine):
        # start in manual mode, then switch to automatic if user intends to
        machine.go_to_state('manual')

    def exit(self, machine: StateMachine):
        pass

    """ Helper Functions """

    def check_tags(self, machine: StateMachine):
        # check the quality of given tags
        for tag_folder, values in TAGS.items():
            for tag_name in values:
                node = machine.get_node(".".join([tag_folder, tag_name]))
                data_value = node.get_data_value()
                if data_value.StatusCode.name != 'Good':
                    # bad status code, raise error
                    raise ua.UaStatusCodeError(
                        code_to_name_doc[StatusCodes.Bad][1])

    def inspect_manual(self):
        while True:
            is_inspected = input("Is visual inspection complete?[y/n]").strip()
            if is_inspected.lower() != 'y':
                print("Please complete the visual inspection")
            else:
                break

    def prime_pump(self, machine: StateMachine):
        # ask if priming needed

        inp = input("Is pump priming needed?[y/n]").strip().lower()
        if inp != 'y':
            return

        prime_pump_node = machine.get_node(PRIME_PUMP)
        prime_pump_node.set_value(1)
        # start pump, and ask user to press enter to stop priming
        while True:
            inp = input("Press enter to stop prime pump").strip().lower()
            if inp == "":
                prime_pump_node.set_value(0)
                break


class AutomaticState(State):
    """
    enter: choose set_point -> if valid set-point then update set-point.
    update: if end_plotting -> exit/manual
    exit: Exit State if exit else Manual State
    """

    def __init__(self):
        self.sp: float = 0.
        self.steps = STEPS
        self.pressure_node = None
        self.sp_list = [2, 0, -2, 1, 5]  # random set-point list
        self.counter = 0

    def name(self):
        return 'automatic'

    def enter(self, machine: StateMachine):
        try:
            # start plotting function
            if not machine.init_plotter:
                machine.process = mp.Process(target=plot_wrapper)
                machine.process.start()
                machine.init_plotter = True

            # switch to program mode
            machine.toggle_control_mode()

            # get SP from user, after x steps change SP
            self.get_sp()
            self.pressure_node = machine.get_node(OUTPUT_PRESSURE)
            machine.get_pid()

        except ValueError as e:
            print(e)
            self.enter()

        except Exception as e:
            print(e)
            machine.go_to_state('exit')

    def update(self, machine: StateMachine):

        # run pid control loop for x steps
        for _ in range(self.steps):
            y = self.pressure_node.get_value()
            machine.pid.step(y, self.sp)

        # change SP
        if self.counter < len(self.sp_list):
            self.sp = self.sp_list[self.counter]
            self.counter += 1
        else:
            machine.go_to_state('exit')

    def exit(self, machine: StateMachine):
        # reset counter, sp
        self.counter = 0
        self.sp = 0.

        # ask if want to switch to manual or exit
        machine.change_oper_mode()

    """Helper Function(s)"""

    def get_sp(self):
        # ask if MV needs to be changed, if yes, get new mv value from user
        try:
            sp = float(input("Enter a value for SP:").strip())
            if SP_HIGH >= sp >= SP_LOW:
                self.sp = sp
                return True
            else:
                raise ValueError(
                    "Entered value out of bounds, please try again")
        except:
            raise ValueError("Invalid value, please try again")


class ManualState(State):
    """
    enter: choose MV -> if valid then update actuator position
    update: if oper_mode changed -> exit/automatic
    exit: Exit State if exit else Automatic State
    """

    def __init__(self):
        self.mv = 0.0

    def name(self):
        return 'manual'

    def enter(self, machine: StateMachine):
        # start plotting function
        if not machine.init_plotter:
            machine.process = mp.Process(target=plot_wrapper)
            machine.process.start()
            machine.init_plotter = True

        machine.toggle_control_mode()

    def update(self, machine: StateMachine):
        try:
            # get mv from user
            if not self.change_mv():
                return
            # update mv on server and the experiment
            self.set_mv(machine)

            # setting mv, changing oper mode triggered by key-patterns
            machine.change_oper_mode()

        except ValueError as e:
            print(e)
            return

        except Exception as e:
            print(e)
            machine.go_to_state('exit')

    def exit(self, machine: StateMachine):
        pass

    """ Helper Function(s) """

    def change_mv(self):
        # ask if mv needs to be changed, if yes, get new mv value from user
        try:
            inp = input('Change MV ?[y/n]').strip().lower()
            if inp != 'y':
                return True

            mv = float(input("Enter a value for MV:").strip())
            if MV_HIGH >= mv >= MV_LOW:
                self.mv = mv
                return True
            else:
                print(f"mv value: {mv} is out of bounds")
        except:
            raise ValueError("Invalid value, please try again")

    def set_mv(self, machine: StateMachine):
        try:
            pumpspeed_node = machine.get_node(PUMP_SPEED)
            pumpspeed_node.set_value(self.mv)
        except:
            raise ua.UaError(
                "Unable to set node value, check server connection or tag id")


class ExitState(State):
    """
    Exit State, on exiting/reset
    enter:  disconnect client
    update: go to state->start / shutdown
    exit: exit if shutdown else goto Start State
    """

    def __init__(self):
        pass

    def name(self):
        return 'exit'

    def enter(self, machine: StateMachine):
        machine.client.close_session()
        machine.client.disconnect()
        if isinstance(machine.process, mp.Process):
            machine.process.terminate()
            machine.process.join()
            machine.process.close()

    def update(self, machine: StateMachine):
        try:

            inp = int(input("""Select an option for exiting:
            1. Restart
            2. Shutdown
            >>>""").strip())

            if inp == 1:
                machine.init_plotter = False
                machine.go_to_state('start')

            elif inp == 2:

                print("Closing session...")
                exit()
        except Exception as e:
            print(e)
            print("Invalid option entered, please try again.")

    def exit(self, machine: StateMachine):
        pass


def plot_wrapper():
    plot = StateMachine()
    plot.plotter()


def main():
    machine = StateMachine()
    machine.add_state(StartState())
    machine.add_state(ManualState())
    machine.add_state(AutomaticState())
    machine.add_state(ExitState())

    machine.go_to_state('start')
    while True:
        machine.update()


if __name__ == "__main__":
    main()
