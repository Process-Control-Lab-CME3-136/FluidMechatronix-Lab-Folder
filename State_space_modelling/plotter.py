import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from datetime import datetime
import collections
import time
import os
import pytz
from dateutil import parser

#import opcua
#from opcua import Client, ua, uamethod

BAD_ID_UNKNOWN = "BadNodeIdUnknown"
CSV_FILENAME = 'Sim_data.csv'
GIF_NAME = 'movie.gif'
STATUS_CODE = 6

PLOT_COLORS = mcolors.TABLEAU_COLORS
DIR_PATH = os.path.abspath(os.path.dirname(__file__))
ENDPOINT = "opc.tcp://admin@localhost:4843/freeopcua/server"

old_tz = pytz.timezone("Africa/Accra")
curr_tz = pytz.timezone("America/Edmonton")


class Queue:
    # Basic implementation of queue data structure

    def __init__(self, maxlen=100):
        # init
        self.items = collections.deque(maxlen=maxlen)
        self.maxlen = maxlen

    def __str__(self):
        print(self.items)

    def enqueue(self, item):
        self.items.append(item)

    def dequeue(self):
        return self.items.popleft()

    def is_full(self):
        return len(self.items) == self.maxlen

    def is_empty(self):
        return len(self.items) == 0

    def flush_queue(self):
        """
        Clear queue of any leftover elements
        """
        self.items.clear()

    def peek(self, index=-1):
        """
        Peek at the last element added to the queue
        """
        if len(self.items) >= index >= 0:
            return self.items[index]
        else:
            return self.items[-1]

    def queue_slice(self, start, end):
        return list(self.items)[start:end]


class Plotter:
    def __init__(self, window=25, interval=200, save_count=100, save_gif=True, gif_path=GIF_NAME):
        self.window = window
        self.interval = interval
        self.save_count = save_count
        self.save_gif = save_gif
        self.gif_path = gif_path
        self.x_delta = 0.1

    def parse_datetime(self, start_time: datetime, time_stamp: datetime):

        x = datetime.replace(parser.parse(str(time_stamp)), tzinfo=None)
        x = old_tz.localize(x).astimezone(curr_tz)
        x = (x-start_time).total_seconds()

        return x

    def cont_plot(self):

        # set start time for reference
        start_time = datetime.now(curr_tz)

        # initialize data generator by passing a string tag id.
        data_gen = self.data_gen(start_time)

        # initial values to pass to pyplot and adjust figure window
        x_min, y_min = next(data_gen)
        window = self.window

        # Create queue to store values as (x, y) pairs
        queue = Queue(2000)
        queue.flush_queue()
        queue.enqueue((x_min, y_min))

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        x_min = min(0, x_min)
        y_min = min(0, y_min)
        ax.set_xlim(x_min-window//10, window*0.9)
        ax.set_ylim(y_min-window//10, y_min + window//10)

        line, = ax.plot([x_min], y_min, 'b', marker=".")

        def animate(i):

            nonlocal x_min, y_min
            x, y = next(data_gen)
            if i != 0:
                x_prev = queue.peek()[0]
                if x <= x_prev:
                    # same data point, use zoh.
                    x = x_prev + self.x_delta

                else:
                    # is it necessary?
                    self.x_delta = x-x_prev

            queue.enqueue((x, y))

            queue_slice = queue.queue_slice(0, queue.maxlen)
            queue_slice = list(zip(*queue_slice))
            line.set_data(queue_slice)

            # x can only be >= previous x
            if x >= window:
                ax.set_xlim((x_min + self.x_delta, x + self.x_delta+window//10))
                x_min = x - window

            if line.get_data()[0][0] < ax.get_xlim()[0]-window*1.1:
                # pop elements not shown in window from queue
                queue.dequeue()

            if y > ax.get_ylim()[1]:
                del_y = y-queue.peek()[1] + window//10
                ax.set_ylim(top=y+del_y)

            elif y < y_min:
                y_min = y
                ax.set_ylim(bottom=y_min-window//10)

            return line,

        ani = FuncAnimation(fig, animate, interval=self.interval,
                            blit=False, save_count=self.save_count)
        plt.xlabel("Time / sec")
        plt.ylabel("Output y(t) /kPa")
        plt.title("Output Response /kPa for Input /rpm")

        plt.tight_layout()
        plt.show()

        if self.save_gif:
            # Time taken to save depends on save_count defined above.
            if os.getcwd() != DIR_PATH:
                self.gif_path = os.path.join(DIR_PATH, self.gif_path)

            # NOTE: when saving as a gif, make sure the number of plotted points exceeds save_count
            ani.save(self.gif_path, writer="Pillow")


class ExperimentPlot(Plotter):
    """
    Class implementation for plotting experimental data using opc.read
    arguments:  window, interval, save_count, save_gif, gif_path
    """
    def __init__(self, client:opcua.Client, tag_id:str, **kwargs):
        super().__init__(**kwargs)
        self.set_client(client)
        self.set_tag_id(tag_id)

    def set_client(self, client:opcua.Client):
        self.client = client
        self.client.connect()
    
    def set_tag_id(self, tag_id):

        # For now hardcoded to Pressure Value
        self.tag_id = tag_id

    def get_node(self):
        """
        Get Node object from tag id
        """
        return self.client.get_node(self.tag_id)

    def data_gen(self, start_time: datetime):

        # Get node object and initialize a generator
        node = self.get_node()

        while True:
            data_val = node.get_data_value()
            t_stamp = data_val.SourceTimestamp
            value = data_val.Value.Value

            x = self.parse_datetime(start_time, t_stamp)

            yield (x, value)
            time.sleep(0.1)

    def disconnect(self):
        self.client.close_session()
        self.client.disconnect()


class SimulatedPlot(Plotter):
    """
    Class implementation for plotting simulated data using env.run()
    arguments:  window, interval, save_count, save_gif, gif_path
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def data_gen(self, start_time: datetime):

        # For now replace simulated values by random values

        while True:
            x = datetime.now(old_tz)
            x = self.parse_datetime(start_time, x)

            value = np.random.random()*10
            yield (x, value)
            time.sleep(0.1)

    def disconnect(self):
        exit()



def main():
    try:
        
       # client = Client(ENDPOINT)
        
        online_plot = SimulatedPlot()
        online_plot.cont_plot()

    except Exception as e:
        print(e, e.args)

    finally:
        online_plot.disconnect()


if __name__ == "__main__":
    main()
