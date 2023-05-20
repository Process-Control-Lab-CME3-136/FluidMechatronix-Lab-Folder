import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

seed = 1000
np.random.seed(seed)
CSV_PATH = "./PID_test.csv"


class Environment:
    def __init__(self, A, B, C, x_0=10):
        self.x_0 = x_0
        self.x_t = x_0
        self.A = A
        self.B = B
        self.C = C

    def reset(self):
        # reset all the state variables
        self.x_t = self.x_0
        return self.x_t

    def step(self, u_t):
        # update state variable x_t -> x_{t+1} for discrete t
        self.x_t = self.A*self.x_t + self.B*u_t
        y_t = self.C*self.x_t
        return self.x_t, y_t


class PID:
    def __init__(self, Ts=0.1, K_p=0.3, K_i=0.01, K_d=0.0, cutoff_H=0, cutoff_L=0):
        self._prev_value = 0.0  # e(t-1)
        self._saturated = False  # check if error > cutoff value

        self.Ts = Ts
        self.K_p = K_p
        self.K_i = K_i
        self.K_d = K_d
        self.cutoff_H = cutoff_H
        self.cutoff_L = cutoff_L

    def step(self, CV: float, SP: float) -> float:

        error = SP - CV

        error = self.update(error, CV)
        return error

    def check_saturation(self, y: float):

        # check for saturation
        if self.cutoff_H >= y >= self.cutoff_L:
            self._saturated = False
        else:
            # clip desired cutoff to [low, high]
            y = np.clip(y, self.cutoff_L, self.cutoff_H)
            self._saturated = True

        return y


class PositionPID(PID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._i_error = 0.0

    def update(self, error: float, CV: float) -> float:

        p_error = self.K_p*error
        d_error = self.K_d*(self._prev_value-CV)/self.Ts
        i_error = self.K_i*error*self.Ts
        I_output = self._i_error

        if not self._saturated:
            self._i_error += i_error

        self._prev_value = CV   # set previous value to current value of CV
        # Cutoff desired is current CV + p + d + i errors
        CO_desired = p_error + d_error + I_output

        return CO_desired


class VelocityPID(PID):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev2_value = self._prev_value  # y(t-2) = y(t-1)
        self._prev_error = 0.0  # e(t-1)

    def update(self, error: float, CV: float) -> float:
        p_error = self.K_p*(error-self._prev_error)
        i_error = self.K_i*error*self.Ts

        # to prevent derivative kick, using y(t) terms instead of error.
        d_error = -self.K_d*(CV-2*self._prev_value +
                             self._prev2_value)/self.Ts  # -Kd( y(t) - 2*y(t-1) + y(t-2) )
        self._prev2_value = self._prev_value
        self._prev_value = CV
        self._prev_error = error

        du = p_error + i_error + d_error

        return du


def plotter(csv_path=CSV_PATH):

    plt_colors = mcolors.TABLEAU_COLORS

    sim_data = pd.read_csv(csv_path)
    # incrementing steps for consistent dimensions
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5), sharex='all')

    # Plot input u(t) and state variable x(t) v/s time
    ax1.plot(sim_data[sim_data.columns[0]]*PID().Ts, sim_data.y,  plt_colors['tab:green'],
             sim_data[sim_data.columns[0]]*PID().Ts, sim_data.SP,  plt_colors['tab:red'])

    ax1.set(ylabel="Output y(t) /kPa",
            title="Output response v/s Time (Simulated)", xlabel="Time /sec")

    plt.tight_layout()
    plt.grid()
    plt.show()


def run_simulation(pid: PID, A=1., B=1., steps=100, csv_path=CSV_PATH, SP_list=[0], save_data=True):
    # Instantiate environment and update x_t iteratively
    env = Environment(A, B, 1., x_0=SP_list[0])

    sim_data = np.zeros((steps*len(SP_list)+1, 3))  # data = (x_all, u, y)
    # initialize data with (x_all, u, y) = (x_0, 0, 0)
    sim_data[0] = env.x_t, SP_list[0], 0

    pid = pid
    y = 0
    for j, SP in enumerate(SP_list):
        for i in range(steps*j, steps*(j+1)):
            # iterate "steps" times over steps size increments
            # e.g.: for 100 steps, iterate over [100, 200) then [200, 300) and so on.

            u = pid.step(y, SP)
            x_new, y = env.step(u)
            y = pid.check_saturation(y)

            sim_data[i+1] = (x_new, SP, y)

        # save data to "Sim_data.csv" on local directory
        print(f"K_d={pid.K_d}")
        print(f"max value: {sim_data.max()}")
        print(
            f"SP:{SP}, steady_state error:{SP-sim_data.tolist()[steps*(j+1)-1][-1]}")
    if save_data:
        sim_data_df = pd.DataFrame(sim_data, columns=["x", "SP", "y"])
        sim_data_df.to_csv(csv_path)


def main():

    steps = 50  # Number of samples taken
    # pre-defined list of set-points to iterate over
    SP_list = [100, -10, 100, 150]

    # pid = PositionPID(Ts=0.1, K_p=0.3, K_i=0.001, K_d=0.01,
    #                 cutoff_H=155, cutoff_L=-20)
    pid = VelocityPID(Ts=0.1, K_p=0.1, K_i=2, K_d=0.01,
                      cutoff_H=160, cutoff_L=-20)
    run_simulation(pid, steps=steps, A=1., B=1.,
                   SP_list=SP_list, save_data=True)
    plotter()


if __name__ == "__main__":
    main()
