import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation  as animation
import matplotlib.colors as mcolors
import pandas as pd
import os

# Run this script in the State_space_modelling directory.

plot_colors = mcolors.TABLEAU_COLORS
dir_path = "State_space_modelling"
abs_path = os.path.abspath(dir_path)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))


window=50

df = pd.read_csv('Sim_data.csv')
x_data = df[df.columns[0]][0:window]
y_data = df.y[0:window]


line, = ax.plot(x_data, y_data, plot_colors['tab:blue'], marker=".")


def animate(i):
    line.set_ydata(df.y[i:window+i])
    line.set_xdata(df[df.columns[0]][i:window+i])
    
    ax.set_xlim(i, window+i)
    ax.set_ylim((df.y[0:window+i].min(), df.y[0:window+i].max()))

    if i+window >= len(df.index):
        ani.event_source.stop()
    
    
    return line,

ani = animation.FuncAnimation(fig, animate, interval=200, blit=False, save_count=100)
plt.xlabel("Time / sec")
plt.ylabel("Output y(t) /kPa")
plt.title("Output Response /kPa for Input /rpm")
plt.show()

# Time taken to save depends on save_count defined above.
ani.save('movie.gif', writer='Pillow')
