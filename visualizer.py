import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particlesystem import ParticleSystem


class ParticleVisualizer:
    def __init__(self, particle_system, time_step, xlim, ylim):
        self.system = particle_system
        self.time_step = time_step

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')

        self.scat = self.ax.scatter(
            [p.position[0] for p in self.system.particles],
            [p.position[1] for p in self.system.particles],
            c="brown", s=10
        )

    def update(self, frame):
        self.system.step(self.time_step, self.ax.get_xlim(), self.ax.get_ylim())

        positions = np.array([p.position for p in self.system.particles])
        self.scat.set_offsets(positions)

        return self.scat,

    def animate(self, interval=30, frames=200):
        anim = FuncAnimation(
            self.fig, self.update, frames=frames, interval=interval, blit=True
        )
        plt.show()
