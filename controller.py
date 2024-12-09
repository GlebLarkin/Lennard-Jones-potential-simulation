import numpy as np
from particlesystem import ParticleSystem
from visualizer import ParticleVisualizer
from particle import Particle


class Controller:
    def __init__(self, num_particles, epsilon, sigma, interaction_radius, r_min, time_step, xlim, ylim):
        self.particles = [
            Particle(
                x=np.random.uniform(xlim[0], xlim[1]),
                y=np.random.uniform(ylim[0], ylim[1])
            ) for _ in range(num_particles)
        ]
        self.system = ParticleSystem(
            particles=self.particles,
            epsilon=epsilon,
            sigma=sigma,
            interaction_radius=interaction_radius,
            xlim=xlim,
            ylim=ylim,
            r_min=r_min
        )
        self.visualizer = ParticleVisualizer(
            particle_system=self.system,
            time_step=time_step,
            xlim=xlim,
            ylim=ylim
        )

    def start_simulation(self, interval=50, frames=100):
        self.visualizer.animate(interval=interval, frames=frames)
