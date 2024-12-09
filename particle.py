import numpy as np

class Particle:
    def __init__(self, x, y, mass=1.0):
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.force = np.zeros(2, dtype=float)
        self.mass = mass

    def update_position(self, dt):
        acceleration = self.force / self.mass
        self.position += self.velocity * dt + 0.5 * acceleration * dt ** 2
        self.velocity += acceleration * dt

    def reset_force(self):
        self.force = np.zeros(2, dtype=float)

    def reflect_from_walls(self, xlim, ylim, restitution_coefficient=0.5):
        if self.position[0] <= xlim[0]:
            self.position[0] = xlim[0]
            self.velocity[0] = -self.velocity[0] * restitution_coefficient
        elif self.position[0] >= xlim[1]:
            self.position[0] = xlim[1]
            self.velocity[0] = -self.velocity[0] * restitution_coefficient

        if self.position[1] <= ylim[0]:
            self.position[1] = ylim[0]
            self.velocity[1] = -self.velocity[1] * restitution_coefficient
        elif self.position[1] >= ylim[1]:
            self.position[1] = ylim[1]
            self.velocity[1] = -self.velocity[1] * restitution_coefficient

    def __repr__(self):
        return f"Particle(position={self.position}, velocity={self.velocity}, force={self.force}, mass={self.mass})"
