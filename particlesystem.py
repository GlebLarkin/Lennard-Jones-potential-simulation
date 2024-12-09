import numpy as np

class ParticleSystem:
    def __init__(self, particles, epsilon, sigma, interaction_radius, xlim, ylim, r_min=0.1, force_limit=100):
        """
        Инициализация системы частиц в 2D.

        :param particles: Список объектов Particle.
        :param epsilon: Константа ε для потенциала Леннарда-Джонса.
        :param sigma: Константа σ для потенциала Леннарда-Джонса.
        :param interaction_radius: Радиус взаимодействия частиц.
        :param xlim: Пределы по оси x.
        :param ylim: Пределы по оси y.
        :param r_min: Минимальное расстояние для предотвращения бесконечно большой силы.
        :param force_limit: Ограничение максимальной силы, предотвращающее бесконечные ускорения.
        """
        self.particles = particles
        self.epsilon = epsilon
        self.sigma = sigma
        self.interaction_radius = interaction_radius
        self.xlim = xlim
        self.ylim = ylim
        self.r_min = r_min
        self.force_limit = force_limit  # Ограничение силы

    def compute_force(self, r):
        """
        Вычисляет силу взаимодействия с учетом деформации при столкновении.
        """
        r = max(r, self.r_min)
        sigma_over_r = self.sigma / r
        sigma_over_r6 = sigma_over_r ** 6
        sigma_over_r12 = sigma_over_r6 ** 2
        force = 24 * self.epsilon * (2 * sigma_over_r12 - sigma_over_r6) / r


        force = min(force, self.force_limit)

        return -force

    def update_forces(self):
        """
        Обновляет силы, действующие на частицы, с учетом радиуса взаимодействия и сопротивления.
        """
        n = len(self.particles)
        # Обнуляем силы перед расчетом
        for particle in self.particles:
            particle.reset_force()

        for i in range(n):
            for j in range(i + 1, n):
                # Вычисляем расстояние между частицами
                r_vec = self.particles[j].position - self.particles[i].position
                r = np.linalg.norm(r_vec)

                # Проверяем радиус взаимодействия
                if r > 0 and r < self.interaction_radius:
                    f = self.compute_force(r)
                    force_vec = f * r_vec / r
                    self.particles[i].force += force_vec
                    self.particles[j].force -= force_vec  # Третий закон Ньютона

    def update_positions(self, dt):
        """
        Обновляет координаты частиц на основе сил и отталкивает их от стен.

        :param dt: Шаг интегрирования по времени.
        """
        for particle in self.particles:
            particle.update_position(dt)
            particle.reflect_from_walls(self.xlim, self.ylim)  # Проверяем столкновение с стенками

    def step(self, dt, xlim, ylim):
        """
        Выполняет один шаг симуляции.

        :param dt: Шаг времени.
        :param xlim: Границы по оси X.
        :param ylim: Границы по оси Y.
        """
        self.update_forces()
        self.update_positions(dt)
