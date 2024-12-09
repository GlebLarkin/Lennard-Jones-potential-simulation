# Модель газа Ван-дер-Ваальса
## Моделирование газа Ван-дер-Ваальса с помощью потенциала Леннарда-Джонса
![Очень шакальная и лагучая визуализация взаимодействия 50 частиц](https://github.com/GlebLarkin/Lennard-Jones-potential-simulation/blob/main/output_compress-video-online.com_.gif)

Для работы симуляции необходимы библиотеки numpy и matplotlib

### А что собственно происходит?
[Потенциал Леннарда-Джонса](https://ru.wikipedia.org/wiki/Потенциал_Леннарда-Джонса) описывает зависимость потенцильной энергии взаимодействия двух неполярных молекул от расстояния между ними:
```math
U(r) = \varepsilon \left((\frac{\sigma}{r})^{12} - (\frac{\sigma}{r})^{6}\right)
```
где 
r — модуль вектора $\vec{r}$ (вектор, по модулю равный расстоянию между центрами двух частиц и направленный от одной к другой), $\varepsilon$ и $\sigma$ — константы.
Градиент потенциала есть сила взаимодейстия:
```math
\vec{F} = - \vec{\nabla} U(r)= 24\varepsilon \frac{\sigma^6}{r^7} \left(2 \frac{\sigma^6}{r^7} - 1 \right) \frac{\vec{r}}{r}
```
Так, частицы отталкиваются при малых расстояниях между ними и притягиваются при больших расстояниях. Такая можель хорошо описывает поведение веществ с малой плотностью, особенно хорошо - газов.
Подобная модель [газа](https://ru.wikipedia.org/wiki/Уравнение_Ван-дер-Ваальса) была предложена Ван-дер-Ваальсом и добавляла в модель идеального газа константы, учитыващие существование минимального объема, который может занимать газ, и наличие межмолекулярных сил притяжения.

Попробуем проверить модель реального газа при его описании уравнением состояния газа Ван-дер-Ваальса, если моделировать межмолекулярное взаимодействие с помощью модели потенциала Леннарда-Джонса.

### А как это работает?
#### Particle
Частица - основной объект, используемый в симуляции. Частица простая - она умеет только только перемещаться (метод update_position) и отскакивать от стен, теряя энергию (метод reflect_from_walls). 

_Казалось бы, если при отскоке от стен теряется энергия, то система должна стремится к равновесию. Но за два часа расчетов система к равновесию так и не пришла. Я думаю, тут дело в том, что в системе нет сухого трения, энергия убывает пропорционально скорости, поэтому система стремится к равновесию ассимтотически. К тому же, из-за наличия сил притяжения, которые начинают действовать на расстоянии меньшем, чем размер "коробки с молекулами", достичь устойивого равновесия крайне тяжело, ведь для этого нужна одновременная остановка всех частиц в нужных координатах._

Метод update_position рассчитывает новые координаты частицы с помощью простейшего метода Эйлера - при перемещении сила, дейстивующая на частицу считается постоянной, поэтому используются кинематические формулы равноускоренного движения. 
```math
r = r_0 + \dot{\mathbf{r}} h + \frac{1}{2} \ddot{\mathbf{r}} h^2
```

```math
\dot{\mathbf{r}} = \dot{\mathbf{r_0}} + \ddot{\mathbf{r}} h
```
где r - радиус вектор, h - шаг по времени

Также на каждом шаге симуляции важно сбрасывать значние силы, дейстивующую на частицы, в противном случае она будет накапливаться при каждом обновлении координат частиц, что приведет к неавдекватном поведению системы. Для этого используется метод reset_force.
```python
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
```

#### Particle system
Через класс ParticleSystem реализована сама модель взаимодействие с частицами. Рассмотрим его методы:
- compute_force вычисляет силу силу взаимодействия согласно выражению для градиента потенциала Леннарда-Джонса. Из интересного тут - так как частицы движутся не непрерывно, а "прыгают" на дискретные расстояния, две частицы могут оказаться в одной точке, и сила взаимодейстия станет бесконечно большой. Не менее страшна ситуация, при которой частицы оказываются очень близко. Поэтому я добавил минимальный радиус взаимодействия - если расстояние между двумя частицами меньше, чем этот радиус, то она взаимодейстивуют так, как будто они находятся на расстоянии минимального радиуса.
-  update_forces обновляет поле force класса particle. Для оптимизации вычислений я добавил и максимальный радиус взаимодействия - давлекие частицы, слабо вляющие друг на друга, не будут взаимодействовать.
-  update_positions перемещает все частицы на каждом шаге.
-  step вызывает все другие методы ParticleSystem в правильном порядке

```python
class ParticleSystem:
    def __init__(self, particles, epsilon, sigma, interaction_radius, xlim, ylim, r_min=0.1):

        self.particles = particles
        self.epsilon = epsilon
        self.sigma = sigma
        self.interaction_radius = interaction_radius
        self.xlim = xlim
        self.ylim = ylim
        self.r_min = r_min
        self.force_limit = force_limit

    def compute_force(self, r):

        r = max(r, self.r_min)
        sigma_over_r6 = self.sigma / r ** 6
        sigma_over_r12 = sigma_over_r6 ** 2
        force = 24 * self.epsilon * (2 * sigma_over_r12 - sigma_over_r6) / r

        return -force

    def update_forces(self):
        n = len(self.particles)

        for particle in self.particles:
            particle.reset_force()

        for i in range(n):
            for j in range(i + 1, n):
                r_vec = self.particles[j].position - self.particles[i].position
                r = np.linalg.norm(r_vec)

                if r > 0 and r < self.interaction_radius:
                    f = self.compute_force(r)
                    force_vec = f * r_vec / r
                    self.particles[i].force += force_vec
                    self.particles[j].force -= force_ve

    def update_positions(self, dt):
        for particle in self.particles:
            particle.update_position(dt)
            particle.reflect_from_walls(self.xlim, self.ylim)

    def step(self, dt, xlim, ylim):
        self.update_forces()
        self.update_positions(dt)
```

#### Particle Visualizer
Этот класс отвечает за отображение симуляции. Для этого будем использовать славный matplotlib.animation. 
Рассмотрим и его методы:
- Основной метод класса - animate, в котором вызывается функция FuncAnimation из библиотеки matplotlib.animation. FuncAnimation обновляет окно с отображаемыми частицами (причем при blit=True она делает это оптимизированно - обновляет лишь объекты с измененные координатами). Она требует реализации функции update - функции, которая говорит FuncAnimation, что делать на каждом фрейме. Рассмотрим ее.
- update на каждом шаге говорит ParticleSystem сделать просчитать один шаг симуляции, затем обновляет координаты всех отображаемых частиц. update возвращает кортеж с одним значением - этого требует параметр blit=True функции FuncAnimation

_К сожалению, я не очень хорошо спроектировал код, и этот класс напрямую взаимодейстивует с Particle system - именно этот класс запускает работу Particle system, хотя эта обязанность должна лежать на контроллере.
К счастью, я не сильно собираюсь расширять и поддерживать этот проект, поэтому ошибка нефатальная._
```python
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
```

#### Controller
Контроллер - главный гость программы. Он мог бы всем управлять, но по итогу только создает и правильно инициализирует объекты всех нужных классов и запускает симуляцию методом start_simulation. Из интеремного - частицы спаунятся случайно.
```python
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
```
#### main
В main по сути реализуется конфиг всей симуляции, а также создается объект класса controller. В целом ничего интересного. Все коэффициенты подобраны эмпирически. Если хотите посмотреть на частички - запускайте его, предварительно скопировав все остальные файлы в одну папку.
```python
num_particles = 30
epsilon = 2
sigma = 1.01
interaction_radius = 8
time_step = 0.006
xlim = (0, 10)
ylim = (0, 10)
r_min = 1e-4

controller = Controller(
    num_particles=num_particles,
    epsilon=epsilon,
    sigma=sigma,
    interaction_radius=interaction_radius,
    r_min=r_min,
    time_step=time_step,
    xlim=xlim,
    ylim=ylim
)

controller.start_simulation(interval=5, frames=500)
```
### И что в итоге?
А в итоге python для таких симуляций в реальном времени не подходит (хотя работу модели можно сильно ускорить, если хорошенько посидеть с профайлером и поискать, где я забыл подвязать numpy). Всего моделька на моем ноутбуке тянет +- 50 частиц, что невероятно мало для проверки уравнения состояния газа Ван-дер-Ваальса (его модель, как и вся статистическая физика, работает при малых относительных погрешностях, те при условии $N >> \sqrt{N}$, те на количестве частиц порядка 10 000). Но выглядит красиво :)
