from controller import Controller

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
