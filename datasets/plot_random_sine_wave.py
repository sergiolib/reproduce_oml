from datasets.synth_datasets import gen_tasks, gen_sine_data
import matplotlib.pyplot as plt


tasks = gen_tasks(10)
x_traj, y_traj, _, _ = gen_sine_data(tasks)


plt.scatter(x_traj[0][0][:][:, 0], y_traj[0][0][:])
plt.scatter(x_traj[1][0][:][:, 0], y_traj[1][0][:])
# plt.scatter(x_traj[2][0][:][:, 0], y_traj[2][0][:])
# plt.scatter(x_traj[3][0][:][:, 0], y_traj[3][0][:])
# plt.scatter(x_traj[4][0][:][:, 0], y_traj[4][0][:])
plt.grid()
plt.show()
