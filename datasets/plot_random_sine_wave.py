import synth_datasets
import matplotlib.pyplot as plt

sine_waves = synth_datasets.gen_sine_data(n_id=900, n_samples=320)
partition = synth_datasets.partition_sine_data(sine_waves)

plt.plot(partition["pretraining"]["z"][:320], partition["pretraining"]["y"][:320], "b.")
plt.grid()
plt.savefig("sample.png")
