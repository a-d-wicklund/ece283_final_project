import numpy as np
from matplotlib import pyplot as plt
from mido import MidiFile, MidiTrack, Message
import midi


samples = midi.midi_to_samples('piano_concerto.mid')

samples1 = np.array(samples[0])
print(samples1.shape)

for i, inst in enumerate(samples):
    for j, measure in enumerate(inst):
        if np.all(measure == 0):
            samples[i] = np.delete(samples[i], j, axis=0)
print('after reduction samples shape is {}'.format(samples.shape))

for i in range(3*3):
    plt.subplot(3,3,i+1)
    plt.imshow(samples1[i])
    plt.title("measure {}".format(i+1))
plt.show()
