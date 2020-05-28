import numpy as np
from matplotlib import pyplot as plt
from mido import MidiFile, MidiTrack, Message
import midi


# samples = midi.midi_to_samples('C:\\Users\\alecw\\Documents\\UCSB\\2019-2020\\ECE 283\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\0\\009count.mid')


# for i, k in enumerate(range(3*3)):
#     plt.subplot(3,3,i+1)
#     plt.imshow(samples[0][k])
#     plt.title("measure {}".format(k+1))
# plt.show()


# Test Converting back to MIDI

# samples = np.load('samples.npy')
# midi.samples_to_midi(samples, fname='midi_out.mid')
