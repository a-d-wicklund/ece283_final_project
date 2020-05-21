import numpy as np
from matplotlib import pyplot as plt
from mido import MidiFile, MidiTrack, Message
import midi


samples = np.array(midi.midi_to_samples('piano_concerto.mid'))
print(samples.shape)



for i in range(3*3):
    plt.subplot(3,3,i+1)
    plt.imshow(samples[i])
    plt.title("measure {}".format(i+1))
plt.show()


# mid = MidiFile('piano_concerto.mid')

# piano_tracks = [0] # list of tracks belonging to piano. Include 
# for i, track in enumerate(mid.tracks):
#     for msg in track:
#         if msg.type == 'program_change':
#             if 0 <= msg.program < 6:
#                 print("found piano in track {}".format(i))
#                 piano_tracks.append(i)

# print(piano_tracks)



# track = mid.tracks[2]
# for msg in track:
#     if msg.type == 'note_on':
#         print(msg.note - (128-96)//2)


# Tracks 