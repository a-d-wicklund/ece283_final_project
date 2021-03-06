"""
Load all songs in midi format from directory and subdirectories and save them
as a numpy array of samples (in .npy file format). The numpy array takes the form
(N x 96 x 96), where N is the total number of measures taken from the midi files

"""

import midi
import os
import util
import numpy as np

def main(from_file=True):

    if from_file:
        return np.load('samples.npy')
    patterns = {}
    all_samples = []
    all_lens = []

    SAVE_SONGS = True
    print("Loading Songs...")


    for root, _, files in os.walk('./Classical_Piano_piano-midi.de_MIDIRip', topdown=False):  # Was testing with single folder
        for file in files:
            path = os.path.join(root, file)
            # If "format" is not part of the filename, skip it
            if not (path.endswith('.mid') or path.endswith('.midi')) or not 'format' in path:
                continue
            try:
                print(path)
                tracks = midi.midi_to_samples(path)
            except:
                print("ERROR: ", path)
                continue
            for track in tracks:

                if len(track) < 8:  # If less than 8 measures, discard.
                    continue

                samples, lens = util.generate_add_centered_transpose(track)
                # all_samples.append(samples)
                # all_lens.append(lens)
                all_samples += samples
                all_lens += lens
            


    assert (sum(all_lens) == len(all_samples))
    print("Saving " + str(len(all_samples)) + " samples...")
    all_samples = np.array(all_samples, dtype=np.uint8)
    all_lens = np.array(all_lens, dtype=np.uint32)
    print(all_lens)
    if SAVE_SONGS:
        np.save('samples.npy', all_samples)
        np.save('lengths.npy', all_lens)
    else:
        return all_samples, all_lens
    print("Done")
