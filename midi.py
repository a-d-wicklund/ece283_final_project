from mido import MidiFile, MidiTrack, Message
import numpy as np

num_notes = 96
samples_per_measure = 96

def find_piano_tracks(mid):

	piano_tracks = [0] # include the 0th element because it often has time signature data
	for i, track in enumerate(mid.tracks):
		for msg in track:
			if msg.type == 'program_change':
				if 0 <= msg.program < 6:
					print("found piano in track {}".format(i))
					piano_tracks.append(i)
	return piano_tracks


def midi_to_samples(fname):

	"""
	Construct matrix form of midi songs
	
	Input:	fname : string
	            file name of midi file
	
	Output:	samples : list of numpy arrays
	            Each numpy array is (N x samples_per_measure x num_notes)
	            where N is the number of measures for that instrument
	            It is a list because there can be multiple piano tracks in a song
	"""

	
	has_time_sig = False
	flag_warning = False
	mid = MidiFile(fname)
	ticks_per_beat = mid.ticks_per_beat
	ticks_per_measure = 4 * ticks_per_beat

	piano_tracks_idxs = find_piano_tracks(mid)

	for i, track in enumerate(np.array(mid.tracks)[piano_tracks_idxs]):
		for msg in track:
			if msg.type == 'time_signature':
				new_tpm = msg.numerator * ticks_per_beat * 4 / msg.denominator
				if has_time_sig and new_tpm != ticks_per_measure:
					flag_warning = True
				ticks_per_measure = new_tpm
				has_time_sig = True
	if flag_warning:
		print("  ^^^^^^ WARNING ^^^^^^")
		print("    " + fname)
		print("    Detected multiple distinct time signatures.")
		print("  ^^^^^^ WARNING ^^^^^^")
		return []
	
	# Create a dictionary of notes. The key-value pair consists of a note as the key and a list 
	# of start-stop times as the value. 

	all_notes = []
	for i, track in enumerate(np.array(mid.tracks)[piano_tracks_idxs]):
		all_notes_inst = {}
		abs_time = 0
		for j, msg in enumerate(track):
			# FOR DEBUGGING PURPOSES
			if j == 40:
				print("nothing")
			# END DEBUGGING

			abs_time += msg.time
			if msg.type == 'note_on' and msg.velocity != 0:
				note = msg.note - (128 - num_notes) // 2 # Scale down note value (Why?)
				assert(note >= 0 and note < num_notes)
				if note not in all_notes_inst:
					# Create a new note item in the dictionary
					all_notes_inst[note] = []
				else:
					# append note event to the end of the list for that note
					single_note = all_notes_inst[note][-1]
					# first check that there's not a hanging "note_on" event 
					if len(single_note) == 1:
						print("found hanging note_on event")
						single_note.append(single_note[0] + 1)
				# Append the sample number to a new start-stop pair for the start
				all_notes_inst[note].append([int(abs_time * samples_per_measure / ticks_per_measure)])
				
			elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
				note = msg.note - (128 - num_notes) // 2 # Scale down note value (Why?)
				if len(all_notes_inst[note][-1]) != 1:
					print("encountered note_off event without note_on event")
					continue
				all_notes_inst[note][-1].append(int(abs_time * samples_per_measure / ticks_per_measure))
		all_notes.append(all_notes_inst)

	# Make sure there are no hanging "note_on" events
	for notes in all_notes:
		for note in notes:
			for start_end in notes[note]:
				if len(start_end) == 1:
					print("found hanging note_on event")
					start_end.append(start_end[0] + 1)

	# Now we have a dictionary made of note:start-stop list pairs. 

	all_samples = []
	for notes in all_notes:
		samples = []
		for note in notes:
			# For each note
			for start, end in notes[note]:
				sample_ix = start // samples_per_measure
				while len(samples) <= sample_ix:
					samples.append(np.zeros((samples_per_measure, num_notes), dtype=np.uint8))
				sample = samples[sample_ix]
				start_ix = start - sample_ix * samples_per_measure
				if True: # set to true if you want notes that are held to be shown as such
					end_ix = min(end - sample_ix * samples_per_measure, samples_per_measure)
					while start_ix < end_ix:
						sample[start_ix, note] = 1
						start_ix += 1
				else:
					sample[start_ix, note] = 1

		if samples: # if there are actually music notes in this track
			all_samples.append(np.array(samples))

	# Remove all measures containing no piano notes
	for i, inst in enumerate(all_samples):
		del_list = []
		for j, measure in enumerate(inst):
			if np.all(measure == 0):
				del_list.append(j)
		all_samples[i] = np.delete(all_samples[i], del_list, axis=0)

	return all_samples

def samples_to_midi(samples, fname, ticks_per_sample, thresh=0.5):
	mid = MidiFile()
	track = MidiTrack()
	mid.tracks.append(track)
	ticks_per_beat = mid.ticks_per_beat
	ticks_per_measure = 4 * ticks_per_beat
	ticks_per_sample = ticks_per_measure / samples_per_measure
	abs_time = 0
	last_time = 0
	for sample in samples:
		for y in range(sample.shape[0]):
			abs_time += ticks_per_sample
			for x in range(sample.shape[1]):
				note = x + (128 - num_notes)/2
				if sample[y,x] >= thresh and (y == 0 or sample[y-1,x] < thresh):
					delta_time = abs_time - last_time
					track.append(Message('note_on', note=note, velocity=127, time=delta_time))
					last_time = abs_time
				if sample[y,x] >= thresh and (y == sample.shape[0]-1 or sample[y+1,x] < thresh):
					delta_time = abs_time - last_time
					track.append(Message('note_off', note=note, velocity=127, time=delta_time))
					last_time = abs_time
	mid.save(fname)
