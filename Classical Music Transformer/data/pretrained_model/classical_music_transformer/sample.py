from model import *
from settings import *
import os
import time
import pretty_midi as pm

class Item(object):
	def __init__(self, name, start, end, pitch, dur, chord, tone):
		self.name = name
		self.start = start
		self.end = end
		self.pitch = pitch
		self.dur = dur
		self.chord = chord
		self.tone = tone

	def __repr__(self):
		return 'Item(name={}, start={}, end={}, pitch={}, chord={}, dur={}, tone={})'.format(
			self.name, self.start, self.end, self.pitch, self.chord, self.dur, self.tone)


def seq2input(seq, d_type=5, d_barpos=34, d_tone=25, d_chord=109, d_dur=65, d_pitch=129):
	type_oh = []
	barpos_oh = []
	tone_oh = []
	chord_oh = []
	dur_oh = []
	pitch_oh = []
	for i in seq:
		type_vec = [0]*d_type
		barpos_vec = [0]*d_barpos
		tone_vec = [0]*d_tone
		chord_vec = [0]*d_chord
		dur_vec = [0]*d_dur
		pitch_vec = [0]*d_pitch

		if i == "bar":
			type_vec[0] = 1
			barpos_vec[33] = 1
			tone_vec[0] = 1
			chord_vec[0] = 1 
			dur_vec[0] = 1
			pitch_vec[0] = 1

		elif "tonality" in i:
			tname = i.split("tonality")[1]
			type_vec[2] = 1
			barpos_vec[0] = 1
			tone_vec[int(allchordtype_dict[tname]) + 1] = 1
			chord_vec[0] = 1 
			dur_vec[0] = 1
			pitch_vec[0] = 1
	
		elif "chord" in i:
			cname = i.split("chord")[1]
			type_vec[3] = 1
			barpos_vec[0] = 1
			tone_vec[0] = 1
			chord_vec[int(allchordtype_dict[cname]) + 1] = 1 
			dur_vec[0] = 1
			pitch_vec[0] = 1

		elif i == "phrase":
			type_vec[4] = 1
			barpos_vec[0] = 1
			tone_vec[0] = 1
			chord_vec[0] = 1 
			dur_vec[0] = 1
			pitch_vec[0] = 1

		elif "note" in i:
			dur = int(i.split("dur")[1])
			nname = i.split("dur")[0].split("note")[1]
			pitch = int(pitchtype_dict_rev[nname[:-1]]) + int(nname[-1])*12 + 12
			if pitch < 0 or pitch > 127:
				continue
			type_vec[1] = 1
			barpos_vec[0] = 1
			tone_vec[0] = 1
			chord_vec[0] = 1
			dur_vec[dur] = 1
			pitch_vec[pitch + 1] = 1
		
		elif "pos" in i:
			pname = int(i.split("pos")[1])
			type_vec[0] = 1
			barpos_vec[pname + 1] = 1
			tone_vec[0] = 1
			chord_vec[0] = 1 
			dur_vec[0] = 1
			pitch_vec[0] = 1

		type_oh.append(type_vec)
		barpos_oh.append(barpos_vec)
		tone_oh.append(tone_vec)
		chord_oh.append(chord_vec)
		dur_oh.append(dur_vec)
		pitch_oh.append(pitch_vec)
	
	type_oh = np.array(type_oh)
	barpos_oh = np.array(barpos_oh)
	tone_oh = np.array(tone_oh)
	chord_oh = np.array(chord_oh)
	dur_oh = np.array(dur_oh)
	pitch_oh = np.array(pitch_oh)
	return type_oh, barpos_oh, tone_oh, chord_oh, dur_oh, pitch_oh

def nucleus_sampling(vec2, ro = 0.9):
	vec = np.array(vec2)
	sorted_vec = np.sort(vec)[::-1]
	sorted_index = np.argsort(vec)[::-1]
	cumsum_sorted_vec = np.cumsum(sorted_vec)
	dropped = cumsum_sorted_vec > ro
	if np.sum(dropped) > 0:
		last_index = np.where(dropped)[0][0] + 1
		candi_index = sorted_index[:last_index]
	else:
		candi_index = sorted_index[:]

	candi_probs = [vec[i] for i in candi_index]
	candi_probs /= sum(candi_probs)
	choice = np.random.choice(candi_index, size=1, p=candi_probs)[0]
	return choice

def choose_type(logit):
	shape = logit.shape
	tmp = pred_type.view(shape[0]*shape[1], shape[2])
	for i in range(len(tmp)):
		chosen_type = nucleus_sampling(tmp[i].softmax(-1).tolist())
		tmp[i, :] = 0
		tmp[i, chosen_type] = 1

	return tmp.reshape(shape[0], shape[1], shape[2])

def onehot2item(t, barpos, tone, chord, dur, pitch, bpm = 120):
    note_unit = 1/(8*(bpm/60))
    bar_unit = 4/(bpm/60)
    allitem = []
    bar_count = 0
    POS = 0
    for i in range(len(t)):
        if t[i, 0] == 1:
            tmp = int(torch.argmax(barpos[i, :]))
            if tmp == 33:
                bar_count += 1
                POS = 0
                
            elif tmp != 0:
                POS = int(torch.argmax(barpos[i, 1:33])) 
        
        elif t[i, 1] == 1:
            DUR = int(torch.argmax(dur[i, 1:])) + 1
            PITCH = int(torch.argmax(pitch[i, 1:]))
            note = Item(name="note", start=(bar_count - 1)*bar_unit + POS*note_unit, end=(bar_count - 1)*bar_unit + (POS+DUR)*note_unit, pitch=PITCH, dur=DUR, chord=-1, tone=-1)
            allitem.append(note)
            
    return allitem

def item2midi(item, fn, tempo=120.0):
    mid = pm.PrettyMIDI(initial_tempo = tempo)
    inst = pm.Instrument(0, is_drum=False)
    for i in item:
        if i.name != 'note':
            continue
        inst.notes.append(pm.Note(velocity=90, pitch=i.pitch, start=i.start, end=i.end))
    mid.instruments.append(inst)
    mid.write(fn)


### GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### model settings
d_type = 5
d_barpos = 34
d_tone = 25
d_chord = 109
d_dur = 65
d_pitch = 129
d_attention = 512
heads = 8
N = 12
model = Transformer()
if torch.cuda.is_available():
	model.cuda()


model.load_state_dict(torch.load("./checkpoint.pt"))

### make start seq
# start_seq = ["bar", "phrase", "pos0", "noteC4dur8"]
start_seq = ["bar", "phrase"]
curbar = 1
curpos = 0
curchord = -1
curtone = -1
chords = []
tones = []
phrases = []

### seq to input
type_oh, barpos_oh, tone_oh, chord_oh, dur_oh, pitch_oh = seq2input(start_seq)
type_oh = torch.from_numpy(type_oh).type(torch.FloatTensor).cuda()
barpos_oh = torch.from_numpy(barpos_oh).type(torch.FloatTensor).cuda()
tone_oh = torch.from_numpy(tone_oh).type(torch.FloatTensor).cuda()
chord_oh = torch.from_numpy(chord_oh).type(torch.FloatTensor).cuda()
dur_oh = torch.from_numpy(dur_oh).type(torch.FloatTensor).cuda()
pitch_oh = torch.from_numpy(pitch_oh).type(torch.FloatTensor).cuda()

### sampling
num_event = 500
for i in range(len(start_seq) - 1, num_event):
	if i > 127:
		type_input = type_oh[i-127:i+1].unsqueeze(0)
		barpos_input = barpos_oh[i-127:i+1].unsqueeze(0)
		tone_input = tone_oh[i-127:i+1].unsqueeze(0)
		chord_input = chord_oh[i-127:i+1].unsqueeze(0)
		dur_input = dur_oh[i-127:i+1].unsqueeze(0)
		pitch_input = pitch_oh[i-127:i+1].unsqueeze(0)
	else:
		type_input = type_oh[:i+1].unsqueeze(0)
		barpos_input = barpos_oh[:i+1].unsqueeze(0)
		tone_input = tone_oh[:i+1].unsqueeze(0)
		chord_input = chord_oh[:i+1].unsqueeze(0)
		dur_input = dur_oh[:i+1].unsqueeze(0)
		pitch_input = pitch_oh[:i+1].unsqueeze(0)

	with torch.no_grad():
		h, pred_type = model.forward_hidden(type_input, barpos_input, tone_input, chord_input, dur_input, pitch_input, create_mask(type_input))

	pred_type = choose_type(pred_type)
	with torch.no_grad():
		pred_barpos, pred_tone, pred_chord, pred_dur, pred_pitch = model.forward_sampling(h, pred_type)

	
	type_vec = torch.tensor([0]*d_type).type(torch.FloatTensor).cuda()
	barpos_vec = torch.tensor([0]*d_barpos).type(torch.FloatTensor).cuda()
	tone_vec = torch.tensor([0]*d_tone).type(torch.FloatTensor).cuda()
	chord_vec = torch.tensor([0]*d_chord).type(torch.FloatTensor).cuda()
	dur_vec = torch.tensor([0]*d_dur).type(torch.FloatTensor).cuda()
	pitch_vec = torch.tensor([0]*d_pitch).type(torch.FloatTensor).cuda()

	### Bar & Pos
	if pred_type[0, -1, 0] == 1:
		type_vec[0] = 1

		sampled_barpos = nucleus_sampling(pred_barpos[0, -1, :].softmax(-1).tolist())
		barpos_vec[sampled_barpos] = 1
		# bar 
		if sampled_barpos == 33:
			curbar += 1
			curpos = 0
		elif sampled_barpos != 0:
			curpos = sampled_barpos - 1

		tone_vec[0] = 1
		chord_vec[0] = 1
		dur_vec[0] = 1
		pitch_vec[0] = 1

	### tone
	elif pred_type[0, -1, 2] == 1:
		type_vec[2] = 1
		barpos_vec[0] = 1

		sampledtone = nucleus_sampling(pred_tone[0, -1, 1:].softmax(-1).tolist())
		curtone = sampledtone
		tone_vec[sampledtone + 1] = 1
		chord_vec[0] = 1
		dur_vec[0] = 1
		pitch_vec[0] = 1
		print("tone", alltonetype_dict_rev[curtone], "at bar", curbar, "pos", curpos)
		tones.append([curtone, curbar, curpos])

	elif pred_type[0, -1, 3] == 1:
		type_vec[3] = 1
		barpos_vec[0] = 1
		tone_vec[0] = 1

		sampledchord = nucleus_sampling(pred_chord[0, -1, 1:].softmax(-1).tolist())
		curchord = sampledchord
		chord_vec[sampledchord + 1] = 1
		dur_vec[0] = 1
		pitch_vec[0] = 1
		print("chord", allchordtype_dict_rev[curchord], "at bar", curbar, "pos", curpos)
		chords.append([curchord, curbar, curpos])


	elif pred_type[0, -1, 4] == 1:
		type_vec[4] = 1
		barpos_vec[0] = 1
		tone_vec[0] = 1
		chord_vec[0] = 1
		dur_vec[0] = 1
		pitch_vec[0] = 1
		print("phrase at bar", curbar, "pos", curpos)
		phrases.append([curbar, curpos]) 

	else:
		type_vec[1] = 1
		barpos_vec[0] = 1
		tone_vec[0] = 1
		chord_vec[0] = 1
		sampleddur = nucleus_sampling(pred_dur[0, -1, 1:].softmax(-1).tolist())
		dur_vec[sampleddur + 1] = 1
		sampledpitch = nucleus_sampling(pred_pitch[0, -1, 1:].softmax(-1).tolist())
		pitch_vec[sampledpitch + 1] = 1
		print("pitch:", pitchtype_dict[sampledpitch % 12], "dur:", sampleddur + 1, "at bar:", curbar, "pos", curpos)

	type_oh = torch.cat([type_oh, type_vec.unsqueeze(0)], 0)
	barpos_oh = torch.cat([barpos_oh, barpos_vec.unsqueeze(0)], 0)
	tone_oh = torch.cat([tone_oh, tone_vec.unsqueeze(0)], 0)
	chord_oh = torch.cat([chord_oh, chord_vec.unsqueeze(0)], 0)
	dur_oh = torch.cat([dur_oh, dur_vec.unsqueeze(0)], 0)
	pitch_oh = torch.cat([pitch_oh, pitch_vec.unsqueeze(0)], 0)

item2midi(onehot2item(type_oh, barpos_oh, tone_oh, chord_oh, dur_oh, pitch_oh), "output.mid")
