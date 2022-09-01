import xlrd
import math
import numpy as np
import numpy.lib.recfunctions as rfn
from settings import *

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

def r2tconvert(chords):
    """
    Translate roman numeral representations  into chord symbols, and add chord symbols into chord labels.
    :param chords:
    :return: rtchords
    """

    # Create scales of all keys
    temp = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    keys = {}
    for i in range(11):
        majtonic = temp[(i*4)%7] + int(i/7)*'+' + int(i%7>5)*'+'
        mintonic = temp[(i*4-2)%7].lower() + int(i/7)*'+' + int(i%7>2)*'+'

        scale = list(temp)
        for j in range(i):
            scale[(j+1)*4%7-1] += '+'
        majscale = scale[(i*4)%7:] + scale[:(i*4)%7]
        minscale = scale[(i*4+5)%7:] + scale[:(i*4+5)%7]
        minscale[6] += '+'
        keys[majtonic] = majscale
        keys[mintonic] = minscale

    for i in range(1, 9):
        majtonic = temp[(i*3)%7] + int(i/7)*'-' + int(i%7>1)*'-'
        mintonic = temp[(i*3-2)%7].lower() + int(i/7)*'-' + int(i%7>4)*'-'

        scale = list(temp)
        for j in range(i):
            scale[(j+2)*3%7] += '-'
        majscale = scale[(i*3)%7:] + scale[:(i*3)%7]
        minscale = scale[(i*3+5)%7:] + scale[:(i*3+5)%7]
        if len(minscale[6]) == 1:
            minscale[6] += '+'
        else:
            minscale[6] = minscale[6][:-1]

        keys[majtonic] = majscale
        keys[mintonic] = minscale

    # Translate chords
    outputQ = {'M':'M', 'm':'m', 'M7':'M7', 'm7':'m7', 'D7':'7', 'a':'aug', 'd':'dim', 'd7':'dim7', 'h7':'m7(b5)','a6':'7'}
    tchords = []
    for rchord in chords:
        key = str(rchord['key'])
        str_degree = str(rchord['degree'])

        if '/' not in str_degree: # case: not secondary chord
            if len(str_degree) == 1: # case: degree = x
                degree = int(float(str_degree))
                root = keys[key][degree-1]
            elif len(str_degree) == 2 and ('+' in str_degree[0] or '-' in str_degree[0]): # case: degree = -x or +x
                if str(rchord['quality']) != 'a6': # case: chromatic chord, -x
                    degree = int(float(str_degree[1]))
                    root = keys[key][abs(degree)-1]
                    if '+' not in root:
                        root += str_degree[0]
                    else:
                        root = root[:-1]
                else: # case: augmented 6th
                    degree = 6
                    root = keys[key][degree-1]
                    if str(rchord['key'])[0].isupper(): # case: major key
                        if '+' not in root:
                            root += '-'
                        else:
                            root = root[:-1]
            elif len(str_degree) == 2 and ('+' in str_degree[1] or '-' in str_degree[1]): # case: degree = x+
                degree = int(float(str_degree[0]))
                root = keys[key][degree - 1]

        elif '/' in str_degree: # case: secondary chord
            degree = str_degree
            if '+' not in degree.split('/')[0]:
                n = int(degree.split('/')[0]) # numerator
            else:
                n = 6
            d = int(degree.split('/')[1]) # denominator
            if d > 0:
                key2 = keys[key][d-1] # secondary key
            else:
                key2 = keys[key][abs(d)-1] # secondary key
                if '+' not in key2:
                    key2 += '-'
                else:
                    key2 = key2[:-1]

            if '+' in degree.split('/')[0]:
                n = 6
            root = keys[key2][n-1]
            if '+' in degree.split('/')[0]:
                if key2.isupper(): # case: major key
                    if '+' not in root:
                        root += '-'
                    else:
                        root = root[:-1]

        # Re-translate root for enharmonic equivalence
        if '++' in root: # if root = x++
            root = temp[(temp.index(root[0]) + 1)%7]
        elif '--' in root: # if root = x--
            root = temp[(temp.index(root[0]) - 1) % 7]

        if '-' in root: # case: root = x-
            if ('F' not in root) and ('C' not in root): # case: root = x-, and x != F and C
                root = temp[((temp.index(root[0]))-1)%7] + '+'
            else:
                root = temp[((temp.index(root[0]))-1)%7] # case: root = x-, and x == F or C
        elif ('+' in root) and ('E' in root or 'B' in root): # case: root = x+, and x == E or B
            root = temp[((temp.index(root[0]))+1)%7]

        quality = outputQ[str(rchord['quality'])]
        tchord = root + quality
        tchords.append(tchord)

    tchords = np.array(tchords, dtype= [('tchord', '<U10')])
    rtchords = rfn.merge_arrays((chords,tchords), flatten=True, usemask=False) # merge rchords and tchords into one structured array

    return rtchords

def load_phrase_labels(directory="./BPS_FH_Dataset/", resolution=8):
    phrases = [None for _ in range(32)]
    for i in range(32):
        fileDir = directory + str(i+1) + "/phrases.xlsx"
        workbook = xlrd.open_workbook(fileDir)
        sheet = workbook.sheet_by_index(0)
        phrase = []
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            cols[0] = float(cols[0])
            cols[1] = float(cols[1])
            phrase.append(tuple(cols))
        phrases[i] = phrase

    return phrases

def load_chord_labels(directory="./BPS_FH_Dataset/"):
    """
    Load chords of each piece and add chord symbols into the labels.
    :param directory: the path of the dataset
    :return: chord_labels
    """

    dt = [('onset', 'float'), ('end', 'float'), ('key', '<U10'), ('degree', '<U10'), ('quality', '<U10'), ('inversion', 'int'), ('rchord', '<U10')] # datatype
    chord_labels = [None for _ in range(32)]
    for i in range(32):
        fileDir = directory + str(i+1) + "/chords.xlsx"

        workbook = xlrd.open_workbook(fileDir)
        sheet = workbook.sheet_by_index(0)
        chords = []
        for rowx in range(sheet.nrows):
            cols = sheet.row_values(rowx)
            if isinstance(cols[3], float): # if type(degree) == float
                cols[3] = int(cols[3])
            chords.append(tuple(cols))
        chords = np.array(chords, dtype=dt) # convert to structured array
        chord_labels[i] = r2tconvert(chords) # translate rchords into chord symbols

    return chord_labels

def load_events(chords, tones, phrases, directory="./BPS_FH_Dataset/", resolution=8):
    bar_res = resolution * 4
    dt = [('onset', 'float'), ('pitch', 'int'), ('mPitch', 'int'), ('duration', 'float'), ('staffNum', 'int'), ('measure', 'int')] # datatype
    pieces = [None for _ in range(32)]
    for i in range(32):
        fileDir = directory + str(i+1) + "/notes.csv"
        notes = np.genfromtxt(fileDir, delimiter=',', dtype=dt) # read notes from .csv file
        length = math.ceil((max(notes[-20:]['onset'] + notes[-20:]['duration']) - notes[0]['onset'])*resolution) # length of the piece
        tdeviation = abs(notes[0]['onset']) # deviation of start time
        note_events = []
        for note in notes:
            pitch = note['pitch']
            start = int(round((note['onset'] + tdeviation)*resolution))
            dur = int(round(note['duration']*resolution))
            note_events += [Item(name="note", start=start, end=start+dur, pitch=pitch, chord=-1, dur=dur, tone=-1)]
        
        bar_events = [Item(name="bar", start=j*bar_res, end=(j+1)*bar_res, pitch=-1, chord=-1, dur=bar_res, tone=-1) for j in range(math.ceil((0.001 + note_events[-1].start - 0)/32))]
        chord_events = [Item(name="chord", start=j[0], end=j[1], pitch=-1, chord=j[2], dur=j[1]-j[0], tone=-1) for j in chords[i]]
        tone_events = [Item(name="tone", start=j[0], end=j[1], pitch=-1, chord=-1, dur=j[1]-j[0], tone=j[2]) for j in tones[i]]
        phrase_events = [Item(name="phrase", start=j[0], end=j[1], pitch=-1, chord=-1, dur=-1, tone=j[2]) for j in phrases[i]]
        
        all_events = note_events + bar_events + chord_events + tone_events + phrase_events
        all_events = sorted(all_events, key=lambda x:(x.start, x.pitch, x.chord, x.tone, x.dur))
        pieces[i] = all_events
        
    return pieces

def data_aug(item):
    # up 6 down 5
    event_seq = []
    for i in range(12):
        pitch_change = i - 5
        tmp = []
        for i in item:
            if i.name == 'bar' or i.name == 'phrase':
                tmp.append(i)
            elif i.name == 'note':
                tmp.append(Item(name="note", start=i.start, end=i.end, pitch=i.pitch+pitch_change, chord=i.chord, dur=i.dur, tone=i.tone))
            elif i.name == 'chord':
                if i.chord + 9*pitch_change < 0:
                    tmp.append(Item(name="chord", start=i.start, end=i.end, pitch=i.pitch, chord=108+i.chord+9*pitch_change, dur=i.dur, tone=i.tone))
                elif i.chord + 9*pitch_change >= 108:
                    tmp.append(Item(name="chord", start=i.start, end=i.end, pitch=i.pitch, chord=i.chord+9*pitch_change-108, dur=i.dur, tone=i.tone)) 
                else:
                    tmp.append(Item(name="chord", start=i.start, end=i.end, pitch=i.pitch, chord=i.chord+9*pitch_change, dur=i.dur, tone=i.tone))
            elif i.name == 'tone':
                newtone = i.tone + 2*pitch_change
                if newtone < 0:
                    tmp.append(Item(name="tone", start=i.start, end=i.end, pitch=i.pitch, chord=i.chord, dur=i.dur, tone=24+newtone))
                elif newtone >= 24:
                    tmp.append(Item(name="tone", start=i.start, end=i.end, pitch=i.pitch, chord=i.chord, dur=i.dur, tone=newtone-24)) 
                else:
                    tmp.append(Item(name="tone", start=i.start, end=i.end, pitch=i.pitch, chord=i.chord, dur=i.dur, tone=newtone))

     
            
        event_seq.append(tmp)
        
    return event_seq

def event2multihot(item, bpm = 120):
    ### Type ###
    # 0 : Bar & Pos
    # 1 : Dur & Pitch
    # 2 : Tone
    # 3 : Chord
    # 4 : Phrase

    ### Bar & Pos ###
    # 0 : ignore
    # 1-32 : Pos
    # 33 : Bar

    ### Tone ###
    # 0 : ignore
    # 1-24 : chord

    ### Chord ###
    # 0 : ignore
    # 1-108 : chord

    ### Dur ###
    # 0 : ignore
    # 1-64 : Dur

    ### Pitch ###
    # 0 : ignore
    # 1-128 : Pitch

    type_len = 5
    barpos_len = 1 + 32 + 1
    dur_len = 64 + 1
    pitch_len = 128 + 1
    tone_len = 24 + 1
    chord_len = 108 + 1

    note_unit = 1/(8*(bpm/60))
    
    multihotvec = []
    curpos = -1
    curchordtype = -1
    curtonetype = -1
    for i in item:
        type_vec = [0]*type_len
        barpos_vec = [0]*barpos_len
        tone_vec = [0]*tone_len
        chord_vec = [0]*chord_len
        dur_vec = [0]*dur_len
        pitch_vec = [0]*pitch_len
        tone_chroma = [0]*12
        chord_chroma = [0]*12

        if i.name == 'bar':
            type_vec[0] = 1
            barpos_vec[33] = 1
            tone_vec[0] = 1
            chord_vec[0] = 1 
            dur_vec[0] = 1
            pitch_vec[0] = 1
            
            tmpvec = type_vec + barpos_vec + tone_vec + chord_vec + dur_vec + pitch_vec + tone_chroma + chord_chroma
            multihotvec.append(tmpvec)
        
        elif i.name == 'phrase':
            type_vec[4] = 1
            barpos_vec[0] = 1
            tone_vec[0] = 1
            chord_vec[0] = 1 
            dur_vec[0] = 1
            pitch_vec[0] = 1
            
            tmpvec = type_vec + barpos_vec + tone_vec + chord_vec + dur_vec + pitch_vec + tone_chroma + chord_chroma
            multihotvec.append(tmpvec)
        
        elif i.name == 'chord':
            type_vec[3] = 1
            barpos_vec[0] = 1
            tone_vec[0] = 1
            chord_vec[int(i.chord) + 1] = 1 
            dur_vec[0] = 1
            pitch_vec[0] = 1

            curchordtype = int(i.chord)
            
            tmpvec = type_vec + barpos_vec + tone_vec + chord_vec + dur_vec + pitch_vec + tone_chroma + chord_chroma
            multihotvec.append(tmpvec)
            
        elif i.name == 'tone':
        	### only add tone event when tone changes
            if curtonetype != int(i.tone):
                type_vec[2] = 1
                barpos_vec[0] = 1
                tone_vec[int(i.tone) + 1] = 1
                chord_vec[0] = 1
                dur_vec[0] = 1
                pitch_vec[0] = 1

                curtonetype = int(i.tone)

                tmpvec = type_vec + barpos_vec + tone_vec + chord_vec + dur_vec + pitch_vec + tone_chroma + chord_chroma
                multihotvec.append(tmpvec)
        
        elif i.name == 'note':
        	### check if pos changes
            if int(i.start % 32) + 1 != curpos:              
                type_vec[0] = 1
                barpos_vec[int(i.start % 32) + 1] = 1
                tone_vec[0] = 1
                chord_vec[0] = 1
                dur_vec[0] = 1
                pitch_vec[0] = 1

                curpos = int(i.start % 32) + 1

                tmpvec = type_vec + barpos_vec + tone_vec + chord_vec + dur_vec + pitch_vec + tone_chroma + chord_chroma
                multihotvec.append(tmpvec)
                
            type_vec = [0]*type_len
            barpos_vec = [0]*barpos_len
            tone_vec = [0]*tone_len
            chord_vec = [0]*chord_len
            dur_vec = [0]*dur_len
            pitch_vec = [0]*pitch_len

            type_vec[1] = 1
            barpos_vec[0] = 1
            tone_vec[0] = 1
            chord_vec[0] = 1

            dur = int(i.dur)
            ### ignore ornaments
            if dur == 0: 
                continue
            ### cut notes where note length > 64 (2 full notes)
            elif dur > 64: 
                dur_vec[64] = 1
            else:
                dur_vec[dur] = 1
            
            pitch_vec[int(i.pitch) + 1] = 1
            tone_chroma = all_tone_chroma[curtonetype]
            chord_chroma = all_chord_chroma[curchordtype]
            
            tmpvec = type_vec + barpos_vec + tone_vec + chord_vec + dur_vec + pitch_vec + tone_chroma + chord_chroma
            multihotvec.append(tmpvec)
            
    return multihotvec

### eight 32nd notes in one beat ###
resolution = 8 

### chord labels ###
### [onset, end, key, degree, quality, inversion, rchord, tchord] ###
chord_labels = load_chord_labels()

### [onset, end, sonataform_1, sonataform_2, phrase] ###
phrase_labels = load_phrase_labels()

### get all phrase types ###
phrasetype = np.unique(np.array([j[-1] for i in phrase_labels for j in i ]))

### assign phrase type to num ###
### phrase:num ###
phrase_dict = {phrasetype[i]:i for i in range(len(phrasetype))}

### onset, end, chord/tone/phrase ###
chords = [None for _ in range(32)]
tones = [None for _ in range(32)]
phrases = [None for _ in range(32)]
for i in range(32):
    tdeviation = abs(chord_labels[i]['onset'][0])
    chords[i] = np.array([[(j[0]+tdeviation)*resolution, (j[1]+tdeviation)*resolution, allchordtype_dict[j[-1]]] for j in chord_labels[i]])
    tones[i] = np.array([[(j[0]+tdeviation)*resolution, (j[1]+tdeviation)*resolution, alltonetype_dict[j[2]]] for j in chord_labels[i]])
    phrases[i] = [[(float(j[0])+tdeviation)*resolution, (float(j[1])+tdeviation)*resolution, phrase_dict[j[-1]]] for j in phrase_labels[i]]
    
### make into event sequences ###
pieces = load_events(chords, tones, phrases)

### train pieces
train_indices = [4, 11, 16, 20, 26, 31, 3, 8, 12, 17, 23, 21, 27, 29, 30, 10, 1, 2, 7, 18, 28, 15, 25, 5, 19, 22, 14, 19, 24, 6]

### test pieces
test_indices = [0, 13]

train = []
for i in train_indices:
    aug_event = data_aug(pieces[i])
    for j in aug_event:
         for k in event2multihot(j):
            train.append(k)
    
test = []
for i in test_indices:
    aug_event = data_aug(pieces[i])
    for j in aug_event:
         for k in event2multihot(j):
            test.append(k)
            
train = np.array(train)
test = np.array(test)
np.save('train_BPSFH.npy', train)
np.save('test_BPSFH.npy', test)