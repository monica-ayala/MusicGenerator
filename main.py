import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from keras import backend as k
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.optimizers import Adamax
from collections import Counter
import random
import music21
from music21 import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

## BEGINING OF LOADING THE DATA 

filepath = "training_data/Piano/" 

piano_files= []
for i in os.listdir(filepath):
    if i.endswith(".mid"):
        tr = filepath+i
        midi = converter.parse(tr)
        piano_files.append(midi)
            
def extract_notes(file):
    notes = []
    pick = None
    for j in file:
        songs = instrument.partitionByInstrument(j)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))

    return notes

Corpus= extract_notes(piano_files)
print("# of Notes in the piano dataset:", len(Corpus))

def chords_n_notes(Snippet):
    Melody = []
    offset = 0 
    for i in Snippet:
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".") 
            notes = [] 
            for j in chord_notes:
                inst_note=int(j)
                note_snip = note.Note(inst_note)            
                notes.append(note_snip)
                chord_snip = chord.Chord(notes)
                chord_snip.offset = offset
                Melody.append(chord_snip)
        else: 
            note_snip = note.Note(i)
            note_snip.offset = offset
            Melody.append(note_snip)
        offset += 1
    melody_stream = stream.Stream(Melody)   
    return melody_stream

count_num = Counter(Corpus)
print("Total unique notes in the Corpus:", len(count_num))

Notes = list(count_num.keys())
Recurrence = list(count_num.values())
print("Average appearance of one note - ", sum(Recurrence) / len(Recurrence))
print("Note that appeared the most - ", max(Recurrence), "times")
print("Note that appeared the least - ", min(Recurrence), "time")

sns.set_style("whitegrid")

plt.figure(figsize=(12, 6), facecolor="#FFD1DC")
bins = np.arange(0, len(Recurrence) + 1, 1)
plt.hist(range(len(Recurrence)), bins=bins, weights=Recurrence, color="#CF578A", edgecolor="white", alpha=0.7)
plt.axhline(y=100, color="#DBACC1", linestyle="--", linewidth=2, label="Threshold")
plt.title("Frequency of Notes in the Corpus", fontsize=16)
plt.xlabel("Note ID (Number associated to note)", fontsize=12)
plt.ylabel("Times it appears in the Corpus", fontsize=12)
plt.legend()
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

rare_note = []
for index, (key, value) in enumerate(count_num.items()):
    if value < 100:
        m =  key
        rare_note.append(m)
        
for element in Corpus:
    if element in rare_note:
        Corpus.remove(element)
        
print("Notes that appear less than 100 times - ", len(rare_note))

print("Length of Corpus after elemination the rare notes:", len(Corpus))

## BEGINING OF PREPROCESSING

symb = sorted(list(set(Corpus)))

L_corpus = len(Corpus) 
L_symb = len(symb) 

mapping = dict((c, i) for i, c in enumerate(symb))
reverse_mapping = dict((i, c) for i, c in enumerate(symb))

print("Characters in Corpus - " , L_corpus)
print("Unique characters in Corpus - " , L_symb)

length = 40 # Can be modified
features = []
targets = []
for i in range(0, L_corpus - length, 1):
    feature = Corpus[i:i + length]
    target = Corpus[i + length]
    features.append([mapping[j] for j in feature])
    targets.append(mapping[target])
    
L_datapoints = len(targets)
print("Sequences in the Corpus - ", L_datapoints)

X = (np.reshape(features, (L_datapoints, length, 1)))/ float(L_symb)
y = tf.keras.utils.to_categorical(targets) 

train_chords, chords_seed, train_target, target_seed = train_test_split(X, y, test_size=0.6)

## BEGINING OF MODEL IMPLEMENTAITON
model_file_path = "saved_model.keras"

if os.path.exists(model_file_path):
    model = keras.models.load_model(model_file_path)
else:
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(y.shape[1], activation='softmax'))
    
    opt = Adamax(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt) 

    history = model.fit(train_chords, train_target, batch_size=150, epochs=200)
    model.save(model_file_path)
    
    # Plotting the learning plot for model loss
    plt.figure(figsize=(15, 4), facecolor="#FFD1DC")
    plt.title("Learning Evaluation: Loss over epoch", fontsize=16)
    sns.lineplot(data=history_df["loss"], color="#CF578A", linewidth=2)
    plt.ylabel("Training Loss", fontsize=12)
    plt.xlabel("Number of epoch", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

model.summary()

from music21 import stream

def Malody_Generator(model, chords_seed, reverse_mapping, length, L_symb, Note_Count, user_input, normal):
    
    seed = chords_seed[np.random.randint(0, len(chords_seed) - 1)]
    Music = ""
    Notes_Generated = []
    
    if(not normal):
        user_input_indices = user_input
        user_input_normalized = np.array(user_input_indices) / float(L_symb)
        user_input_normalized = user_input_normalized[:length]
        seed[-len(user_input_normalized):] = user_input_normalized.reshape(-1, 1)

    for i in range(Note_Count):
        seed = seed.reshape(1, length, 1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.5  # Can be changed
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index / float(L_symb)
        Notes_Generated.append(index)
        Music = [reverse_mapping[char] for char in Notes_Generated]
        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]

    Melody = chords_n_notes(Music)
    return Music, Melody

user_input = [0,13,128,0,13,12,11,18,1,2,0,13,128,0,13,12,11,18,1,2,0,13,128,0,13,12,11,18,1,2,0,13,128,0,13,12,11,18,1,2,0]
generated_music, generated_melody = Malody_Generator(model, chords_seed, reverse_mapping, length, L_symb, 100, user_input, 0)
generated_music_normal, generated_melody_normal = Malody_Generator(model, chords_seed, reverse_mapping, length, L_symb, 100, [], 1)
generated_melody.write('midi','Melody_Generated.mid')
generated_melody_normal.write('midi','Melody_Generated_1.mid')

## RESULTS

def play_music(file):
    b = music21.converter.parse(file)

    import random
    keyDetune = []
    for i in range(127):
        keyDetune.append(random.randint(-30, 30))

    for n in b.flatten().notesAndRests:
        if isinstance(n, note.Note):
            n.pitch.microtone = keyDetune[n.pitch.midi]
        elif isinstance(n, chord.Chord):
            for note_in_chord in n.pitches:
                note_in_chord.microtone = keyDetune[note_in_chord.midi]

    sp = music21.midi.realtime.StreamPlayer(b)
    sp.play()
    
play_music("Melody_Generated_1.mid")