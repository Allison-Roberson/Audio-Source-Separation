import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf
from sklearn.decomposition import NMF

# path to aud
audio_file = r'\Users\AHoff\OneDrive\Documents\Python Scripts\guitar_drum.wav'
sample_rate = 44100
audio_sound, sr = librosa.load(audio_file, sr = sample_rate)
ipd.Audio(audio_sound, rate=sr)

#plotting the waveform
fig, ax = plt.subplots(figsize=(10, 3))
librosa.display.waveshow(audio_sound, sr=sr)
ax.set(title='The sound waveform', xlabel='Time [s]')
plt.show()

#plotting the spectrogram
FRAME = 512
HOP = FRAME // 2
sound_stft = librosa.stft(audio_sound, n_fft = FRAME, hop_length = HOP)
sound_stft_Magnitude = np.abs(sound_stft)
sound_stft_Angle = np.angle(sound_stft)
Spec = librosa.amplitude_to_db(sound_stft_Magnitude, ref = np.max)
librosa.display.specshow(Spec,y_axis = 'hz',sr=sr,hop_length=HOP,x_axis ='time',cmap= matplotlib.cm.jet)
plt.title('Audio spectrogram')

# Obtaining the matrix representation of the spectrogram
spectrogram_matrix = sound_stft_Magnitude.T  # Transpose to have time as rows and frequencies as columns
print("Spectrogram matrix shape:", spectrogram_matrix.shape)
print("Spectrogram matrix rank:", np.linalg.matrix_rank(spectrogram_matrix))

# Printing the actual data matrix of the spectrogram
print("Spectrogram matrix:")
print(spectrogram_matrix)

plt.show()

# Computing the nonnegative rank using NMF
nmf_model = NMF(n_components=np.linalg.matrix_rank(spectrogram_matrix), init='random', random_state=0, max_iter=5000)
W = nmf_model.fit_transform(spectrogram_matrix)
H = nmf_model.components_
reconstructed_matrix = np.dot(W, H)
nonnegative_rank = np.linalg.norm(spectrogram_matrix - reconstructed_matrix, 'fro')
print("Spectrogram matrix nonnegative rank:", nonnegative_rank)

# Assuming 'spectrogram_matrix' contains your spectrogram matrix

# Define the file path
output_file = "spectrogram_matrix.txt"

# Write the matrix to the file
with open(output_file, "w") as f:
    # Iterate over each row in the matrix
    for row in spectrogram_matrix:
        # Convert the row to a string with tab-separated values
        row_str = "\t".join(map(str, row))
        # Write the row to the file
        f.write(row_str + "\n")

print("Spectrogram matrix has been saved to:", output_file)

# Define file paths
W_file = "W_matrix.txt"
H_file = "H_matrix.txt"

# Write W matrix to a file
with open(W_file, "w") as f:
    for row in W:
        row_str = "\t".join(map(str, row))
        f.write(row_str + "\n")

print("W matrix has been saved to:", W_file)

# Write H matrix to a file
with open(H_file, "w") as f:
    for row in H:
        row_str = "\t".join(map(str, row))
        f.write(row_str + "\n")

print("H matrix has been saved to:", H_file)