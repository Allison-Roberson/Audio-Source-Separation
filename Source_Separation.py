import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import IPython.display as ipd
import soundfile as sf

# path to audio file
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

epsilon = 1e-10 # error to introduce
V = sound_stft_Magnitude + epsilon
K, N = np.shape(V)

#number of sources
S = 2
print(f"S = {S} : Number of Sources to separate")

def divergence(V,W,H, beta = 2):
   
    """
    beta = 2 : Euclidean cost function
    beta = 1 : Kullback-Leibler cost function
    beta = 0 : Itakura-Saito cost function
    """
   
    if beta == 0 : return np.sum( V/(W@H) - math.log10(V/(W@H)) -1 )
   
    if beta == 1 : return np.sum( V*math.log10(V/(W@H)) + (W@H - V))
   
    if beta == 2 : return 1/2*np.linalg.norm(W@H-V)
   
def plot_NMF_iter(W, H,beta,iteration = None):
   
    f = plt.figure(figsize=(4,4))
    f.suptitle(f"NMF Iteration {iteration}, for beta = {beta}", fontsize=8,)
   
    # definitions for the axes
    V_plot = plt.axes([0.35, 0.1, 1, 0.6])
    H_plot = plt.axes([0.35, 0.75, 1, 0.15])
    W_plot = plt.axes([0.1, 0.1, 0.2, 0.6])

    D = librosa.amplitude_to_db(W@H, ref = np.max)

    librosa.display.specshow(W,y_axis = 'hz', sr=sr, hop_length=HOP,x_axis ='time',cmap= matplotlib.cm.jet, ax=W_plot)
    librosa.display.specshow(H,y_axis = 'hz', sr=sr, hop_length=HOP,x_axis ='time',cmap= matplotlib.cm.jet, ax=H_plot)
    librosa.display.specshow(D,y_axis = 'hz', sr=sr, hop_length=HOP,x_axis ='time',cmap= matplotlib.cm.jet, ax=V_plot)

    W_plot.set_title('Dictionnary U', fontsize=10)
    H_plot.set_title('Temporal activations V', fontsize=10)

    W_plot.axes.get_xaxis().set_visible(False)
    H_plot.axes.get_xaxis().set_visible(False)
    V_plot.axes.get_yaxis().set_visible(False)
   
def NMF(V, S, beta = 2,  threshold = 0.01, MAXITER = 5000, display = True , displayEveryNiter = None):
   
    """
    inputs :
    --------
   
        V         : Mixture signal : |TFST|
        S         : The number of sources to extract
        beta      : Beta divergence considered, default=2 (Euclidean)
        threshold : Stop criterion
        MAXITER   : The number of maximum iterations, default=1000
        display   : Display plots during optimization :
        displayEveryNiter : only display last iteration
                                                           
   
    outputs :
    ---------
     
        W : dictionary matrix [KxS], W>=0
        H : activation matrix [SxN], H>=0
        cost_function : the optimised cost function over iterations
       
   Algorithm :
   -----------
   
    1) Randomly initialize W and H matrices
    2) Multiplicative update of W and H
    3) Repeat step (2) until convergence or after MAXITER
   
       
    """
    counter  = 0
    cost_function = []
    beta_divergence = 1
   
    K, N = np.shape(V)
   
    # Initialisation of W and H matrices : The initialization is generally random
    W = np.abs(np.random.normal(loc=0, scale = 2.5, size=(K,S)))    
    H = np.abs(np.random.normal(loc=0, scale = 2.5, size=(S,N)))
   
    # Plotting the first initialization
    if display == True : plot_NMF_iter(W,H,beta,counter)


    while beta_divergence >= threshold and counter <= MAXITER:
       
        # Update of W and H
        H *= (W.T@(((W@H)**(beta-2))*V))/(W.T@((W@H)**(beta-1)) + 10e-10)
        W *= (((W@H)**(beta-2)*V)@H.T)/((W@H)**(beta-1)@H.T + 10e-10)
       
       
        # Compute cost function
        beta_divergence =  divergence(V,W,H, beta = 2)
        cost_function.append( beta_divergence )
       
        if  display == True  and counter%displayEveryNiter == 0  : plot_NMF_iter(W,H,beta,counter)

        counter +=1
   
    if counter -1 == MAXITER : print(f"Stop after {MAXITER} iterations.")
    else : print(f"Convergeance after {counter-1} iterations.")
       
    return W,H, cost_function

beta = 2
W, H, cost_function = NMF(V,S,beta = beta, threshold = 0.01, MAXITER = 5000, display = True , displayEveryNiter = 1000)

# Plot the cost function
plt.figure(figsize=(5,3))
plt.plot(cost_function)
plt.title("Cost Function")
plt.xlabel("Number of iteration")
plt.ylabel(f"Beta Divergence for beta = {beta} ")

#After NMF, each audio source S can be expressed as a frequency mask over time
f, axs = plt.subplots(nrows=1, ncols=S,figsize=(20,5))
filtered_spectrograms = []
for i in range(S):
    axs[i].set_title(f"Frequency Mask of Audio Source s = {i+1}")
    # Filter eash source components
    WsHs = W[:,[i]]@H[[i],:]
    filtered_spectrogram = W[:,[i]]@H[[i],:] /(W@H) * V
    # Compute the filtered spectrogram
    D = librosa.amplitude_to_db(filtered_spectrogram, ref = np.max)
    # Show the filtered spectrogram
    librosa.display.specshow(D,y_axis = 'hz', sr=sr,hop_length=HOP,x_axis ='time',cmap= matplotlib.cm.jet, ax = axs[i])
   
    filtered_spectrograms.append(filtered_spectrogram)
   
reconstructed_sounds = []
for i in range(S):
    reconstruct = filtered_spectrograms[i] * np.exp(1j*sound_stft_Angle)
    new_sound = librosa.istft(reconstruct, n_fft = FRAME, hop_length = HOP)
    reconstructed_sounds.append(new_sound)

# Ensure all reconstructed sounds have the same length
min_length = min(len(sound) for sound in reconstructed_sounds)    
reconstructed_sounds = [sound[:min_length] for sound in reconstructed_sounds]

# Plotting the waveform
colors = ['r', 'g','b','c']
fig, ax = plt.subplots(nrows=S, ncols=1, sharex=True, figsize=(10, 8))
for i in range(S):
    librosa.display.waveshow(reconstructed_sounds[i], sr=sr, color = colors[i], ax=ax[i],label=f'Source {i}',x_axis='time')
    ax[i].set(xlabel='Time [s]')
    ax[i].legend()

#exporting new audio files of each source    
for i, sound in enumerate(reconstructed_sounds):
    output_file=f'reconstructed_sound_{i}.wav'
    sf.write(output_file, sound, sr)
    print(f'Reconstructed sound {i} saved as {output_file}')