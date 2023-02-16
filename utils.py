import matplotlib.pyplot as plt 
import tensorflow as tf
import pandas as pd
import numpy as np
import IPython.display as ipd
import librosa
from librosa.display import specshow
from sklearn.model_selection import train_test_split

# Matplotlib
def set_default(style=['dark_background'], figsize=(15, 8), dpi=100):
    plt.style.use(style)
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    
def display_distribution(df, column_name, labels=None):
    """Displays the distribution of a column in a DataFrame."""
    column_data = df[column_name]
    column_counts = column_data.value_counts(normalize=True)
    
    plt.bar(column_counts.index, column_counts.values, align='center')
    if labels:
        plt.gca().set_xticks(labels)
    plt.xlabel(column_name)
    plt.ylabel('Proportion')
    plt.title(f'Distribution of {column_name}')
    plt.show()
    
def plot_signal(signal, time, n_channels):
    if n_channels == 1:
        plt.plot(time, signal, label="Mono")
    elif n_channels == 2:
        plt.plot(time, signal[:, 0], label="Left channel")
        plt.plot(time, signal[:, 1], label="Right channel")
        plt.legend()    
    
def librosa_audio_signal(signal, sampling_rate, signal_path, class_name):
    n_samples = signal.shape[0]
    n_channels = signal.shape[1] if signal.ndim == 2 else 1
    
    duration = n_samples / sampling_rate
    
    time = np.linspace(0., duration, n_samples)
    
    X = librosa.stft(signal) # represents a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) over short overlapping windows
    X_dB = librosa.amplitude_to_db(np.abs(X)) # convert an amplitude spectrogram to dB-scaled spectrogram
    
    mel_spect = librosa.feature.melspectrogram(y=signal, sr=sampling_rate)
    
    mfccs = librosa.feature.mfcc(y=signal, sr=sampling_rate)
      
    display(plt.gcf())
    display(ipd.Audio(signal_path, rate=sampling_rate))
    print(
        f'class name: {class_name}',
        f'The signal is {duration:.1f}s long.',
        f'The sampling rate is {sampling_rate * 1e-3} kHz, with {n_samples} samples.',
        f'There is {n_channels} audio channel(s).',
        sep='\n')
    
    plt.clf()
    
    plt.subplot(4, 1, 1)
    plot_signal(signal, time, n_channels)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Signal')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.subplot(4, 1, 2)
    specshow(X_dB, sr=sampling_rate, x_axis='time', y_axis='hz')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Spectrogram');
    
    plt.subplot(4, 1, 3)
    specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Mel Spectrogram');
    
    plt.subplot(4, 1, 4)
    specshow(mfccs, x_axis='time')
    plt.title('MFCCs');
    
    plt.tight_layout()

def librosa_get_examples(metadata, nb_examples, classes):
    data = []
    for id in list(metadata["classID"].unique()):
        data.extend(metadata.query("classID == @id").head(nb_examples).index)
            
    for row in data:
        signal, sr = librosa.load(metadata.iloc[row].signal_path)
        infos = {
            'signal': signal,
            'sampling_rate': sr,
            'signal_path': metadata.iloc[row].signal_path,
            'class': metadata.iloc[row]['class']
        }
        if infos['class'] in classes:
            librosa_audio_signal(
                infos['signal'],
                infos['sampling_rate'],
                infos['signal_path'],
                infos['class']
            )
    
# Tensorflow
def try_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        return print("GPU available")
    else:
        return print("No GPU available")

# Pandas
def load_data(path, index_col=False):
    try:
        if path.endswith('.csv'):
            return pd.read_csv(path, low_memory=False, index_col=index_col)
        elif path.endswith('.pkl'):
            return pd.read_pickle(path)
    except FileNotFoundError:
        print("File not found")

# Librosa
def normalize_w_librosa(data, norm=2):
    return librosa.util.normalize(data.tolist(), norm=norm)

# For models
def generate_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(
        f'Shape of training samples = {X_train.shape}',
        f'Shape of validation samples = {X_val.shape}', # the inverse of the period
        f'Shape of testing samples = {X_test.shape}',
        sep='\n')
    return X_train, X_test, X_val, y_train, y_test, y_val 

def display_best_hyperparameters(grid_search):
    print("Best hyperparameters:")
    for key, value in grid_search.best_params_.items():
        print(f"{key}: {value}")
    print("Best accuracy:", grid_search.best_score_)
    
def evaluate_model(model, X, y, batch_size=32):
    print("Evaluate on test data")
    results = model.evaluate(X, y, batch_size=batch_size)
    print("test loss, test acc:", results)
    return results[1]

def display_leaning_curves(history):
    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca()
    plt.show()        
        