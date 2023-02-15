import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import IPython.display as ipd
import librosa
from librosa.display import specshow
import tensorflow as tf
from sklearn.model_selection import train_test_split

def set_default(style=['dark_background'], figsize=(15, 8), dpi=100):
    plt.style.use(style)
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)
    

def try_gpu():
    if tf.test.is_gpu_available():
        print("GPU available")
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        return True
    else:
        print("No GPU available")
        return False
    
def load_data(path, pickle=False, index_col=False):
    try:
        if path.endswith('.csv'):
            return pd.read_pickle(path) if pickle else pd.read_csv(path, low_memory=False, index_col=index_col)
    except FileNotFoundError:
        print("File not found")
        
def display_distributions(data):
    plt.hist(data)
    plt.show()       
        
def display_distributions_w_labels(labels, counts):
    plt.bar(labels, counts, align='center')
    plt.gca().set_xticks(labels)
    plt.show()
    
def normalize_w_librosa(data, norm=2):
    return librosa.util.normalize(data.tolist(), norm=norm)

def generate_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    print(
        f'Shape of training samples = {X_train.shape}',
        f'Shape of validation samples = {X_val.shape}', # the inverse of the period
        f'Shape of testing samples = {X_test.shape}',
        sep='\n')
    return X_train, X_test, X_val, y_train, y_test, y_val    

def display_leaning_curves(history):
    pd.DataFrame(history.history).plot()
    plt.grid(True)
    plt.gca()
    plt.show()
    
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

def librosa_audio_signal(signal, sampling_rate, channels, signal_path, class_name, X, X_dB, mfcc):
    T = signal.size / sampling_rate # la longueur du son
    dt = 1 / sampling_rate # durée d'un échantillon -> sampling_rate nb échantillons par secondes
    t = np.r_[0:T:dt]
    
    spec = np.abs(librosa.stft(signal, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    mel_spect = librosa.feature.melspectrogram(y=signal, sr=sampling_rate)
    
    
    display(plt.gcf())
    display(ipd.Audio(signal_path, rate=sampling_rate))
    print(
        f'class name: {class_name}',
        f'x[k] has {signal.size} samples',
        f'the sampling rate is {sampling_rate * 1e-3} kHz',
        f'x(t) is {T:.1f}s long',
        f'the number of channels is {channels}',
        sep='\n')
    
    plt.clf()
    
    plt.subplot(3, 1, 1)
    plt.plot(t, signal)
    plt.xlim([0, T])
    plt.xlabel('time [s]')
    plt.ylabel('amplitude [/]')
    plt.title('Signal')
    plt.setp(plt.gca().get_xticklabels(), visible=False)

    plt.subplot(3, 1, 2)
    specshow(librosa.power_to_db(mel_spect, ref=np.max), sr=sampling_rate, x_axis='time', y_axis='mel');
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    plt.title('Spectrogram');
    
    plt.subplot(3, 1, 3)
    specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    plt.title('Mel Spectrogram');
    
    plt.tight_layout()

def librosa_get_examples(metadata, nb_examples, classes):
    data = []
    for id in list(metadata["classID"].unique()):
        data.extend(metadata.query("classID == @id").head(nb_examples).index)
            
    for row in data:
        signal, sr = librosa.load(metadata.iloc[row].signal_path)
        # Min-max normalization
        signal = librosa.util.normalize(signal, axis=0, norm=np.inf)
        X = librosa.stft(signal)
        X_dB = librosa.amplitude_to_db(np.abs(X))
        mfcc = librosa.feature.mfcc(y=signal, sr=sr)
        infos = {
            'signal': signal,
            'sampling_rate': sr,
            'channels': signal.shape[0] if len(signal.shape) > 1 else 1,
            'signal_path': metadata.iloc[row].signal_path,
            'class': metadata.iloc[row]['class'],
            'X': X,
            'X_dB': X_dB,
            'mfcc': mfcc,
        }
        if infos['class'] in classes:
            librosa_audio_signal(
                infos['signal'],
                infos['sampling_rate'],
                infos['channels'],
                infos['signal_path'],
                infos['class'],
                infos['X'],
                infos['X_dB'],
                infos['mfcc']
            )


    
    