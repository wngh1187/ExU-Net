import torch
import torchaudio

class LogMelspectrogram():
    """Extract Log-Melspectrogram from raw waveform using torchaudio.
    Note that this module automatically synchronizes device with input tensor.
    """
    def __init__(self, winlen, winstep, nfft, samplerate, nfilts, premphasis, winfunc):
        super(LogMelspectrogram, self).__init__()
        
        self.device = 'cpu'
        self.premphasis = premphasis
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            n_mels=nfilts,
            n_fft=nfft,
            win_length=winlen,
            hop_length=winstep,
            sample_rate=samplerate,
            window_fn=winfunc
        )
    
    def __call__(self, x):
        # synchronize device
        if self.device != x.device:
            self.device = x.device
            self.mel_spec.to(x.device)
            
        # pre-emphasis
        if self.premphasis is not None:
            x = x[:, 1:] - self.premphasis * x[:, 0:-1]
        
        
        # melspectrogram
        x = x.type(torch.float32)
        x = self.mel_spec(x)

        # log-mel
        x = x + 1e-6
        x = torch.log(x)

        return x