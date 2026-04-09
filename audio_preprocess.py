import librosa
import numpy as np
import soundfile as sf

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def load_audio(self, file_path):
        """加载音频，统一采样率"""
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        return y, sr

    def denoise_audio(self, y):
        """简单降噪处理"""
        # 预加重
        y = librosa.effects.preemphasis(y)
        # 移除静音段
        y, _ = librosa.effects.trim(y, top_db=20)
        return y

    def extract_mel_spectrogram(self, y):
        """提取梅尔频谱特征"""
        mel_spect = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # 转换为对数梅尔频谱
        log_mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        return log_mel_spect

    def process_audio_file(self, file_path, save_path=None):
        """完整处理单个音频文件"""
        y, sr = self.load_audio(file_path)
        y = self.denoise_audio(y)
        mel_spect = self.extract_mel_spectrogram(y)
        
        if save_path:
            sf.write(save_path, y, sr)
        
        return mel_spect

if __name__ == "__main__":
    # 测试用例
    preprocessor = AudioPreprocessor()
    # 替换成你自己的音频文件路径
    test_file = "test_audio.wav"
    mel_spec = preprocessor.process_audio_file(test_file)
    print(f"梅尔频谱形状: {mel_spec.shape}")
