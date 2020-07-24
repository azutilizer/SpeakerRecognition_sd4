import os
import sidekit


def convert_wavto_hd5_from_libri_librosa(corpus_dir, feat_dir, samplerate=8000):
    low_freq = 100
    higer_freq = 4000  # 3800
    if samplerate == 16000:
        low_freq == 100
        higer_freq = 8000

    extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=samplerate,
                                          lower_frequency=low_freq, #200
                                          higher_frequency=higer_freq, #3800
                                          filter_bank="log",
                                          filter_bank_size=40,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=20,
                                          vad="snr", #"snr"
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)

    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        feat_subdir_path = os.path.join(feat_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        if not os.path.exists(feat_subdir_path):
            os.mkdir(feat_subdir_path)
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            spk_feat_dir_path = os.path.join(feat_subdir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            if not os.path.exists(spk_feat_dir_path):
                os.mkdir(spk_feat_dir_path)
            else:
                continue
            print("speaker: {} processing".format(spk_dir))
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                uttr_feat_dir_path = os.path.join(spk_feat_dir_path, uttr_dir)
                if not os.path.isdir(uttr_dir_path):
                    continue
                if not os.path.exists(uttr_feat_dir_path):
                    os.mkdir(uttr_feat_dir_path)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    feat_path = os.path.join(uttr_feat_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    extractor.save_librosa(show=show_name,
                                           channel=0,
                                           input_audio_filename=audio_path,
                                           output_feature_filename=feat_path)

def convert_wav16_to_hd5_from_libri(corpus_dir, feat_dir):
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=16000,
                                          lower_frequency=100, #200
                                          higher_frequency=8000, #3800
                                          filter_bank="log",
                                          filter_bank_size=40,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=20,
                                          vad="snr",
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)


    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        feat_subdir_path = os.path.join(feat_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        if not os.path.exists(feat_subdir_path):
            os.mkdir(feat_subdir_path)
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            spk_feat_dir_path = os.path.join(feat_subdir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            if not os.path.exists(spk_feat_dir_path):
                os.mkdir(spk_feat_dir_path)
            else:
                continue
            print("speaker: {} processing".format(spk_dir))
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                uttr_feat_dir_path = os.path.join(spk_feat_dir_path, uttr_dir)
                if not os.path.isdir(uttr_dir_path):
                    continue
                if not os.path.exists(uttr_feat_dir_path):
                    os.mkdir(uttr_feat_dir_path)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    feat_path = os.path.join(uttr_feat_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    extractor.save(show=show_name,
                                   channel=0,
                                   input_audio_filename=audio_path,
                                   output_feature_filename=feat_path)

def convert_wav_to_hd5(corpus_dir, feat_dir):
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=None,
                                          lower_frequency=133.3, #200
                                          higher_frequency=4000, #3800
                                          filter_bank="log",
                                          filter_bank_size=40,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=20,
                                          vad="snr",
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)


    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        feat_subdir_path = os.path.join(feat_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        if not os.path.exists(feat_subdir_path):
            os.mkdir(feat_subdir_path)
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            spk_feat_dir_path = os.path.join(feat_subdir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            if not os.path.exists(spk_feat_dir_path):
                os.mkdir(spk_feat_dir_path)
            else:
                continue
            print("speaker: {} processing".format(spk_dir))
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                uttr_feat_dir_path = os.path.join(spk_feat_dir_path, uttr_dir)
                if not os.path.isdir(uttr_dir_path):
                    continue
                if not os.path.exists(uttr_feat_dir_path):
                    os.mkdir(uttr_feat_dir_path)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    feat_path = os.path.join(uttr_feat_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    extractor.save(show=show_name,
                                   channel=0,
                                   input_audio_filename=audio_path,
                                   output_feature_filename=feat_path)


def convert_wav_to__plp_hd5(corpus_dir, feat_dir):
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=None,
                                          filter_bank="log",
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=13,
                                          vad="snr",
                                          snr=40,
                                          pre_emphasis=0.97,
                                          feature_type='plp',
                                          rasta_plp=True,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)


    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        feat_subdir_path = os.path.join(feat_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        if not os.path.exists(feat_subdir_path):
            os.mkdir(feat_subdir_path)
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            spk_feat_dir_path = os.path.join(feat_subdir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            if not os.path.exists(spk_feat_dir_path):
                os.mkdir(spk_feat_dir_path)
            else:
                continue
            print("speaker: {} processing".format(spk_dir))
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                uttr_feat_dir_path = os.path.join(spk_feat_dir_path, uttr_dir)
                if not os.path.isdir(uttr_dir_path):
                    continue
                if not os.path.exists(uttr_feat_dir_path):
                    os.mkdir(uttr_feat_dir_path)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    feat_path = os.path.join(uttr_feat_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    extractor.save(show=show_name,
                                   channel=0,
                                   input_audio_filename=audio_path,
                                   output_feature_filename=feat_path)


def convert_wav_to__plp_16_hd5(corpus_dir, feat_dir):
    extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=16000,
                                          filter_bank="log",
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=13,
                                          vad="snr",
                                          snr=40,
                                          pre_emphasis=0.97,
                                          feature_type='plp',
                                          rasta_plp=True,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)

    # target_spk_list = ['1272', '1462', '737', '742', '75', '753', '766', '77', '774', '778', '779', '780', '782', '789', '791', '792', '797', '807', '810', '811', '82', '826', '844', '845', '846', '85', '851', '876', '884', '886', '895', '91', '915', '92', '921', '923', '927', '937', '94', '951', '956', '960', '964', '969', '976', '978', '982', '985']
    first_start = False
    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        feat_subdir_path = os.path.join(feat_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        if not os.path.exists(feat_subdir_path):
            os.mkdir(feat_subdir_path)
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            spk_feat_dir_path = os.path.join(feat_subdir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            if not os.path.exists(spk_feat_dir_path):
                os.mkdir(spk_feat_dir_path)
            else:
                continue

            #if not spk_dir in target_spk_list:
            #    continue

            print("speaker: {} processing".format(spk_dir))
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                uttr_feat_dir_path = os.path.join(spk_feat_dir_path, uttr_dir)
                if not os.path.isdir(uttr_dir_path):
                    continue
                if not os.path.exists(uttr_feat_dir_path):
                    os.mkdir(uttr_feat_dir_path)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    feat_path = os.path.join(uttr_feat_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    extractor.save(show=show_name,
                                   channel=0,
                                   input_audio_filename=audio_path,
                                   output_feature_filename=feat_path)

if __name__ == '__main__':
    corpus_dir = "D:/Corpus/LibriSpeech"
    feat_dir = "H:/HJB/Corpus/LibiSpeechMFCClibrosa8"
    convert_wavto_hd5_from_libri_librosa(corpus_dir, feat_dir, samplerate=8000)
    # convert_wav16_to_hd5_from_libri(corpus_dir, feat_dir)
    # feat_dir = "H:/HJB/Corpus/LibiSpeechPLP16"
    # convert_wav_to__plp_16_hd5(corpus_dir, feat_dir)
    # convert_wav_to_hd5(corpus_dir, feat_dir)
    # plp_dir = "D:/Corpus/LibiSpeechPLP"
    # convert_wav_to__plp_hd5(corpus_dir, plp_dir)
