import os
import sidekit

sub_data = ["enroll", "test"]


def safe_makedir(dir_name):
    """This function takes a directory name as an argument"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("Directory <{}> created.".format(dir_name))


def convert_wav2h5(base_dir, out_dir, samplerate=8000):
    low_freq = 100
    higher_freq = 4000  # 3800

    extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                          feature_filename_structure=None,
                                          sampling_frequency=samplerate,
                                          lower_frequency=low_freq,  # 200
                                          higher_frequency=higher_freq,  # 3800
                                          filter_bank="log",
                                          filter_bank_size=40,
                                          window_size=0.025,
                                          shift=0.01,
                                          ceps_number=20,
                                          vad="snr",  # "snr"
                                          snr=40,
                                          pre_emphasis=0.97,
                                          save_param=["vad", "energy", "cep", "fb"],
                                          keep_all_features=True)

    for sub_data_name in sub_data:
        sub_data_path = os.path.join(base_dir, sub_data_name)
        out_path = os.path.join(out_dir, sub_data_name)

        safe_makedir(out_path)

        if not os.path.exists(sub_data_path):
            continue

        for audio_file in os.listdir(sub_data_path):
            if not audio_file.endswith('.wav'):
                continue
            audio_path = os.path.join(sub_data_path, audio_file)
            feature_path = os.path.join(out_path, audio_file[:-4]+'.h5')
            show_name = audio_file.split('.wav')[0]
            extractor.save(show=show_name,
                           channel=0,
                           input_audio_filename=audio_path,
                           output_feature_filename=feature_path)
            print("{} : {}  saved.".format(show_name, audio_file))


def main():
    data_dir = os.path.join(".", "results", "audio")
    feature_dir = os.path.join(".", "results", "feature")
    convert_wav2h5(data_dir, feature_dir)


if __name__ == '__main__':
    main()
    print("Successfully Finished.")
