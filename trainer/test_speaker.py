import os
import mfcc_ubm_trainer
import train_i_vector
import shutil

def extract_test_feat(ubm_feat_dir, corpus_dir, test_feat_dir):
    ubm_feat_list = mfcc_ubm_trainer.ubm_file_list(ubm_feat_dir)

    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            print("speaker: {} processing".format(spk_dir))
            uttr_num =0
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    test_path = os.path.join(test_feat_dir, "{}.h5".format(show_name))
                    if show_name in ubm_feat_list:
                        continue
                    uttr_num += 1
                    if uttr_num <= 3:
                        shutil.copyfile(audio_path, test_path)
                    else:
                        break


def extract_enroll_feat(ubm_feat_dir, corpus_dir, test_feat_dir):
    ubm_feat_list = mfcc_ubm_trainer.ubm_file_list(ubm_feat_dir)

    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            print("speaker: {} processing".format(spk_dir))
            max_file_size = 0
            max_file_path = ""
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    file_size = os.path.getsize(audio_path)
                    if file_size > max_file_size:
                        max_file_size = file_size
                        max_file_path = audio_path
            if os.path.exists(max_file_path):
                show_name = os.path.basename(max_file_path).split('.flac')[0]
                test_path = os.path.join(test_feat_dir, "{}.h5".format(show_name))
                shutil.copyfile(max_file_path, test_path)


def extract_ubm_plp_feat(ubm_feat_dir, corpus_dir, ubm_plp_feat_dir):
    ubm_feat_list = mfcc_ubm_trainer.ubm_file_list(ubm_feat_dir)

    for train_dir in os.listdir(corpus_dir):
        train_dir_path = os.path.join(corpus_dir, train_dir)
        if not os.path.isdir(train_dir_path):
            continue
        for spk_dir in os.listdir(train_dir_path):
            spk_dir_path = os.path.join(train_dir_path, spk_dir)
            if not os.path.isdir(spk_dir_path):
                continue
            print("speaker: {} processing".format(spk_dir))
            uttr_num =0
            for uttr_dir in os.listdir(spk_dir_path):
                uttr_dir_path = os.path.join(spk_dir_path, uttr_dir)
                for audio_file in os.listdir(uttr_dir_path):
                    if '.flac' not in audio_file:
                        continue
                    # print(audio_file)
                    audio_path = os.path.join(uttr_dir_path, audio_file)
                    show_name = audio_file.split('.flac')[0]
                    test_path = os.path.join(ubm_plp_feat_dir, "{}.h5".format(show_name))
                    if show_name not in ubm_feat_list:
                        continue
                    shutil.copyfile(audio_path, test_path)


def factor_analysis_tv(feat_dir):
    train_i_vector.test_tv_matrix(feat_dir)


def factor_analysis_plp_tv(feat_dir):
    train_i_vector.test_tv_plp_matrix(feat_dir)


def test_ivector_score(feat_dir):
    train_i_vector.test_ivector_score(feat_dir)


def test_plp_ivector_score(feat_dir):
    train_i_vector.test_plp_ivector_score(feat_dir)


def test_ivector_extract():
    train_i_vector.test_i_vector_extract()


def test_plp_ivector_extract():
    train_i_vector.test_plp_i_vector_extract()


if __name__ == "__main__":
    ubm_feature_dir = "D:/Corpus/LibiSpeechUBMMFCC"
    corpus_feature_dir = "D:/Corpus/LibiSpeechMFCC"
    enroll_feat_dir = "D:/Corpus/LibiSpeechEnrollMFCC"
    test_feat_dir = "D:/Corpus/LibiSpeechTestMFCC"
    ubm_plp_feat_dir = "D:/Corpus/LibiSpeechUBMPLP"
    corpus_plp_feature_dir = "D:/Corpus/LibiSpeechPLP"
    test_plp_feat_dir = "D:/Corpus/LibiSpeechSmallTestPLP"

    # extract_enroll_feat(ubm_feature_dir, corpus_feature_dir, enroll_feat_dir)
    # extract_test_feat(ubm_feature_dir, corpus_feature_dir, test_feat_dir)
    # factor_analysis_tv(enroll_feat_dir)
    # test_ivector_extract()
    # test_ivector_score(test_feat_dir)
    # extract_ubm_plp_feat(ubm_feature_dir, corpus_plp_feature_dir, ubm_plp_feat_dir)
    # extract_ubm_plp_feat(test_feat_dir, corpus_plp_feature_dir, test_plp_feat_dir)
    # factor_analysis_plp_tv(test_plp_feat_dir)
    # test_plp_ivector_extract()
    test_plp_ivector_score(test_plp_feat_dir)
