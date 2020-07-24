import os
from numpy.random import shuffle
# from feat_util import mfcc_extaction_with_vad

def read_lines(txt_file):
    with open(txt_file) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    return [x.strip() for x in content]

def librispeech_list(corpus_dir):
    speaker_id = "SPEAKERS.TXT"
    speaker_id_path = os.path.join(corpus_dir, speaker_id)
    txt_data = read_lines(speaker_id_path)
    speaker_info = {}
    for line in txt_data:
        if line[0] == ";":
            continue
        line_info = line.split("|")
        speaker_id = line_info[0].strip()
        speaker_sex = line_info[1].strip()
        speaker_info[speaker_id] = speaker_sex

    man_count = 0
    female_count = 0

    for speaker in speaker_info:
        sex = speaker_info[speaker].lower()
        if sex == "m":
            man_count += 1
        elif sex == "f":
            female_count += 1

    print("males: {}, females: {}".format(man_count, female_count))
    return speaker_info

def make_speaker_list(corpus_dir):
    real_male_num = 0
    real_female_num = 0
    speaker_info = librispeech_list(corpus_dir)
    male_list = []
    female_list = []

    for sub_dir in os.listdir(corpus_dir): # train_clean, dev_clean
        sub_dir_path = os.path.join(corpus_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        print("proccessing speaker: {}".format(sub_dir))
        for speaker_id in os.listdir(sub_dir_path):
            sex = speaker_info[speaker_id].lower()
            if sex == "m":
                real_male_num += 1
                male_list.append(speaker_id)
            elif sex == "f":
                real_female_num += 1
                female_list.append(speaker_id)

    shuffle(male_list)
    male_list = male_list[:len(female_list)]
    print("real males: {}, real females: {}".format(len(male_list), len(female_list)))
    return male_list, female_list

def make_speaker_feature(corpus_dir, out_dir):
    male_list, female_list = make_speaker_list(corpus_dir)
    total_num = 0
    for sub_dir in os.listdir(corpus_dir): # train_clean, dev_clean
        sub_dir_path = os.path.join(corpus_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for speaker_id in os.listdir(sub_dir_path):
            if speaker_id in male_list or speaker_id in female_list:
                total_num += 1
            else:
                continue

            print("proccessing speaker: {}".format(speaker_id))
            txt_path = os.path.join(out_dir, "m_lib_{}.feat".format(speaker_id))
            if speaker_id in female_list:
                txt_path = os.path.join(out_dir, "f_lib_{}.feat".format(speaker_id))
            speaker_id_path = os.path.join(sub_dir_path, speaker_id)
            speaker_file_list = []
            for sp_sub_dir in os.listdir(speaker_id_path):
                sp_sub_dir_path = os.path.join(speaker_id_path, sp_sub_dir)
                for wav_file in os.listdir(sp_sub_dir_path):
                    if wav_file.find(".flac") == -1 and wav_file.find(".FLAC") == -1:
                        continue
                    wav_path = os.path.join(sp_sub_dir_path, wav_file)
                    speaker_file_list.append(wav_path)

            print("wave file num: {}".format(len(speaker_file_list)))
            shuffle(speaker_file_list)
            speaker_file_list = speaker_file_list[:15]
            speech_feat = []
            for flac_file in speaker_file_list:
                feat = mfcc_extaction_with_vad(flac_file)
                speech_feat.extend(feat[0])
            shuffle(speech_feat)
            sp_file = open(txt_path, "w")
            sp_file.write("{},{}\n".format(len(speech_feat), len(speech_feat[0])))
            for feat in speech_feat:
                line = ",".join("{:.4f}".format(value) for value in feat)
                line += "\n"
                sp_file.write(line)
            sp_file.close()

    print("total speakers: {}".format(total_num))

def make_speaker_id_folder(corpus_dir, speakers_dir):
    import shutil
    male_list, female_list = make_speaker_list(corpus_dir)
    # corpus_info_file = open(os.path.join(speaker_dir, "librispeech_all.csv"), "w")
    total_num = 0
    for sub_dir in os.listdir(corpus_dir): # train_clean, dev_clean
        sub_dir_path = os.path.join(corpus_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for speaker_id in os.listdir(sub_dir_path):
            if speaker_id in male_list or speaker_id in female_list:
                total_num += 1
            else:
                continue
            print("proccessing speaker: {}".format(speaker_id))
            dst_sp_dir = os.path.join(speakers_dir, speaker_id)

            # create speaker dir
            if not os.path.exists(dst_sp_dir):
                os.mkdir(dst_sp_dir)

            speaker_id_path = os.path.join(sub_dir_path, speaker_id)
            speaker_file_list = []
            for sp_sub_dir in os.listdir(speaker_id_path):
                sp_sub_dir_path = os.path.join(speaker_id_path, sp_sub_dir)
                for wav_file in os.listdir(sp_sub_dir_path):
                    if wav_file.find(".flac") == -1 and wav_file.find(".FLAC") == -1:
                        continue
                    wav_path = os.path.join(sp_sub_dir_path, wav_file)
                    speaker_file_list.append(wav_path)

            shuffle(speaker_file_list)
            speaker_file_list = speaker_file_list[:10]

            print("wave file num: {}".format(len(speaker_file_list)))
            for flac_file in speaker_file_list:
                shutil.copy(flac_file, os.path.join(dst_sp_dir, os.path.basename(flac_file)))

    print("total speakers: {}".format(total_num))

def check_corpus_trainingdata(corpus_dir):
    # corpus_info_file = open(os.path.join(speaker_dir, "librispeech_all.csv"), "w")
    invalid_speaker_list = []
    for sub_dir in os.listdir(corpus_dir): # train_clean, dev_clean
        sub_dir_path = os.path.join(corpus_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for speaker_id in os.listdir(sub_dir_path):
            speaker_id_path = os.path.join(sub_dir_path, speaker_id)
            speaker_file_list = []
            print("proccessing speaker: {}".format(speaker_id))
            for sp_sub_dir in os.listdir(speaker_id_path):
                sp_sub_dir_path = os.path.join(speaker_id_path, sp_sub_dir)
                for wav_file in os.listdir(sp_sub_dir_path):
                    if wav_file.find(".flac") == -1 and wav_file.find(".FLAC") == -1:
                        continue
                    wav_path = os.path.join(sp_sub_dir_path, wav_file)
                    speaker_file_list.append(wav_path)
                    break
            if len(speaker_file_list) == 0:
                invalid_speaker_list.append("{}".format(speaker_id))
    print(invalid_speaker_list)


def make_sidekit_ubm_trainingdata(corpus_dir, featrue_dir):
    import shutil
    male_list, female_list = make_speaker_list(corpus_dir)
    # corpus_info_file = open(os.path.join(speaker_dir, "librispeech_all.csv"), "w")
    total_num = 0
    for sub_dir in os.listdir(corpus_dir): # train_clean, dev_clean
        sub_dir_path = os.path.join(corpus_dir, sub_dir)
        if not os.path.isdir(sub_dir_path):
            continue
        for speaker_id in os.listdir(sub_dir_path):
            if speaker_id in male_list or speaker_id in female_list:
                total_num += 1
            else:
                continue
            print("proccessing speaker: {}".format(speaker_id))
            # dst_sp_dir = os.path.join(featrue_dir, speaker_id)

            # create speaker dir
            #if not os.path.exists(dst_sp_dir):
            #    os.mkdir(dst_sp_dir)

            speaker_id_path = os.path.join(sub_dir_path, speaker_id)
            speaker_file_list = []
            for sp_sub_dir in os.listdir(speaker_id_path):
                sp_sub_dir_path = os.path.join(speaker_id_path, sp_sub_dir)
                for wav_file in os.listdir(sp_sub_dir_path):
                    if wav_file.find(".flac") == -1 and wav_file.find(".FLAC") == -1:
                        continue
                    wav_path = os.path.join(sp_sub_dir_path, wav_file)
                    speaker_file_list.append(wav_path)

            shuffle(speaker_file_list)
            speaker_file_list = speaker_file_list[:24]

            print("wave file num: {}".format(len(speaker_file_list)))
            for flac_file in speaker_file_list:
                shutil.copy(flac_file, os.path.join(featrue_dir, os.path.basename(flac_file).replace('.flac', '.h5')))

    print("total speakers: {}".format(total_num))

def make_corpus_info_text(corpus_dir):
    txt_file = open(os.path.join(corpus_dir, "libirispeaker_info.csv"), "w")
    print(os.path.basename(corpus_dir))
    txt_file.write("file, material, speaker\n")
    for sp_dir in os.listdir(corpus_dir): # speaker directory
        sp_dir_path = os.path.join(corpus_dir, sp_dir)
        if not os.path.isdir(sp_dir_path):
            continue
        print("process: {}".format(sp_dir))
        for wav_file in os.listdir(sp_dir_path):
            train_file_path = os.path.join(os.path.basename(corpus_dir), sp_dir, wav_file)
            material = "train"
            speaker_id = sp_dir
            txt_file.write("{}, {}, {}\n".format(train_file_path, material, speaker_id))
    txt_file.close()

def make_ubm_list(ubm_feature_dir):
    ubm_list_file = open('ubm_list.txt', 'w')
    for h5_file in os.listdir(ubm_feature_dir):
        if h5_file.find('.h5') == -1:
            continue
        ubm_list_file.write(os.path.join(ubm_feature_dir, h5_file)+"\n")
    ubm_list_file.close()

corpus_dir = "./"
out_dir = "/mnt/hgfs/Librispeech/TXT"
speaker_dir = "/mnt/hgfs/Librispeech/LibriSpeakers"
total_feature_dir = "H:/HJB/Corpus/LibiSpeechMFCC16"
ubm_feature_dir = "H:/HJB/Corpus/LibiSpeechUBMMFCC16"
tv_feature_dir = "H:/HJB/Corpus/LibiSpeechTVMFCC16"
total_feature_dir = "H:/HJB/Corpus/LibiSpeechPLP16"
ubm_feature_dir = "H:/HJB/Corpus/LibiSpeechUBMPLP16"
tv_feature_dir = "H:/HJB/Corpus/LibiSpeechTVPLP16"
spk_info = librispeech_list(corpus_dir)
aa = 0
# make_speaker_list(corpus_dir)
# make_speaker_feature(corpus_dir, out_dir)
# make_speaker_id_folder(corpus_dir, speaker_dir)
# make_corpus_info_text(speaker_dir)
# make_sidekit_ubm_trainingdata(total_feature_dir, tv_feature_dir)
# check_corpus_trainingdata(total_feature_dir)
# make_ubm_list(ubm_feature_dir)
