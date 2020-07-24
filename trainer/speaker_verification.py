import os
import sidekit
import pickle
import numpy
import sys


def id_map_list(ubm_feat_dir):
    rightlist = []
    leftlist = []

    for h5_file in os.listdir(ubm_feat_dir):
        if h5_file.find('.h5') == -1:
            continue
        show_name = h5_file.split('.h5')[0]
        rightlist.append(show_name)
        leftlist.append(show_name.split('-')[0])
    return leftlist, rightlist

def get_id_map(leftlist, rightlist):
    idmap = sidekit.IdMap()
    idmap.leftids = numpy.array(leftlist)
    idmap.rightids = numpy.array(rightlist)
    idmap.start = numpy.empty((len(leftlist)), dtype="|O")
    idmap.stop = numpy.empty((len(leftlist)), dtype="|O")
    return idmap

def get_ndx_map(leftlist, rightlist):
    ndx = sidekit.Ndx()
    ndx.modelset = numpy.array(list(set(leftlist)))
    ndx.segset = numpy.array(rightlist)
    ndx.trialmask = numpy.ones((len(ndx.modelset), len(ndx.segset)), dtype='bool')
    ndx.validate()
    return ndx

def get_key_map(ndx, leftlist, rightlist):
    key = sidekit.Key()
    key.modelset = ndx.modelset
    key.segset = ndx.segset
    model_num = len(key.modelset)
    seg_num = len(key.segset)
    model_set = list(set(leftlist))
    seg_list = rightlist
    key.tar = numpy.zeros((model_num, seg_num), dtype='bool')
    for m_idx, m_name in enumerate(model_set):
        for s_idx, s_name in enumerate(seg_list):
            fm_name = s_name.split('-')[0]
            if fm_name == m_name:
                key.tar[m_idx, s_idx] = True

    key.non = numpy.ones((model_num, seg_num), dtype='bool')
    for m_idx, m_name in enumerate(model_set):
        for s_idx, s_name in enumerate(seg_list):
            fm_name = s_name.split('-')[0]
            if fm_name == m_name:
                key.non[m_idx, s_idx] = False
    key.validate()
    return key


class EnrollSpeakerInfo(object):
    def __init__(self, spk_info_file):
        self.speaker_info = {}
        if os.path.exists(spk_info_file):
            self.load_speaker_info(spk_info_file)


    def add_speaker(self, speaker_name, feat_file_path):
        if speaker_name not in self.speaker_info:
            self.speaker_info[speaker_name] = []
        self.speaker_info[speaker_name].append(feat_file_path)


    def remove_speaker(self, speaker_name):
        if speaker_name in self.speaker_info:
            for feat_file in self.speaker_info[speaker_name]:
                try:
                    os.remove(feat_file)
                except Exception as ex:
                    print("failed removing file")
            del self.speaker_info[speaker_name]


    def uuid_from_name(self, speaker_name):
        spk_uuid = abs(hash(speaker_name)) % (10**8)
        return spk_uuid


    def get_show_name(self, speaker_name):
        spk_uuid = self.uuid_from_name(speaker_name)
        show_name  = "{}-0".format(spk_uuid)
        if speaker_name in self.speaker_info:
            show_name = "{}-{}".format(spk_uuid, len(self.speaker_info[speaker_name]))
        return show_name


    def get_feat_path(self, enroll_feat_dir, show_name):
        return os.path.join(enroll_feat_dir, "{}.h5".format(show_name))

    def get_show_name_from_path(self, feat_path):
        show_name, ext = os.path.splitext(os.path.basename(feat_path))
        return show_name

    def save_speaker_info(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump(self.speaker_info, f)


    def load_speaker_info(self, model_path):
        with open(model_path, "rb") as f:
            self.speaker_info = pickle.load(f)


    def find_speaker_name_from_show_name(self, show_name):
        spk_name = ""
        for speaker in self.speaker_info:
            for file_path in self.speaker_info[speaker]:
                f_show_name = self.get_show_name_from_path(file_path)
                if f_show_name == show_name:
                    spk_name = speaker
                    return spk_name
        return spk_name

class SpeakerVerification(object):
    def __init__(self):
        self.spk_info_file = "model/speaker_info.dat"
        self.ubm_file = "model/speaker_model"
        self.tv_file = "model/channel_model"
        self.spk_enroll_file = 'model/spk_enroll.dat'
        self.feature_dir = "feature"
        if not os.path.exists(self.feature_dir):
            os.mkdir(self.feature_dir)
        self.speaker_info = EnrollSpeakerInfo(self.spk_info_file)


    def enroll_feat(self, show_name, audio_path, enroll_feat_dir):
        extractor = sidekit.FeaturesExtractor(audio_filename_structure=None,
                                              feature_filename_structure=None,
                                              sampling_frequency=None,
                                              lower_frequency=133.3,  # 200
                                              higher_frequency=4000,  # 3800
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

        # print(audio_file)
        feat_path = self.speaker_info.get_feat_path(enroll_feat_dir, show_name)
        if os.path.exists(feat_path):
            os.remove(feat_path)
        extractor.save(show=show_name,
                       channel=0,
                       input_audio_filename=audio_path,
                       output_feature_filename=feat_path)
        return feat_path


    def enroll_speaker(self, speaker_name, audio_file):
        spk_show_name = self.speaker_info.get_show_name(speaker_name)
        enroll_feat_path = self.speaker_info.get_feat_path(self.feature_dir, spk_show_name)
        if os.path.exists(audio_file):
            self.enroll_feat(spk_show_name, audio_file, self.feature_dir)
            self.speaker_info.add_speaker(speaker_name, enroll_feat_path)
            self.speaker_info.save_speaker_info(self.spk_info_file)


    def get_total_enroll_stat(self, enroll_feat_dir):
        nbThread = 1
        leftlist, rightlist = id_map_list(enroll_feat_dir)
        id_map = get_id_map(leftlist, rightlist)
        ubm = sidekit.Mixture(mixture_file_name=self.ubm_file)
        feature_filename = os.path.join(os.path.abspath(enroll_feat_dir),"{}.h5")
        features_server = sidekit.FeaturesServer(features_extractor=None,
                                                 feature_filename_structure=feature_filename,
                                                 sources=None,
                                                 dataset_list=["energy", "cep", "vad"],
                                                 mask="[1-13]",
                                                 feat_norm="cmvn",
                                                 global_cmvn=None,
                                                 dct_pca=False,
                                                 dct_pca_config=None,
                                                 sdc=False,
                                                 sdc_config=None,
                                                 delta=True,
                                                 double_delta=True,
                                                 delta_filter=None,
                                                 context=None,
                                                 traps_dct_nb=None,
                                                 rasta=True,
                                                 keep_all_features=False)

        enroll_stat = sidekit.StatServer(id_map, ubm=ubm)
        enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server,
                                    seg_indices=range(enroll_stat.segset.shape[0]), num_thread=nbThread)
        # enroll_stat.write(self.spk_stat_file)
        return enroll_stat


    def get_verify_spk_stat(self, audio_file):
        nbThread = 1
        show_name = "test_speaker"
        feat_path = self.enroll_feat(show_name, audio_file, self.feature_dir)
        feat_dir = os.path.dirname(os.path.abspath(feat_path))
        leftlist = [show_name]
        rightlist = [show_name]
        id_map = get_id_map(leftlist, rightlist)
        test_ndx = get_ndx_map(leftlist, rightlist)
        ubm = sidekit.Mixture(mixture_file_name=self.ubm_file)
        feature_filename = os.path.join(feat_dir, "{}.h5")
        features_server = sidekit.FeaturesServer(features_extractor=None,
                                                 feature_filename_structure=feature_filename,
                                                 sources=None,
                                                 dataset_list=["energy", "cep", "vad"],
                                                 mask="[1-13]",
                                                 feat_norm="cmvn",
                                                 global_cmvn=None,
                                                 dct_pca=False,
                                                 dct_pca_config=None,
                                                 sdc=False,
                                                 sdc_config=None,
                                                 delta=True,
                                                 double_delta=True,
                                                 delta_filter=None,
                                                 context=None,
                                                 traps_dct_nb=None,
                                                 rasta=True,
                                                 keep_all_features=False)

        enroll_stat = sidekit.StatServer(id_map, ubm=ubm)
        enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server,
                                    seg_indices=range(enroll_stat.segset.shape[0]), num_thread=nbThread)
        # enroll_stat.write(self.spk_stat_file)
        return enroll_stat, test_ndx


    def extract_i_vector(self, stats_speaker):
        ubm = sidekit.Mixture(mixture_file_name=self.ubm_file)
        tv_file = self.tv_file
        iv, iv_uncertaniny = self.i_vector_with_stat(stats_speaker, ubm, tv_file)
        return iv, iv_uncertaniny


    def i_vector_with_stat(self, stat, ubm_model, tv_file):
        tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(tv_file)
        fa  = sidekit.FactorAnalyser(mean=tv_mean, Sigma=tv_sigma, F=tv)
        iv, iv_uncertainty = fa.extract_ivectors_single(ubm=ubm_model, stat_server=stat, uncertainty=True)
        return iv, iv_uncertainty


    def adaption_all_enrolled_speakers(self):
        spk_stat = self.get_total_enroll_stat(self.feature_dir)
        spk_i_vect, _ = self.extract_i_vector(spk_stat)
        if os.path.exists(self.spk_enroll_file):
            os.remove(self.spk_enroll_file)
        spk_i_vect.write(self.spk_enroll_file)


    def verify(self, audio_file):
        spk_stat, test_ndx = self.get_verify_spk_stat(audio_file)
        spk_i_vect, _ = self.extract_i_vector(spk_stat)
        show_name = self.verified_speaker(spk_i_vect, test_ndx)
        if show_name == "" or show_name == "None":
            print("This audio file is rejected")
        else:
            speaker_name = self.speaker_info.find_speaker_name_from_show_name(show_name)
            print("{} --> accepted as speaker: {}".format(os.path.basename(audio_file), speaker_name))


    def verified_speaker(self, spk_iv, test_ndx):
        enroll_iv = sidekit.StatServer(self.spk_enroll_file)
        enroll_iv.norm_stat1()
        spk_iv.norm_stat1()
        scores_cos = numpy.dot(enroll_iv.stat1, spk_iv.stat1.transpose())
        numpy.reshape(scores_cos, len(scores_cos))
        max_idx = numpy.argmax(scores_cos)
        max_val = scores_cos[max_idx]
        show_name = "None"
        if max_val > 0.33:
            show_name = enroll_iv.segset[max_idx]
        return show_name


if __name__ == "__main__":
    type = "None"
    audio_path = ""
    speaker_name = "None"
    if len(sys.argv) < 2:
        print("please input command like this:")
        print("python speaker_verification.py enroll speaker_name audio_path")
        print("python speaker_verification.py adapt")
        print("python speaker_verification.py verify audio_path")
        sys.exit(1)
    elif len(sys.argv) == 2:
        type = sys.argv[1]
        if type != "adapt":
            print("wrong input type")
            sys.exit(1)
    elif len(sys.argv) == 3:
        type = sys.argv[1]
        if type != "verify":
            print("wrong input type")
            sys.exit(1)
    else:
        type = sys.argv[1]
        if type != "enroll":
            print("wrong input type")
            sys.exit(1)

    sp_verifier = SpeakerVerification()
    if type == "enroll":
        speaker_name = sys.argv[2]
        audio_path = sys.argv[3]
        sp_verifier.enroll_speaker(speaker_name, audio_path)
    elif type == "adapt":
        sp_verifier.adaption_all_enrolled_speakers()
    elif type == "verify":
        audio_path = sys.argv[2]
        sp_verifier.verify(audio_path)
    else:
        print("wrong input type")
        sys.exit(1)

