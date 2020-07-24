import os
import sidekit

def ubm_file_list(ubm_feat_dir):
    file_list = []
    for h5_file in os.listdir(ubm_feat_dir):
        if h5_file.find('.h5') == -1:
            continue

        file_list.append(h5_file.split('.h5')[0])
    return file_list

def train_ubm_split_mfcc():
    ubm_feature_dir = "D:/Corpus/LibiSpeechUBMMFCC"
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="D:/Corpus/LibiSpeechUBMMFCC/{}.h5",
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask=None, # "[1-13]"
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
    ubm_list = ubm_file_list(ubm_feature_dir)
    ubm = sidekit.Mixture()
    distrib_nb = 1024
    nbThread = 1
    llk = ubm.EM_split(features_server=features_server,
                 feature_list=ubm_list,
                 distrib_nb=distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                 num_thread=nbThread,
                 save_partial='gmm/ubm',
                 output_file_name='libri_ubm',
                 ceil_cov=10,
                 floor_cov=1e-2)
    # llk = ubm.EM_split(features_server, ubm_list, distrib_nb, num_thread=nbThread, save_partial='gmm/ubm')
    ubm.write('gmm/ubm_{}.h5'.format(distrib_nb))
    print("training llk-len, llk: {}".format(len(llk), llk))


def train_ubm_split_mfcc_16():
    ubm_feature_dir = "H:/HJB/Corpus/LibiSpeechUBMMFCC16"
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="H:/HJB/Corpus/LibiSpeechUBMMFCC16/{}.h5",
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-12]", # "[1-13]"
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
    ubm_list = ubm_file_list(ubm_feature_dir)
    ubm = sidekit.Mixture()
    distrib_nb = 1024
    nbThread = 1
    llk = ubm.EM_split(features_server=features_server,
                 feature_list=ubm_list,
                 distrib_nb=distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                 num_thread=nbThread,
                 save_partial='gmm/ubm_16_12',
                 output_file_name='libri_ubm_16_12',
                 ceil_cov=10,
                 floor_cov=1e-2)
    # llk = ubm.EM_split(features_server, ubm_list, distrib_nb, num_thread=nbThread, save_partial='gmm/ubm')
    ubm.write('gmm/ubm_16_12_{}.h5'.format(distrib_nb))
    print("training llk-len, llk: {}".format(len(llk), llk))


def train_ubm_split_plp():
    ubm_feature_dir = "D:/Corpus/LibriSpeechUBMPLPSMall"
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="D:/Corpus/LibriSpeechUBMPLPSMall/{}.h5",
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-13]", # "[1-13]"
                                             feat_norm="cmvn",
                                             global_cmvn=None, #None,True
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
    ubm_list = ubm_file_list(ubm_feature_dir)
    ubm = sidekit.Mixture()
    distrib_nb = 1024
    nbThread = 1
    print("ubm train starting")
    llk = ubm.EM_split(features_server=features_server,
                 feature_list=ubm_list,
                 distrib_nb=distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                 num_thread=nbThread,
                 save_partial='gmm/ubm_plp',
                 output_file_name='libri_ubm_plp',
                 ceil_cov=10,
                 floor_cov=1e-2)
    # llk = ubm.EM_split(features_server, ubm_list, distrib_nb, num_thread=nbThread, save_partial='gmm/ubm')
    ubm.write('gmm/ubm_plp_{}.h5'.format(distrib_nb))
    print("training llk-len, llk: {}".format(len(llk), llk))

def train_ubm_split_plp_16():
    ubm_feature_dir = "H:/HJB/Corpus/LibiSpeechUBMPLP16"
    feature_file_pattern = os.path.join(ubm_feature_dir, "{}.h5")
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure=feature_file_pattern,
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-13]", # "[1-13]"
                                             feat_norm="cmvn",
                                             global_cmvn=None, #None,True
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
    ubm_list = ubm_file_list(ubm_feature_dir)
    ubm = sidekit.Mixture()
    distrib_nb = 1024
    nbThread = 1
    print("ubm train starting")
    llk = ubm.EM_split(features_server=features_server,
                 feature_list=ubm_list,
                 distrib_nb=distrib_nb,
                 iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                 num_thread=nbThread,
                 save_partial='gmm/ubm_plp_16',
                 output_file_name='libri_ubm_plp_16',
                 ceil_cov=10,
                 floor_cov=1e-2)
    # llk = ubm.EM_split(features_server, ubm_list, distrib_nb, num_thread=nbThread, save_partial='gmm/ubm')
    ubm.write('gmm/ubm_plp_16_{}.h5'.format(distrib_nb))
    print("training llk-len, llk: {}".format(len(llk), llk))

if __name__ == "__main__":
    train_ubm_split_plp_16()
    # train_ubm_split_mfcc_16()