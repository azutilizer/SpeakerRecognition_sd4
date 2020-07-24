import os
import numpy
import sidekit

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


def adatpt_stats_with_acoustic_feat_dir(feature_dir, ubm_model):
    nbThread = 1
    #distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure=os.path.join(feature_dir, "{}.h5"),
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-13]",  # None
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
    adatp_stat = sidekit.StatServer(id_map, ubm=ubm_model)
    adatp_stat.accumulate_stat(ubm=ubm_model, feature_server=features_server,
                                seg_indices=range(adatp_stat.segset.shape[0]), num_thread=nbThread)
    return adatp_stat


def i_vector_with_stat(stat, ubm_model, tv_file):
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(tv_file)
    fa  = sidekit.FactorAnalyser(mean=tv_mean, Sigma=tv_sigma, F=tv)
    iv, iv_uncertainty = fa.extract_ivectors_single(ubm=ubm_model, stat_server=stat, uncertainty=True)
    return iv, iv_uncertainty


def i_vector_with_feat_dir(feat_dir, ubm_file, tv_file):
    ubm = sidekit.Mixture(mixture_file_name=ubm_file) #'gmm/ubm_plp_1024.h5'
    feat_stat = adatpt_stats_with_acoustic_feat_dir(feat_dir, ubm)
    return i_vector_with_stat(feat_stat, ubm, tv_file)


if __name__ == "__main__":
    feat_dir = "D:\Corpus\myTest"
    ubm_file = "gmm/ubm_plp_1024.h5"
    tv_file = "data/TV_plp_small_1024"
    iv, iv_uncertainy = i_vector_with_feat_dir(feat_dir, ubm_file, tv_file)
    aa = 0