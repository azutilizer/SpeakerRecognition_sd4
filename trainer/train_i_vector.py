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

def train_tv_matrix(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="D:/Corpus/LibiSpeechUBMMFCC/{}.h5",
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/stat_sre10_core-core_ubm_{}.h5'.format(distrib_nb))


def train_tv_matrix_16K(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    feat_file_name = os.path.join(feature_dir, "{}.h5")
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_16_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure=feat_file_name,
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-19]",
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/stat_sre10_core-core_16_tv_{}.h5'.format(distrib_nb))


def train_plp_tv_matrix(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_plp_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure=os.path.join(feature_dir, "{}.h5"),
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-13]", #None
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/plp_small_stat_sre10_core-core_enroll_{}.h5'.format(distrib_nb))

def train_plp_16_tv_matrix(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_plp_16_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure=os.path.join(feature_dir, "{}.h5"),
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-13]", #None
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/plp_16k_small_stat_sre10_core-core_enroll_{}.h5'.format(distrib_nb))


def enroll_tv_matrix(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="D:/Corpus/LibiSpeechEnrollMFCC/{}.h5",
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/stat_sre10_core-core_enroll_{}.h5'.format(distrib_nb))


def test_tv_matrix(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="D:/Corpus/LibiSpeechTestMFCC/{}.h5",
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/stat_sre10_core-core_test_{}.h5'.format(distrib_nb))


def test_tv_plp_matrix(feature_dir):
    nbThread = 1
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feature_dir)
    id_map = get_id_map(leftlist, rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_plp_1024.h5')
    features_server = sidekit.FeaturesServer(features_extractor=None,
                                             feature_filename_structure="D:/Corpus/LibiSpeechSmallTestPLP/{}.h5",
                                             sources=None,
                                             dataset_list=["energy", "cep", "vad"],
                                             mask="[1-13]", # "[1-13]",None
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
    enroll_stat.accumulate_stat(ubm=ubm, feature_server=features_server, seg_indices=range(enroll_stat.segset.shape[0]) ,num_thread=nbThread)
    enroll_stat.write('data/plp_small_stat_sre10_core-core_test_{}.h5'.format(distrib_nb))


def test_i_vector_extract():
    distrib_nb = 1024
    nbThread = 1
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5("data/TV_{}".format(distrib_nb))
    enroll_stat = sidekit.StatServer('data/stat_sre10_core-core_test_{}.h5'.format(distrib_nb))
    enroll_iv = enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
    enroll_iv.write('data/iv_sre10_core-core_test_{}.h5'.format(distrib_nb))


def test_plp_i_vector_extract():
    distrib_nb = 1024
    nbThread = 1
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5("data/TV_plp_small_{}".format(distrib_nb))
    enroll_stat = sidekit.StatServer('data/plp_small_stat_sre10_core-core_test_{}.h5'.format(distrib_nb))
    enroll_iv = enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
    enroll_iv.write('data/plp_small_iv_sre10_core-core_test_{}.h5'.format(distrib_nb))


def identification_evaluate(score, test_key):
    score_mat = score.scoremat.T
    test_key_mask = test_key.tar.T
    total_num = 0
    valid_num = 0
    for idx, spk_score in enumerate(score_mat):
        target_idx = numpy.where(test_key_mask[idx] == True)
        max_idx = spk_score .argmax()
        max_val = spk_score.max()
        if max_idx in target_idx:
            valid_num += 1
        total_num += 1
    print("total num: {}, valid num: {}-->accuracy {}%".format(total_num, valid_num, valid_num*100./total_num))


def test_ivector_score(feat_dir):
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feat_dir)
    test_ndx = get_ndx_map(leftlist, rightlist)
    test_key = get_key_map(test_ndx, leftlist, rightlist)
    enroll_iv = sidekit.StatServer('data/iv_sre10_core-core_enroll_{}.h5'.format(distrib_nb))
    test_iv = sidekit.StatServer('data/iv_sre10_core-core_test_{}.h5'.format(distrib_nb))
    scores_cos = sidekit.iv_scoring.cosine_scoring(enroll_iv, test_iv, test_ndx, wccn=None)
    # Set the prior following NIST-SRE 2010 settings
    prior = sidekit.logit_effective_prior(0.001, 1, 1)
    # Initialize the DET plot to 2010 settings
    dp = sidekit.DetPlot(window_style='sre10', plot_title='I-Vectors SRE 2010-ext male, cond 5')
    dp.set_system_from_scores(scores_cos, test_key, sys_name='Cosine')
    dp.create_figure()
    dp.plot_rocch_det(0)
    dp.plot_DR30_both(idx=0)
    dp.plot_mindcf_point(prior, idx=0)
    minDCF, Pmiss, Pfa, prbep, eer = sidekit.bosaris.detplot.fast_minDCF(dp.__tar__[0], dp.__non__[0], prior, normalize=True)
    print("minDCF: {}, eer: {}".format(minDCF, eer))
    identification_evaluate(scores_cos, test_key)


def test_plp_ivector_score(feat_dir):
    distrib_nb = 1024
    leftlist, rightlist = id_map_list(feat_dir)
    test_ndx = get_ndx_map(leftlist, rightlist)
    test_key = get_key_map(test_ndx, leftlist, rightlist)
    enroll_iv = sidekit.StatServer('data/plp_small_iv_sre10_core-core_enroll_{}.h5'.format(distrib_nb))
    test_iv = sidekit.StatServer('data/plp_small_iv_sre10_core-core_test_{}.h5'.format(distrib_nb))
    scores_cos = sidekit.iv_scoring.cosine_scoring(enroll_iv, test_iv, test_ndx, wccn=None)
    # Set the prior following NIST-SRE 2010 settings
    prior = sidekit.logit_effective_prior(0.001, 1, 1)
    # Initialize the DET plot to 2010 settings
    dp = sidekit.DetPlot(window_style='sre10', plot_title='I-Vectors SRE 2010-ext male, cond 5')
    dp.set_system_from_scores(scores_cos, test_key, sys_name='Cosine')
    dp.create_figure()
    dp.plot_rocch_det(0)
    dp.plot_DR30_both(idx=0)
    dp.plot_mindcf_point(prior, idx=0)
    minDCF, Pmiss, Pfa, prbep, eer = sidekit.bosaris.detplot.fast_minDCF(dp.__tar__[0], dp.__non__[0], prior, normalize=True)
    print("minDCF: {}, eer: {}".format(minDCF, eer))


def write_list_to_txt(txt_file, tmp_list):
    txt_data = ",".join(tmp_list)
    with open(txt_file, "w") as f:
        f.write(txt_data)


def factor_analysis_tv_mat_16(feat_dir):
    nbThread = 1
    distrib_nb = 1024
    rank_TV = 400
    tv_iteration = 10
    leftlist, rightlist = id_map_list(feat_dir)
    write_list_to_txt("left_data.txt", leftlist)
    write_list_to_txt("rigth_list.txt", rightlist)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_16_1024.h5')
    tv_idmap = get_id_map(leftlist, rightlist)
    tv_stat = sidekit.StatServer.read_subset('../data-speaker-verification/stat_sre10_core-core_16_tv_{}.h5'.format(distrib_nb), tv_idmap)
    tv_mean, tv, _, __, tv_sigma = tv_stat.factor_analysis(rank_f = rank_TV,
                                                           rank_g = 0,
                                                           rank_h = None,
                                                           re_estimate_residual = False,
                                                           it_nb = (tv_iteration,0,0),
                                                           min_div = True,
                                                           ubm = ubm,
                                                           batch_size = 100,
                                                           num_thread = nbThread,
                                                           save_partial = "data/TV_16_{}".format(distrib_nb))
    sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma), "data/TV_16_{}".format(distrib_nb))


def factor_analysis_plp_tv_mat(feat_dir):
    nbThread = 1
    distrib_nb = 1024
    rank_TV = 200
    tv_iteration = 10
    leftlist, rightlist = id_map_list(feat_dir)
    ubm = sidekit.Mixture(mixture_file_name='gmm/ubm_plp_1024.h5')
    tv_idmap = get_id_map(leftlist, rightlist)
    tv_stat = sidekit.StatServer.read_subset('data/plp_small_stat_sre10_core-core_enroll_{}.h5'.format(distrib_nb), tv_idmap)
    tv_mean, tv, _, __, tv_sigma = tv_stat.factor_analysis(rank_f = rank_TV,
                                                           rank_g = 0,
                                                           rank_h = None,
                                                           re_estimate_residual = False,
                                                           it_nb = (tv_iteration,0,0),
                                                           min_div = True,
                                                           ubm = ubm,
                                                           batch_size = 100,
                                                           num_thread = nbThread,
                                                           save_partial = "data/TV_plp_small_{}".format(distrib_nb))
    sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma), "data/TV_plp_small_{}".format(distrib_nb))


def single_plp_ivector_extract():
    from sidekit.factor_analyser import FactorAnalyser
    distrib_nb = 1024
    nbThread = 1
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5("data/TV_{}".format(distrib_nb))
    test_feat_dir = "D:/Corpus/myTest"



def i_vector_extract():
    distrib_nb = 1024
    nbThread = 1
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5("data/TV_{}".format(distrib_nb))
    enroll_stat = sidekit.StatServer('data/stat_sre10_core-core_enroll_{}.h5'.format(distrib_nb))
    enroll_iv = enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
    enroll_iv.write('data/iv_sre10_core-core_enroll_{}.h5'.format(distrib_nb))


def plp_i_vector_extract():
    distrib_nb = 1024
    nbThread = 1
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5("data/TV_plp_small_{}".format(distrib_nb))
    enroll_stat = sidekit.StatServer('data/plp_small_stat_sre10_core-core_enroll_{}.h5'.format(distrib_nb))
    enroll_iv = enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
    enroll_iv.write('data/plp_small_iv_sre10_core-core_enroll_{}.h5'.format(distrib_nb))


if __name__ == "__main__":
    ubm_feature_dir = "D:/Corpus/LibriSpeechUBMPLPSMall"
    tv_feature_dir = "H:/HJB/Corpus/LibiSpeechTVMFCC16"
    enroll_feat_dir = "D:/Corpus/LibiSpeechSmallTestPLP"
    # train_tv_matrix(ubm_feature_dir) # "D:/Corpus/LibiSpeechUBMMFCC"
    # train_tv_matrix_16K(tv_feature_dir)
    # factor_analysis_tv_mat(ubm_feature_dir)
    # factor_analysis_tv_mat_16(tv_feature_dir)
    # enroll_tv_matrix(enroll_feat_dir)
    # i_vector_extract()
    tv_plp_feature_dir = "H:/HJB/Corpus/LibiSpeechTVPLP16"
    # train_plp_tv_matrix(ubm_feature_dir)
    # factor_analysis_plp_tv_mat(ubm_feature_dir)
    # plp_i_vector_extract()
    # test_tv_plp_matrix(enroll_feat_dir)
    # test_plp_i_vector_extract()
    train_plp_16_tv_matrix(tv_plp_feature_dir)


