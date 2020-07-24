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

def write_list_to_txt(txt_file, tmp_list):
    txt_data = ",".join(tmp_list)
    with open(txt_file, "w") as f:
        f.write(txt_data)


def factor_analysis_tv_mat(feat_dir, sample_rate_k, gmm_file, stat_file):
    nbThread = 1
    distrib_nb = 1024
    rank_TV = 300
    tv_iteration = 10
    leftlist, rightlist = id_map_list(feat_dir)
    write_list_to_txt("left_data.txt", leftlist)
    write_list_to_txt("rigth_list.txt", rightlist)
    ubm = sidekit.Mixture(mixture_file_name=gmm_file)
    tv_idmap = get_id_map(leftlist, rightlist)
    tv_stat = sidekit.StatServer.read_subset(stat_file, tv_idmap)
    tv_mean, tv, _, __, tv_sigma = tv_stat.factor_analysis(rank_f = rank_TV,
                                                           rank_g = 0,
                                                           rank_h = None,
                                                           re_estimate_residual = False,
                                                           it_nb = (tv_iteration,0,0),
                                                           min_div = True,
                                                           ubm = ubm,
                                                           batch_size = 100,
                                                           num_thread = nbThread,
                                                           save_partial = "data/TV_{}_{}".format(sample_rate_k, distrib_nb))
    sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma), "data/TV_{}_{}".format(sample_rate_k, distrib_nb))


def train_factor_analysis_tv_mat_with_plp16():
    feat_dir = "/datadrive/libridata/feature/LibiSpeechTVPLP16"
    sample_rate_with_k = 16
    gmm_file = "gmm/ubm_plp_16_1024.h5"
    stat_file = "data/plp_16k_small_stat_sre10_core-core_enroll_1024.h5"
    factor_analysis_tv_mat(feat_dir, sample_rate_with_k, gmm_file, stat_file)


if __name__ == "__main__":
    train_factor_analysis_tv_mat_with_plp16()
