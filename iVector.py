import os
import sidekit_util
import sidekit
import multiprocessing
from config import config
cpu_cnt = int(config.TrainingParamsParser['nbThread'])
# cpu_cnt = multiprocessing.cpu_count()


def train_tv_matrix(feature_dir):
    enroll_stat = sidekit_util.adatpt_stats_with_feat_dir(
        feature_dir,
        config.ubm_path
    )
    enroll_stat.write(config.stat_path)


def test_i_vector_extract():
    distrib_nb = int(config.TrainingParamsParser['distrib_nb'])
    nbThread = cpu_cnt
    tv, tv_mean, tv_sigma = sidekit.sidekit_io.read_tv_hdf5(config.tv_path)
    enroll_stat = sidekit.StatServer('./results/data/stat_sre10_core-core_test_{}.h5'.format(distrib_nb))
    enroll_iv = enroll_stat.estimate_hidden(tv_mean, tv_sigma, V=tv, batch_size=100, num_thread=nbThread)[0]
    enroll_iv.write('./results/data/iv_sre10_core-core_test_{}.h5'.format(distrib_nb))


def write_list_to_txt(txt_file, tmp_list):
    txt_data = ",".join(tmp_list)
    with open(txt_file, "w") as f:
        f.write(txt_data)


def factor_analysis_tv_mat(feat_dir):
    nbThread = cpu_cnt
    rank_TV = int(config.TrainingParamsParser['rank_TV'])
    tv_iteration = int(config.TrainingParamsParser['tv_iteration'])
    leftlist, rightlist = sidekit_util.id_map_list(feat_dir)
    write_list_to_txt(os.path.join(config.base_dir, "left_data.txt"), leftlist)
    write_list_to_txt(os.path.join(config.base_dir, "rigth_list.txt"), rightlist)

    ubm_path = config.ubm_path
    ubm = sidekit.Mixture(mixture_file_name=ubm_path)
    tv_idmap = sidekit_util.get_id_map(leftlist, rightlist)

    stat_file = config.stat_path
    tv_stat = sidekit.StatServer.read_subset(stat_file, tv_idmap)
    tv_mean, tv, _, __, tv_sigma = tv_stat.factor_analysis(rank_f=rank_TV,
                                                           rank_g=0,
                                                           rank_h=None,
                                                           re_estimate_residual=False,
                                                           it_nb=(tv_iteration, 0, 0),
                                                           min_div=True,
                                                           ubm=ubm,
                                                           batch_size=int(config.TrainingParamsParser['batch_size']),
                                                           num_thread=nbThread,
                                                           save_partial=config.tv_path)
    sidekit.sidekit_io.write_tv_hdf5((tv, tv_mean, tv_sigma), config.tv_path)


def main():
    ubm_feature_dir = config.get_enroll_feat_dir
    tv_feature_dir = config.stat_path
    train_tv_matrix(ubm_feature_dir)
    factor_analysis_tv_mat(ubm_feature_dir)

    # write enroll data
    sidekit_util.adaption_all_enrolled_speakers()


if __name__ == '__main__':
    main()
