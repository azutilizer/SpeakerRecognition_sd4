import os
import sidekit
import sidekit_util
import multiprocessing
from config import config

cpu_cnt = int(config.TrainingParamsParser['nbThread'])
# cpu_cnt = multiprocessing.cpu_count()


def ubm_file_list(ubm_feat_dir):
    file_list = []
    for h5_file in os.listdir(ubm_feat_dir):
        if h5_file.find('.h5') == -1:
            continue

        file_list.append(h5_file.split('.h5')[0])
    return file_list


def train_ubm_split_mfcc(ubm_feature_dir):
    feat_dir = os.path.join(ubm_feature_dir, "{}.h5")
    features_server = sidekit_util.get_feature_server(feat_dir)
    ubm_list = ubm_file_list(ubm_feature_dir)
    distrib_nb = int(config.TrainingParamsParser['distrib_nb'])
    nbThread = cpu_cnt

    ubm = sidekit.Mixture()
    llk = ubm.EM_split(features_server=features_server,
                       feature_list=ubm_list,
                       distrib_nb=distrib_nb,
                       iterations=(1, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8),
                       num_thread=nbThread,
                       save_partial='ubm',
                       output_file_name=os.path.join(config.data_dir, "gender_ubm"),
                       ceil_cov=10,
                       floor_cov=1e-2)

    ubm_dir = config.ubm_dir
    ubm.write(os.path.join(ubm_dir, "ubm_{}.h5".format(distrib_nb)))
    print("training llk-len, llk: {}".format(len(llk), llk))


def main():
    feature_dir = config.get_enroll_feat_dir
    train_ubm_split_mfcc(feature_dir)


if __name__ == '__main__':
    main()
