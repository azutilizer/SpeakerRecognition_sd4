import os
import numpy
import sidekit
import sidekit_util
from config import config

show_name = "test"
extractor = sidekit_util.get_feature_extractor()
leftlist = [show_name]
rightlist = [show_name]
test_ndx = sidekit_util.get_ndx_map(leftlist, rightlist)
enroll_iv = sidekit.StatServer(config.get_enroll_data)
enroll_iv.norm_stat1()


def verifier(audio_file):
    sub_feat_name = os.path.basename(audio_file).split('.wav')[0]

    temp_dir = os.path.join(config.feat_dir, 'temp', sub_feat_name)
    os.makedirs(temp_dir, exist_ok=True)
    feat_path = os.path.join(temp_dir, "{}.h5".format(show_name))

    extractor.save(show=show_name,
                   channel=0,
                   input_audio_filename=audio_file,
                   output_feature_filename=feat_path)

    speaker_stat = sidekit_util.adatpt_stats_with_feat_dir(
        temp_dir,  # config.feat_dir,
        config.ubm_path
    )

    spk_i_vect, _ = sidekit_util.extract_i_vector(speaker_stat)
    rec_name = verified_speaker(spk_i_vect)
    return rec_name


def verified_speaker(spk_iv):
    spk_iv.norm_stat1()
    scores_cos = numpy.dot(enroll_iv.stat1, spk_iv.stat1.transpose())
    numpy.reshape(scores_cos, len(scores_cos))
    max_idx = numpy.argmax(scores_cos)
    # print(scores_cos)
    return enroll_iv.modelset[max_idx]


"""
if __name__ == "__main__":
    test_file = 'male.00006.wav'
    import shutil
    shutil.rmtree(os.path.join(config.feat_dir, 'temp'), ignore_errors=True)
    result = verifier(test_file)
    print('file: {}  :  {}'.format(os.path.basename(test_file), result))
"""
