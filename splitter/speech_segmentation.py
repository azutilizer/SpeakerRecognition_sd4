import os
import numpy as np
from . import voice_activity_detect as vad
import librosa
from collections import OrderedDict


def get_duration(wave_file):
    y, fs = librosa.load(wave_file, sr=8000)
    nFrames = len(y)
    audio_length = nFrames * (1 / fs)

    return audio_length


def split_audio(wave_file, st_time, ed_time, dst_file):
    """
    :param wave_file: input file
    :param st_time: start time (seconds)
    :param ed_time: end time (seconds)
    :param dst_file:
    :return:
    """
    if os.path.exists(dst_file):
        os.remove(dst_file)
    cmds = "ffmpeg -i \"{}\" -ss {} -to {} \"{}\" -acodec pcm_s16le -ar 8000 -ac 1 -y -loglevel panic".format(
        wave_file, st_time, ed_time, dst_file
    )
    try:
        os.system(cmds)
        return True
    except:
        return False


def cluster_greedy(feature_vectors, cluster_list):
    """
    Use the feature of the first segment to be the baseline, and loop over all the other
    segments to compute the BIC distance between each one and the baseline. Set a BIC
    distance threshold, decide whether this segment is in the same cluster with the baseline
    """
    current_cluster_number = len(cluster_list)
    for index, key in enumerate(feature_vectors.keys()):
        if index == 0:
            base_feature = feature_vectors[key]
            cluster_list[str(current_cluster_number)]=list()
            temp = cluster_list[str(current_cluster_number)]
            temp.append(key)
            cluster_list[str(current_cluster_number)] = temp
        else:
            bic_dis = cluter_on_bic(base_feature, feature_vectors[key])
            if bic_dis < 50:
                temp = cluster_list[str(current_cluster_number)]
                temp.append(key)
                cluster_list[str(current_cluster_number)] = temp

    # Delete the segment related features if they are already clustered.
    for i in cluster_list[str(current_cluster_number)]:
        feature_vectors.pop(i)


def cluter_on_bic(mfcc_s1, mfcc_s2):
    """
    Compute BIC distance between two MFCC features
    """
    mfcc_s = np.concatenate((mfcc_s1, mfcc_s2), axis=1)

    m, n = mfcc_s.shape
    m, n1 = mfcc_s1.shape
    m, n2 = mfcc_s2.shape

    sigma0 = np.cov(mfcc_s).diagonal()
    eps = np.spacing(1)
    realmin = np.finfo(np.double).tiny
    det0 = max(np.prod(np.maximum(sigma0,eps)),realmin)

    part1 = mfcc_s1
    part2 = mfcc_s2

    sigma1 = np.cov(part1).diagonal()
    sigma2 = np.cov(part2).diagonal()

    det1 = max(np.prod(np.maximum(sigma1, eps)), realmin)
    det2 = max(np.prod(np.maximum(sigma2, eps)), realmin)

    BIC = 0.5 * (n * np.log(det0) - n1 * np.log(det1) - n2 * np.log(det2)) - 0.5 * (m + 0.5 * m * (m + 1)) * np.log(n)
    return BIC


def compute_bic(mfcc_v, delta):
    """Speech segmentation based on BIC"""
    m, n = mfcc_v.shape

    sigma0 = np.cov(mfcc_v).diagonal()
    eps = np.spacing(1)
    realmin = np.finfo(np.double).tiny
    det0 = max(np.prod(np.maximum(sigma0, eps)), realmin)

    flat_start = 5

    range_loop = range(flat_start, n, delta)
    x = np.zeros(len(range_loop))
    iter = 0
    for index in range_loop:
        part1 = mfcc_v[:, 0:index]
        part2 = mfcc_v[:, index:n]

        sigma1 = np.cov(part1).diagonal()
        sigma2 = np.cov(part2).diagonal()

        det1 = max(np.prod(np.maximum(sigma1, eps)), realmin)
        det2 = max(np.prod(np.maximum(sigma2, eps)), realmin)

        BIC = 0.5*(n*np.log(det0)-index*np.log(det1)-(n-index)*np.log(det2))-0.5*(m+0.5*m*(m+1))*np.log(n)
        x[iter] = BIC
        iter = iter + 1

    maxBIC = x.max()
    maxIndex = x.argmax()
    if maxBIC > 0:
        return range_loop[maxIndex]-1
    else:
        return -1


def speech_segmentation(mfccs):
    wStart = 0
    wEnd = 50
    wGrow = 50
    delta = 10

    m, n = mfccs.shape

    store_cp = []
    index = 0
    while wEnd < n:
        featureSeg = mfccs[:, wStart:wEnd]
        detBIC = compute_bic(featureSeg, delta)
        index = index + 1
        if detBIC > 0:
            temp = wStart + detBIC
            store_cp.append(temp)
            wStart = wStart + detBIC + 50
            wEnd = wStart + wGrow
        else:
            wEnd = wEnd + wGrow

    return np.array(store_cp)


def multi_segmentation(file, sr, frame_size, frame_shift):
    dur = get_duration(file)
    y, sr = librosa.load(file, sr=sr)

    frame_size = int(frame_size * sr)
    frame_shift = int(frame_shift * sr)
    mfccs = librosa.feature.mfcc(y, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
    seg_point = speech_segmentation(mfccs / mfccs.max())

    seg_point = seg_point * frame_shift

    seg_point = np.insert(seg_point, 0, 0)
    seg_point = np.append(seg_point, len(y)-1)
    rangeLoop = range(len(seg_point) - 1)

    output_segpoint = []
    for i in rangeLoop:
        temp = y[int(seg_point[i]):int(seg_point[i + 1])]
        # add a detection of silence before vad
        max_mean = np.mean(temp[temp.argsort()[-800:]])
        if max_mean < 0.005:
            continue
        # vad detect
        x1, x2 = vad.vad(temp, sr=sr, framelen=frame_size, frameshift=frame_shift, plot=False)
        if len(x1) == 0 or len(x2) == 0:
            continue
        elif seg_point[i + 1] == len(y) - 1:
            continue
        else:
            output_segpoint.append(seg_point[i + 1])

    # If the chosen cluster method is bic, use bic distance to perform clustering
    classify_segpoint = output_segpoint.copy()
    # Add the start and the end of the audio file
    classify_segpoint.insert(0, 0)
    classify_segpoint.append(len(y)-1)
    feature_vectors = OrderedDict()
    for i in range(len(classify_segpoint) - 1):
        tempAudio = y[int(classify_segpoint[i]):int(classify_segpoint[i + 1])]
        mfccs = librosa.feature.mfcc(tempAudio, sr, n_mfcc=12, hop_length=frame_shift, n_fft=frame_size)
        mfccs = mfccs / mfccs.max()
        feature_vectors[str(i)] = mfccs

    # Define a empty cluster before perform clustering
    cluster_list = {}
    # Call the function cluster_greedy recursively
    while len(feature_vectors.keys()) > 0:
        cluster_greedy(feature_vectors, cluster_list)

    output_segpoint.insert(0, 0)
    output_segpoint.append(dur * sr)
    tm_lists = np.asarray(output_segpoint) / float(sr)

    total_segs = []
    # print("audio length: {:.3f}s".format(dur))
    # print('There are total %d clusters' % (len(cluster_list)), 'and they are listed below: ')
    for index, key in enumerate(cluster_list.keys()):
        ids = cluster_list[key]
        segs = ["[{}, {}]".format(tm_lists[int(i)], tm_lists[int(i)+1]) for i in ids]
        segs_str = ', '.join(segs)
        # print('cluster {} : {}'.format(index, segs_str))
        total_segs.extend([[tm_lists[int(i)], tm_lists[int(i)+1]] for i in ids])

    total_segs.sort()
    return total_segs


if __name__ == '__main__':
    test_file = '1559884678_1.wav'
    segs = multi_segmentation(file=test_file, sr=8000, frame_size=0.025, frame_shift=0.01)
    print(segs)
