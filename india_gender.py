import verify
from splitter import speech_segmentation
from textgrid import TextGrid
from config import config
import os
import sys
import warnings
import argparse
from multiprocessing import Process, Queue
import shutil
import time
import queue

warnings.filterwarnings("ignore")
tmp_file = config.temp_audio


class GenderIdentifier:

    def __init__(self, proc=4):
        self.total_results = []
        self.number_of_task = 0
        self.number_of_processes = proc
        self.tasks_to_accomplish = Queue()
        self.tasks_that_are_done = Queue()
        self.processes = []

    def proc_process(self):
        while True:
            gender = 'female'
            try:
                task = self.tasks_to_accomplish.get_nowait()
                file, st, ed, verbose = task
                temp_dir = os.path.join(config.wave_dir, 'temp')
                os.makedirs(temp_dir, exist_ok=True)

                temp_wavefile = os.path.join(temp_dir, "temp_{}_{}.wav".format(st, ed))
                if os.path.exists(temp_wavefile):
                    # os.remove(temp_wavefile)
                    temp_wavefile = "{}_{}.wav".format(tmp_file[:-4], tmp_file[-5])
                if speech_segmentation.split_audio(file, st, ed, temp_wavefile):
                    gender = verify.verifier(temp_wavefile)
                    # self.total_results.append([st, ed, gender])
                    if verbose:
                        print(" [{:.2f}, {:.2f}]   :  {}".format(st, ed, gender))
                        print("----------------------------------------------------")
                else:
                    gender = "noise"
            except queue.Empty:
                break
            else:
                self.tasks_that_are_done.put([st, ed, gender])
                time.sleep(.1)
        return True

    def process(self, file, verbose=False):
        if not os.path.isfile(file):
            print("No such file exist: {}".format(file))
            return
        if not file.lower().endswith(".wav"):
            print("Invalid file type!")
            print("Please input wave file with 8kHz.")
            return

        if verbose:
            print("%10s %8s %1s" % ("--> TESTING", ":", os.path.basename(file)))
        self.total_results = []
        # extract features
        try:
            segs = speech_segmentation.multi_segmentation(file=file, sr=8000, frame_size=0.1, frame_shift=0.025)
            self.processes = []
            self.number_of_task = len(segs)

            for seg in segs:
                st = seg[0]
                ed = seg[1]
                self.tasks_to_accomplish.put([file, st, ed, verbose])

            # creating processes
            for w in range(self.number_of_processes):
                p = Process(target=self.proc_process)
                self.processes.append(p)
                p.start()

            # completing process
            for p in self.processes:
                p.join()

            while not self.tasks_that_are_done.empty():
                res = self.tasks_that_are_done.get()
                self.total_results.append(res)
        except:
            pass

        textgrid_file = file[:-4] + '.TextGrid'
        self.out_textgrid(textgrid_file)

    def out_textgrid(self, dst_file):
        if len(self.total_results) == 0:
            return
        self.total_results.sort(key=lambda x: x[0])
        tgt = TextGrid()
        tgt.add_tier_list("Bookmark", self.total_results)
        # tgt.add_tier_list("SpeakerID", self.total_results)

        tgt.to_TextGrid(dst_file)


def main():
    parser = argparse.ArgumentParser(description="India Gender Recognition")
    parser.add_argument('--file', help='Path to wave file')
    # parser.add_argument('--dir', help='Folder Path to wave files')
    parser.add_argument('--verbose', '-V', default='false', help='log showing')
    parser.add_argument('--proc', '-P', default=4, help='number of parallel process')

    # args = parser.parse_args()
    args, leftovers = parser.parse_known_args()
    if args.file is not None:
        audio_file = args.file
    else:
        print("Audio file is empty!")
        return

    proc = 4
    if args.proc is not None:
        proc = int(args.proc)

    show = False
    if args.verbose is not None:
        show = True if str(args.verbose).lower() == 'true' else False

    if not os.path.exists(audio_file):
        print("No such file: {}".format(audio_file))
        sys.exit(1)

    try:
        shutil.rmtree(os.path.join(config.feat_dir, 'temp'), ignore_errors=True)
        shutil.rmtree(os.path.join(config.wave_dir, 'temp'), ignore_errors=True)
    except:
        print("Failed to remove temporary files.")
        pass

    t1 = time.time()
    gender_identifier = GenderIdentifier(proc)
    gender_identifier.process(audio_file, show)
    t2 = time.time()
    delta_time = t2 - t1
    print("time: ", delta_time)


def save_to_file(data, filename):
    import json
    with open(filename, 'w') as fp:
        json.dump(data, fp)


if __name__ == "__main__":
    main()

    """
    import librosa
    import webrtcvad

    vad = webrtcvad.Vad(1)

    parser = argparse.ArgumentParser(description="India Gender Recognition")
    parser.add_argument('--file', help='Path to wave file')

    # args = parser.parse_args()
    args, leftovers = parser.parse_known_args()
    if args.file is None:
        exit(1)
    audio_file = args.file

    y = open(audio_file, 'rb').read()
    i = 0
    while True:
        chunk = y[1024*i: 1024*(i+1)]
        active = vad.is_speech(chunk, 8000)
        if active:
            print(1)
        else:
            print(0)
    """

