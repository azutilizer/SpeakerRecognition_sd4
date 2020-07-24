import os
import shutil
from feature_utils import safe_makedir
from config import config


def main():
    data_folder = config.base_dir
    train_data_folder = os.path.join(data_folder, "TrainingData")
    enroll_data_folder = os.path.join(config.wave_dir, "enroll")
    test_data_folder = os.path.join(config.wave_dir, "test")

    safe_makedir(enroll_data_folder)
    safe_makedir(test_data_folder)

    for name in os.listdir(train_data_folder):
        sub_data_folder = os.path.join(train_data_folder, name)
        file_list = []
        for root, dirnames, filenames in os.walk(sub_data_folder):
            file_list.extend([os.path.join(root, fname) for fname in filenames])
        data_len = len(file_list)
        enroll_data_len = int(data_len * 0.8)

        print("name: {}".format(name))
        print("data_len: {}".format(data_len))
        print("enroll_data_len: {}".format(enroll_data_len))

        for i, file in enumerate(file_list):
            if i < enroll_data_len:
                shutil.copy(file, os.path.join(enroll_data_folder, os.path.basename(file)))
            else:
                shutil.copy(file, os.path.join(test_data_folder, os.path.basename(file)))


if __name__ == '__main__':
    main()

