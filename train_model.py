import data_init
import feature_utils
import ubm
import iVector


# split data into enroll and test
data_init.main()

# feature extraction
feature_utils.main()

# ubm training based on MFCC
ubm.main()

# i-vector extraction and tv matrix writing
iVector.main()
