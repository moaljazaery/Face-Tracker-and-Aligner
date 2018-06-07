import dlib
import os

from FaceAligner import FaceAligner
from FaceTracker import FaceTracker
from Utils import Utils


# Input list of video paths
# The script reads the videos and Track the faces
# Then output the faces aligned using two different alignment (Tight and Loose)

#inputs
tight_faces_out_folder = "./output/tight/"
loose_faces_out_folder = "./output/loose/"
input_videos = ["sample.avi"]
use_gpu=True

Utils.mkdir_if_not_exist(tight_faces_out_folder)
Utils.mkdir_if_not_exist(loose_faces_out_folder)

detector = dlib.get_frontal_face_detector()

# init the face detector
if use_gpu:
    face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
else:
    # faster face detector
    face_detector =  dlib.get_frontal_face_detector()

#init the face landmarks detector
predictor_5_face_landmarks = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")

#init the object tracker
object_tracker = dlib.correlation_tracker()

# tight face aligner : padding = 0.2
aligner_tight = FaceAligner(face_size=112, face_padding=0.2, predictor_5_face_landmarks=predictor_5_face_landmarks)

# loose face aligner : padding = 0.4
aligner_loose = FaceAligner(face_size=112, face_padding=0.4, predictor_5_face_landmarks=predictor_5_face_landmarks)

# init the Face tracker
face_tracker = FaceTracker(face_detector, object_tracker)
aligners = []

for video_path in input_videos:
    video_key = os.path.basename(video_path)
    #set the out folder of the tight aligner
    aligner_tight.out_dir = os.path.join(tight_faces_out_folder, video_key)
    Utils.mkdir_if_not_exist(aligner_tight.out_dir)

    # set the out folder of the loose aligner
    aligner_loose.out_dir = os.path.join(loose_faces_out_folder, video_key)
    Utils.mkdir_if_not_exist(aligner_loose.out_dir)

    face_tracker.start_tracking(video_path, aligners=[aligner_loose, aligner_tight],detector_min_conf=0.3,tracker_min_conf=20)
