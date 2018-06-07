import dlib
from contextlib import closing
from videosequence import VideoSequence
from skimage import  io

class FaceTracker():
    def __init__(self,detector,object_tracker):
        self.detector=detector
        self.object_tracker=object_tracker


    def start_tracking(self,video_path,detector_min_conf=0.3,tracker_min_conf=20,aligners=[]):
        frames_faces_dic={}
        with closing(VideoSequence(video_path)) as frames:
            is_tracking = False

            for idx, frame in enumerate(frames):
                face_rect=None
                tmp_frme_path="temp.png"
                frame.save(tmp_frme_path)
                input = io.imread(tmp_frme_path)

                if is_tracking:
                    confi = self.object_tracker.update(input)
                    if confi > tracker_min_conf:
                        d = self.object_tracker.get_position()
                        face_rect = dlib.rectangle(int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
                    else:
                        is_tracking = False

                if not is_tracking:
                    dets = self.detector(input)
                    #face is detected start tracking
                    if len(dets) > 0:

                        if (hasattr(dets[0], 'confidence')):
                            if dets[0].confidence > detector_min_conf :
                                is_tracking = True
                                face_rect=dets[0].rect
                                self.object_tracker.start_track(input, face_rect)
                        else:
                            is_tracking = True
                            face_rect = dets[0]
                            self.object_tracker.start_track(input, face_rect)

                if face_rect is not None:
                    frames_faces_dic[idx]=face_rect
                    for aligner in aligners:
                        file_name=str(idx).zfill(5) + ".jpg"
                        aligner.align_face(tmp_frme_path,face_rect,file_name)
        return frames_faces_dic