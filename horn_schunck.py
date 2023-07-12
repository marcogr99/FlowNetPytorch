import sys
import numpy as np
import cv2 as cv
import os.path
from vsp.detector import CvBlobDetector
from vsp.encoder import KeypointEncoder

def main():
  if len(sys.argv) < 2:
    sys.exit()

  # to do: input from outside
  # call from R in data analysis
  speeds = [10,20,30,40,50]
  procedure_id = sys.argv[1]

  sub_img_dim = 300

  blob_detector = CvBlobDetector( min_threshold=0.0,
                                  max_threshold=50.0,
                                  filter_by_color=True,
                                  blob_color=0,
                                  filter_by_area=False,
                                  # min_area=17.05,
                                  # max_area=135.46,
                                  filter_by_circularity=True,
                                  min_circularity=0.62,
                                  filter_by_inertia=True,
                                  min_inertia_ratio=0.27,
                                  filter_by_convexity=True,
                                  min_convexity=0.60 )
  keypoints_encoder = KeypointEncoder()

  for speed in speeds:
    flow_file = open("../../data/" + procedure_id + "_" + str(speed) + "_flow_data.csv", "w")
    flow_file.write("idx,frame_start_idx,frame_end_idx,x_coord,y_coord,flow_mag,flow_ang,dvdx,dudy\n")

    kps_file = open("../../data/" + procedure_id + "_" + str(speed) + "_keypoints.csv", "w")
    kps_file.write("idx,frame_idx,x_coord,y_coord,radius\n")

    frame_idx = 1
    frame_start_idx  = frame_idx
    path = "../../bin/data/tactip/" + procedure_id + "_" + str(speed) + "_" + str(frame_idx) + ".jpg"
    if not os.path.isfile(path):
      sys.exit(0)

    frame_start      = cv.imread(path, cv.IMREAD_GRAYSCALE)
    frame_start      = frame_start[95:(95+sub_img_dim), 150:(150+sub_img_dim)]
    ret, frame_start = cv.threshold(frame_start, 50, 255, cv.THRESH_BINARY_INV)

    keypoints = keypoints_encoder(blob_detector.detect(frame_start))
    kps_data = np.hstack((np.ones((keypoints.shape[0],1),dtype=int)*frame_idx,
                          np.ones((keypoints.shape[0],1),dtype=int)*frame_start_idx,
                          keypoints))
    np.savetxt(kps_file, kps_data, delimiter = ',')

    alpha = 0.1
    idx = 1
    while True:
      frame_idx += 1
      path = "../../bin/data/tactip/" + procedure_id + "_" + str(speed) + "_" + str(frame_idx) + ".jpg"
      if not os.path.isfile(path):
        break

      frame_end      = cv.imread(path, cv.IMREAD_GRAYSCALE)
      frame_end      = frame_end[95:(95+sub_img_dim), 150:(150+sub_img_dim)]
      ret, frame_end = cv.threshold(frame_end, 50, 255, cv.THRESH_BINARY_INV)
      
      # if not different read next image
      diff_frame = frame_start != frame_end
      num_diff_pix = cv.countNonZero(diff_frame.astype(dtype=float))
      if num_diff_pix < (sub_img_dim*sub_img_dim*0.05):
        continue
      else:
        print("Num diff px: ", num_diff_pix)

      frame_end_idx = frame_idx

      print("From FRAME ", frame_start_idx, " To FRAME ", frame_end_idx)
      print("IDX ", idx)
      print("----------------------------------------")

      rows_num   = frame_start.shape[0]
      cols_num   = frame_start.shape[1]
      pixels_num = rows_num*cols_num

      #  to do: use previous flow as initial guess
      u      = np.zeros_like(frame_start, dtype=float)
      v      = np.zeros_like(frame_start, dtype=float)
      dvdx   = np.zeros_like(frame_start, dtype=float)
      dudy   = np.zeros_like(frame_start, dtype=float)
      u_mean = np.zeros_like(frame_start, dtype=float)
      v_mean = np.zeros_like(frame_start, dtype=float)
      Ex     = np.zeros_like(frame_start, dtype=float)
      Ey     = np.zeros_like(frame_start, dtype=float)
      Et     = np.zeros_like(frame_start, dtype=float)
      mask   = np.zeros_like(frame_start, dtype=int)

      frame_start_f = frame_start.astype(dtype=float)
      frame_end_f   = frame_end.astype(dtype=float)

      Ex = (1.0/4.0)*(np.pad(frame_start_f[:,1:] ,((0,0),(0,1)),'edge') - frame_start_f +
                      np.pad(frame_start_f[1:,1:],((0,1),(0,1)),'edge') - np.pad(frame_start_f[1:,:],((0,1),(0,0)),'edge') +
                      np.pad(frame_end_f[:,1:]   ,((0,0),(0,1)),'edge') - frame_end_f +
                      np.pad(frame_end_f[1:,1:]  ,((0,1),(0,1)),'edge') - np.pad(frame_end_f[1:,:]  ,((0,1),(0,0)),'edge'))

      Ey = (1.0/4.0)*(np.pad(frame_start_f[1:,:] ,((0,1),(0,0)),'edge') - frame_start_f +
                      np.pad(frame_start_f[1:,1:],((0,1),(0,1)),'edge') - np.pad(frame_start_f[:,1:],((0,0),(0,1)),'edge') +
                      np.pad(frame_end_f[1:,:]   ,((0,1),(0,0)),'edge') - frame_end_f +
                      np.pad(frame_end_f[1:,1:]  ,((0,1),(0,1)),'edge') - np.pad(frame_end_f[:,1:]  ,((0,0),(0,1)),'edge'))

      Et = (1.0/4.0)*(frame_end_f                                     - frame_start_f +
                      np.pad(frame_end_f[1:,:] ,((0,1),(0,0)),'edge') - np.pad(frame_start_f[1:,:] ,((0,1),(0,0)),'edge') +
                      np.pad(frame_end_f[:,1:] ,((0,0),(0,1)),'edge') - np.pad(frame_start_f[:,1:] ,((0,0),(0,1)),'edge') +
                      np.pad(frame_end_f[1:,1:],((0,1),(0,1)),'edge') - np.pad(frame_start_f[1:,1:],((0,1),(0,1)),'edge'))

      # to do: set rep_num to image cross-section
      # for rep_num in range(np.amax(frame_start.shape)):
      for rep_num in range(50):
        u_mean = (1.0/6.0)*(np.pad(u[:-1,:],((1,0),(0,0))) +
                            np.pad(u[:,1:] ,((0,0),(0,1))) +
                            np.pad(u[1:,:] ,((0,1),(0,0))) +
                            np.pad(u[:,:-1],((0,0),(1,0)))) + \
                  (1.0/12.0)*(np.pad(u[:-1,:-1],((1,0),(1,0))) +
                            np.pad(u[1:,1:]    ,((0,1),(0,1))) +
                            np.pad(u[:-1,1:]   ,((1,0),(0,1))) +
                            np.pad(u[1:,:-1]   ,((0,1),(1,0))))

        v_mean = (1.0/6.0)*(np.pad(v[:-1,:],((1,0),(0,0))) +
                            np.pad(v[:,1:] ,((0,0),(0,1))) +
                            np.pad(v[1:,:] ,((0,1),(0,0))) +
                            np.pad(v[:,:-1],((0,0),(1,0)))) + \
                  (1.0/12.0)*(np.pad(v[:-1,:-1],((1,0),(1,0))) + 
                            np.pad(v[1:,1:]    ,((0,1),(0,1))) +
                            np.pad(v[:-1,1:]   ,((1,0),(0,1))) +
                            np.pad(v[1:,:-1]   ,((0,1),(1,0))))

        u = u_mean - Ex*(Ex*u_mean + Ey*v_mean + Et)/(alpha**2 + Ex**2 + Ey**2)
        v = v_mean - Ey*(Ex*u_mean + Ey*v_mean + Et)/(alpha**2 + Ex**2 + Ey**2)

      dvdx = np.pad(v[:,1:],((0,0),(0,1))) - np.pad(v[:,:-1],((0,0),(1,0)))
      dudy = np.pad(u[1:,:],((0,1),(0,0))) - np.pad(u[:-1,:],((1,0),(0,0)))
      dvdx = np.pad(v[:,1:],((0,0),(0,1))) - np.pad(v[:,:-1],((0,0),(1,0)))
      dudy = np.pad(u[1:,:],((0,1),(0,0))) - np.pad(u[:-1,:],((1,0),(0,0)))

      flow_mag = np.sqrt(u**2 + v**2)
      flow_ang = np.arctan2(v,u)

      flow_data = np.hstack((np.ones((pixels_num,1),dtype=int)*(idx),
                            np.ones((pixels_num,1),dtype=int)*(frame_start_idx),
                            np.ones((pixels_num,1),dtype=int)*(frame_end_idx),
                            np.tile(np.arange(1,cols_num+1,dtype=int),rows_num).reshape(pixels_num,1),
                            np.repeat(np.arange(1,rows_num+1,dtype=int),cols_num).reshape(pixels_num,1),
                            flow_mag.reshape(pixels_num,1),
                            flow_ang.reshape(pixels_num,1),
                            dvdx.reshape(pixels_num,1),
                            dudy.reshape(pixels_num,1)))

      np.savetxt(flow_file, flow_data, delimiter = ',')

      keypoints = keypoints_encoder(blob_detector.detect(frame_end))
      kps_data = np.hstack((np.ones((keypoints.shape[0],1),dtype=int)*idx,
                            np.ones((keypoints.shape[0],1),dtype=int)*frame_end_idx,
                            keypoints))
      np.savetxt(kps_file, kps_data, delimiter = ',')

      idx = idx + 1
      frame_start = frame_end
      frame_start_idx = frame_end_idx

    flow_file.close()
    kps_file.close()


if __name__ == "__main__":
  main()