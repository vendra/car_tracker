__author__ = "Federico Vendramin"
__license__ = "GPL"
__email__ = "federico.vendramin@gmail.com"
__status__ = "Prototype"
__date__ = "21st November 2018"

import numpy as np
import cv2, math
import argparse 
import csv 
from vehicle import Vehicle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
ap.add_argument("-r", "--record", type=str,
    help="Name of the recorded output video file")
ap.add_argument("-c", "--csv", type=str,
    help="Name of the output .csv")
args = vars(ap.parse_args())

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	
else:
	vs = cv2.VideoCapture(args["video"])

cap = vs
tot_frame, fps, width, height = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS), \
                                cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

inter_frame_time = 1.0/fps                                
                                
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

#Blob Params
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 3000
params.maxArea = 20000
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = True
params.blobColor = 255

#Blob Detection
detector = cv2.SimpleBlobDetector_create(params)

#Record
if (args.get("record", False)):
    print("Enabled recording")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args["record"]+'.avi', fourcc, fps, (int(width),int(height)))

#CSV Output
if (args.get("csv", False)):
    outfile = open('output.csv', 'w')
    out_writer = csv.writer(outfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    out_writer.writerow(['ID', 'ENTER_FRAME', 'EXIT_FRAME', 'ENTER_TIME', 'EXIT_TIME', 'TIME'])

       
#TRACKERS
OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

#detected blobs 
keypoints = [] 
id = 1
max_blob_dist = 50
track_size = 50

#Car Counters
car_counter = 0
right_upper_lane = 0
right_lower_lane = 0
left_upper_lane = 0
left_lower_lane = 0

#Flags for pausing and advancing frame by frame
paused = False
step = False
frame_counter = 0

#Sets the line distance from the borders
begin_x = 150
#Sets height of lane split line
left_lane_split = 40
right_lane_split = 210
way_split = 143

#Vehicles Data
vehicle_list = []

#MAIN LOOP
while(1):
    if (not paused) or (step):
        step = False
        ret, frame = cap.read()
        if ret:
            frame_counter += 1
            grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fgmask = fgbg.apply(grey_frame) #Background subtraction
            
            #CONTOURS
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            #opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(closing, kernel)
        
            retvalbin, fgmask = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)
            
            contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = [cv2.convexHull(c) for c in contours]
            
            contour_frame = np.zeros((fgmask.shape[0],fgmask.shape[1],3), np.uint8)
            cv2.drawContours(contour_frame, hull, -1, (255,255,255), thickness=-1)

            
            keypoints = detector.detect(contour_frame) #Use filled contours for blob detection
            
            #im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]),
            #            (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            
            to_delete = []
            #REMOVE
            for idx in range(0, len(vehicle_list)):
                vehicle = vehicle_list[idx]
                (res, bbox) = vehicle.update(frame)
                old_bbox = vehicle.get_old_bbox()
                
                #Computes bounding box centroids
                bbox_x = bbox[0]+bbox[2]/2
                bbox_y = bbox[1]+bbox[3]/2
                old_bbox_x = old_bbox[0]+old_bbox[2]/2
                old_bbox_y = old_bbox[1]+old_bbox[3]/2
               
                #Passed rightmost line AND direction is RIGHT
                if (vehicle.get_direction() == "right" and (bbox_x > (int(width)-begin_x))):
                    to_delete.append(idx)
                    car_counter += 1
                    #Check Lane
                    if bbox_y < right_lane_split:
                        right_upper_lane += 1
                        if (args.get("csv", False)):
                            out_writer.writerow([str(vehicle.get_id()), vehicle.get_first_frame(), frame_counter,
                                                inter_frame_time*vehicle.get_first_frame(), inter_frame_time*frame_counter,
                                                inter_frame_time*frame_counter - inter_frame_time*vehicle.get_first_frame()])
                    else:
                        right_lower_lane += 1
                        if (args.get("csv", False)):
                            out_writer.writerow([str(vehicle.get_id()), vehicle.get_first_frame(), frame_counter,
                                                inter_frame_time*vehicle.get_first_frame(), inter_frame_time*frame_counter,
                                                inter_frame_time*frame_counter - inter_frame_time*vehicle.get_first_frame()])
                #Passed leftmost line AND direction is LEFT                            
                elif (vehicle.get_direction() == "left" and (bbox_x < begin_x)):
                    to_delete.append(idx)
                    car_counter += 1
                    #Check Lane
                    if bbox_y < left_lane_split:
                        print("id: ", vehicle.get_id(), " point: ", bbox_x, " ", bbox_y)
                        left_upper_lane += 1
                        if (args.get("csv", False)):
                            out_writer.writerow([vehicle.get_id(), vehicle.get_first_frame, frame_counter,
                                                inter_frame_time*vehicle.get_first_frame(), inter_frame_time*frame_counter,
                                                inter_frame_time*frame_counter - inter_frame_time*vehicle.get_first_frame()])
                    else:
                        left_lower_lane += 1
                        if (args.get("csv", False)):
                            out_writer.writerow([str(vehicle.get_id()), vehicle.get_first_frame(), frame_counter,
                                                inter_frame_time*vehicle.get_first_frame(), inter_frame_time*frame_counter,
                                                inter_frame_time*frame_counter - inter_frame_time*vehicle.get_first_frame()])
                        
                #If not moving
                elif (math.sqrt((bbox_x-old_bbox_x)**2+(bbox_y-old_bbox_y)**2) < 0.5):
                    #remove the tracker, not moving
                    #Also check that there is no blob inside
                    blob_detected = False
                    for blob in keypoints:
                        if (math.sqrt((bbox_x-blob.pt[0])**2+(bbox_y-blob.pt[1])**2) < 20):
                            blob_detected = True
                            
                    if not blob_detected:
                        to_delete.append(idx)
                else:
                    #Save tracker bbox center position
                    vehicle.set_old_bbox = bbox
                    
                #Update displayed tracker positions
                (x, y, w, h) = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, str(vehicle.get_id()), (x+w,y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, lineType=cv2.LINE_AA)
            
            
            
            to_delete.sort(reverse=True)
            for idx in range(0, len(to_delete)):
                del vehicle_list[to_delete[idx]]
                
                
            #Add TRACKER
            if (len(keypoints) > 0):
                for blob in keypoints:
                    p1 = blob.pt
                    
                    #wait for the bb to be inside tracked area
                    #Avoid double duplicating trackers
                    if ((p1[0]< begin_x and p1[1]>way_split) or 
                       (p1[0]> begin_x+track_size and p1[1]>way_split) or
                       (p1[0]> width - begin_x and p1[1]<= way_split) or
                       (p1[0]< width - begin_x - track_size and p1[1]<=way_split)):
                        continue

                    found = False
                    
                    #Check trackers first
                    for vehicle in vehicle_list:
                        track_box = vehicle.get_bbox()
                        track_box_center = [int(track_box[0]+track_box[2]/2), int(track_box[1]+track_box[3]/2)]
                        cv2.rectangle(frame, (track_box_center[0], track_box_center[1]), (track_box_center[0], track_box_center[1]), (100, 255, 0), 2)
                        box_pt = (track_box[0]+track_box[2]/2 ,track_box[1]+track_box[3]/2)
                        if (math.sqrt((p1[0]-box_pt[0])**2+(p1[1]-box_pt[1])**2) < max_blob_dist): 
                            found = True
            
                    #Check blobs
                    if (not found):
                        size = blob.size/1.5
                        box = (blob.pt[0]-size, blob.pt[1]-size, 2*size, 2*size)
                        print("box: ", box)
                        cv2.rectangle(frame, (int(blob.pt[0]), int(blob.pt[1])), (int(blob.pt[0]+2), int(blob.pt[1])+2), (0, 255, 0), 2)
                        
                        # create a new object tracker for the bounding box and add it
                        # to our multi-object tracker
                        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                        tracker.init(frame, box)
                        
                        if (p1[0] > width/2):
                            vehicle_list.append(Vehicle(id, tracker, "left", frame_counter))
                            print("id: ", id, " dir: left")
                        elif (p1[0] < width/2):
                            print("id: ", id, " dir: right")
                            vehicle_list.append(Vehicle(id, tracker, "right", frame_counter))
                        
                        id += 1
                        
                    else:
                        cv2.rectangle(frame, (int(blob.pt[0]), int(blob.pt[1])), (int(blob.pt[0]+2), int(blob.pt[1])+2), (0, 0, 255), 2)
            
            
            #Visualization
            #Vertical Lines
            cv2.line(frame, (begin_x,0), (begin_x, int(height)), (255,0,0), 3)
            cv2.line(frame, (int(width)-begin_x, 0), (int(width)-begin_x, int(height)), (255,0,0), 3)
            #Horizontal Line - split lane
            cv2.line(frame, (0, right_lane_split), (int(width), right_lane_split), (255,0,0), 2)
            cv2.line(frame, (0, left_lane_split), (int(width), left_lane_split), (255,0,0), 2)
            cv2.line(frame, (0, way_split), (int(width), way_split), (0,0,255), 2)
            
            #Print total count
            cv2.putText(frame, "Total Cars: "+str(car_counter), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "Left Direction Upper Lane: "+str(left_upper_lane),  (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)                           
            cv2.putText(frame, "Left Direction Lower Lane: "+str(left_lower_lane),  (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)
            cv2.putText(frame, "Right Direction Upper Lane: "+str(right_upper_lane),  (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)                           
            cv2.putText(frame, "Right Direction Lower Lane: "+str(right_lower_lane),  (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, lineType=cv2.LINE_AA)
            
            #record video
            if (args.get("record", False)):
                out.write(frame)
            
            #Show some output
            cv2.imshow('Contour', contour_frame)
            cv2.imshow('Output Image', frame)
            
        else: #End of stream
            break

    k = cv2.waitKey(30) & 0xff
    if k == 27 or k == ord("q"):
        break
    if k == ord(" "):
        paused = not paused
    if k == 13:
        paused = True
        step = True
            
cap.release()
if (args.get("record", False)):
    out.release()
cv2.destroyAllWindows()
                    