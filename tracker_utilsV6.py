import cv2
import numpy as np
import mediapipe as mp
import time

L_newTrk_flag = False
L_refPt = (-1,-1)

R_newTrk_flag = False
R_refPt = (-1,-1)

HALF_RECT_CROP_SIZE = [100,150]
CROSS_SZ = 4

OVERLAY_COLORS = [(180,105,250),(0,200,255),(255,255,224)]

#Pose Detector#
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

TRACKER_DRIFT_THRESHOLD = 5 #how much must track drift from correction trk point in either X or Y pixs before we correct it#


def click_and_crop(event, x, y, flags, param):
	global L_refPt, L_newTrk_flag, R_refPt, R_newTrk_flag
	if event == cv2.EVENT_LBUTTONDOWN:
		L_refPt = (x, y)
		print("New track for left click")
		L_newTrk_flag = True
	elif event == cv2.EVENT_LBUTTONUP:
		L_newTrk_flag = False
	if event == cv2.EVENT_RBUTTONDOWN:
		R_refPt = (x, y)
		R_newTrk_flag = True
	elif event == cv2.EVENT_RBUTTONUP:
		R_newTrk_flag = False


def check_n_ret_ROI(pt_x,pt_y, image_shape,cropWH=(HALF_RECT_CROP_SIZE[0],HALF_RECT_CROP_SIZE[1])): #prevent out of bounds trk points#
	img_ht = image_shape[0]
	img_wid = image_shape[1]
	newX,newY = pt_x, pt_y
	if pt_x+cropWH[0]>= img_wid:
		newX = img_wid -1-cropWH[0]
	if pt_y + cropWH[1] >= img_ht:
		newY = img_ht - 1 - cropWH[1]
	if pt_x-cropWH[0]<= 0:
		newX = cropWH[0] + 1
	if pt_y-cropWH[1]<= 0:
		newY = cropWH[1] + 1
	return (newX,newY)

def cropPatch(raw_img,pt_x,pt_y,cropWH=(HALF_RECT_CROP_SIZE[0],HALF_RECT_CROP_SIZE[1])): #for viz on what the tracker is tracking#
	crop = raw_img[(pt_y - cropWH[1]):(pt_y + cropWH[1]), (pt_x - cropWH[0]):(pt_x + cropWH[0])].copy()
	return crop

def drawRect(img,pt_x,pt_y, correctionpt, overlayidx): #plot#
	pt1 = (pt_x-HALF_RECT_CROP_SIZE[0],pt_y-HALF_RECT_CROP_SIZE[1])
	pt2 = (pt_x + HALF_RECT_CROP_SIZE[0], pt_y + HALF_RECT_CROP_SIZE[1])
	cv2.rectangle(img,pt1,pt2,OVERLAY_COLORS[overlayidx],2)
	cv2.line(img, (pt_x-CROSS_SZ,pt_y), (pt_x+CROSS_SZ,pt_y), OVERLAY_COLORS[overlayidx], 2)
	cv2.line(img, (pt_x,pt_y-CROSS_SZ), (pt_x,pt_y+CROSS_SZ), OVERLAY_COLORS[overlayidx], 2)
	cv2.circle(img, correctionpt, 4, OVERLAY_COLORS[overlayidx], -1)
	return img

def getNearestPoint(centroid_list, currPoint):
	minDist = 1e5 #any very large number#
	bestPoint = (0,0)
	for aimpt in centroid_list:
		dist = abs(aimpt[0]-currPoint[0]) + abs(aimpt[1]-currPoint[1])**2
		if dist<minDist:
			minDist = dist
			bestPoint = aimpt
	return bestPoint

def getFarthestPoint(centroid_list, currPoint):
	maxDist = 0 #any very large number#
	bestPoint = (0,0)
	for aimpt in centroid_list:
		dist = abs(aimpt[0]-currPoint[0]) + abs(aimpt[1]-currPoint[1])**2
		if dist>maxDist:
			maxDist = dist
			bestPoint = aimpt
	return bestPoint

def getBestPoint(centroid_list, currPoint, frame):
	#near + has some intersection#
	minDist = 1e5 #any very large number#
	bestPoint = (0,0)
	trk_x1y1x2y2 = getx1y1x2y2(check_n_ret_ROI(currPoint[0], currPoint[1], frame.shape, \
	cropWH=(HALF_RECT_CROP_SIZE[0],HALF_RECT_CROP_SIZE[1])))
	for aimpt in centroid_list:
		pred_x1y1x2y2 = getx1y1x2y2(check_n_ret_ROI(aimpt[0],aimpt[1], frame.shape,\
		cropWH=(HALF_RECT_CROP_SIZE[0],HALF_RECT_CROP_SIZE[1])))
		iou = get_iou(trk_x1y1x2y2,pred_x1y1x2y2)
		dist = abs(aimpt[0]-currPoint[0]) + abs(aimpt[1]-currPoint[1])**2
		if dist<minDist and iou>0.05:
			minDist = dist
			bestPoint = aimpt
	return bestPoint

def postprocess_YOLO_outputs(layerOutputs, overlayFrame, yolo_confi_thresh, yolo_nms_thresh):
    boxes = []
    confidences = []
    classIDs = []
    centroids = []
    (H, W) = overlayFrame.shape[:2]
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > yolo_confi_thresh and classID==0: #only want "person" classID here which is 0#
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID) 
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, yolo_confi_thresh, yolo_nms_thresh)    

    #Ivan: visualise results from: boxes/confidences/classID#
    if len(idxs)>0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centroids.append((x + int(w/2), y + int(h/2))) 
            #patchval = getPatchAvgValue(cleanFrame,[(x + int(w/2), y + int(h/2))])
            # draw a bounding box rectangle and label on the image
            cv2.rectangle(overlayFrame, (x, y), (x + w, y + h), (0,255,0), 2)
            text = "{}: {:.1f}%".format(classIDs[i], round(confidences[i]*100,2))
            cv2.putText(overlayFrame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
	#Ivan: output results in a list of tuples: [(x0,y0), (x1,y1)], list is empty if there is no detections#
    return centroids

def getx1y1x2y2(cent_pt):
	return [int(cent_pt[0]-HALF_RECT_CROP_SIZE[0]/2),int(cent_pt[1]-HALF_RECT_CROP_SIZE[1]/2),\
		int(cent_pt[0]+HALF_RECT_CROP_SIZE[0]/2),int(cent_pt[1]+HALF_RECT_CROP_SIZE[1]/2)]

#Ivan: 29Oct, added IOU to associate predicted box with tracker#
def get_iou(trackerbbox, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(trackerbbox[0], pred[0])
    iy1 = np.maximum(trackerbbox[1], pred[1])
    ix2 = np.minimum(trackerbbox[2], pred[2])
    iy2 = np.minimum(trackerbbox[3], pred[3])
    
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
    
    area_of_intersection = i_height * i_width
    
    # Tracker dimensions.
    gt_height = trackerbbox[3] - trackerbbox[1] + 1
    gt_width = trackerbbox[2] - trackerbbox[0] + 1
    
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
    
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
    
    iou = area_of_intersection / area_of_union
    
    return iou


class tracker:
	def __init__(self, trackerInstance,yolofiles): #use this function to reset tracker#
		self.tracker_name = str(trackerInstance)
		self.trkStatus = False
		self.trkState = 0  #no track,   new track,   tracking,   lost track --> 0,1,2,3
		self.bbox = (-1,-1,0,0) #topleft_x,topleft_y,w,h, format for opencv_tracking API
		self.trk = cv2.legacy_TrackerCSRT.create()#cv2.legacy_TrackerMOSSE.create() #cv2.legacy_TrackerCSRT.create()
		self.trackercrcm = 0.0
		self.curr_trk_point = (-1,-1)
		self.correction_point = (0,0) #point to correct tracker XY if drift or lost track#
		self.posedetector = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.2, min_tracking_confidence=0.4)
		self.yolo_centroid_results = []

		#Ivan: 5 Sept, added YOLO detector#
		self.net = cv2.dnn.readNet(yolofiles[0], yolofiles[1], "darknet")
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
		self.outNames = self.net.getUnconnectedOutLayersNames()
		self.yolo_confi_thresh = 0.25 #detection confidence#
		self.yolo_nms_thresh = 0.5 #nms is related to box resolution (filter out detection boxes that have too much overlap)#
		self.yolo_input_size = 416 	#Note, change 416x416 to any resolution that is of multiple of 32 (i.e. 320x320 or 288x288) for faster performance but poorer accuracy#
		self.swapTargetFlag = False #To swap to farthest target. Once succeed, it will revert to False#

	def initTrk(self, click_frame, trkX, trkY):# run this on click
		self.trk = cv2.legacy_TrackerCSRT.create()# cv2.legacy_TrackerMOSSE.create()#cv2.legacy_TrackerCSRT.create()
		self.bbox = (trkX - HALF_RECT_CROP_SIZE[0] / 2, trkY - HALF_RECT_CROP_SIZE[1] / 2, HALF_RECT_CROP_SIZE[0], HALF_RECT_CROP_SIZE[1])
		self.curr_trk_point = (trkX,trkY)
		self.trk.init(click_frame, self.bbox)
		self.trkState = 1
		self.correction_point = (0,0)
		#print("Initialized track for: {}\n".format(self.tracker_name))

	def updateTrk(self,frame):
		self.trkStatus, self.bbox = self.trk.update(frame)
		if self.trkStatus: #if track goes well, update curr trk point
			#Ivan: new correction, if any#
			if self.correction_point [0] != 0 and self.correction_point [1] != 0:
				bboxlist= list(self.bbox)
				if (abs(self.correction_point[0]-bboxlist[0])>TRACKER_DRIFT_THRESHOLD \
					or abs(self.correction_point[1]-bboxlist[1])>TRACKER_DRIFT_THRESHOLD):
					bboxlist[0] = self.correction_point[0]#int(0.01*bboxlist[0] + 0.99*(self.correction_point[0]))
					bboxlist[1] = self.correction_point[1]#int(0.01*bboxlist[1] + 0.99*(self.correction_point[1]))
					trkX, trkY = check_n_ret_ROI(bboxlist[0],bboxlist[1], frame.shape)
					self.initTrk(frame, trkX, trkY) #re-init tracker and update trk pts#

			self.curr_trk_point = check_n_ret_ROI(int(self.bbox[0]) + int(self.bbox[2] / 2), int(self.bbox[1]) + int(self.bbox[3] / 2), frame.shape)
			self.trkState = 2
		else: #if trkStatus == lost track#
			if self.correction_point[0] != 0 and self.correction_point[1] != 0: #try to recover if Detector has any new trk pts#
				trkX, trkY = check_n_ret_ROI(self.correction_point[0], self.correction_point[1], frame.shape)
				self.initTrk(frame, trkX, trkY)
				self.trkState = 2
			else: #no hope, tracker lose track and detector also cannot track#
				self.trkState = 3

	def main_processTrk(self, greyframe, overlayframe, idx_overlay, refPT, new_trkFlag): #one liner in main program#
		if refPT[0] > 0 and new_trkFlag:
			trkX, trkY = check_n_ret_ROI(refPT[0],  refPT[1],  greyframe.shape)  # correct local trk points so we don't crop out of bounds
			self.initTrk(greyframe, trkX, trkY)

		if self.trkState == 2 or self.trkState == 3:  # tracking or lost track#
			self.updateTrk(greyframe)

		if self.trkState == 2: #still tracking after update
			overlayframe = drawRect(overlayframe, self.curr_trk_point[0], self.curr_trk_point[1], self.correction_point,overlayidx=idx_overlay)  # MT.drawRect(debug_image,trkX,trkY)

			#No reason to see the tracking patch lol, lets save some compute time here!#
			#crop_Patch = cropPatch(overlayframe, self.curr_trk_point[0], self.curr_trk_point[1])
			#crop_Patch_show = cv2.resize(crop_Patch, (150, 150))
			#overlayframe[idx_overlay * 150:idx_overlay * 150 + 150, rgbframe.shape[1] - 150:, :] = crop_Patch_show

		#No reason to see the tracking status? Since detector is reliable alrdy#
		#cv2.putText(overlayframe, "Trk State 0-notrk,1-newtrk,2-trking,3-losttrk: " +
		#						   str(self.trkState) + " XY: (" + str(self.curr_trk_point[0]) + "," + str(self.curr_trk_point[1]) + ")", (5, (idx_overlay + 1) * 35 + 20), 1, 1.5, OVERLAY_COLORS[idx_overlay], 2, cv2.LINE_AA)
		if self.trkState == 1:
			self.trkState = 2  # upgrade new track to tracking for next loop
	
	def DetectorDNNKeypoint(self, frameinput, overlayframe, BODYINDEX=0, DebugPlotAllPose=False):
		frameinputRGB = cv2.cvtColor(frameinput, cv2.COLOR_BGR2RGB)
		results = self.posedetector.process(frameinputRGB)
		if results.pose_landmarks:
			detX = int(results.pose_landmarks.landmark[BODYINDEX].x*frameinput.shape[1])
			detY = int(results.pose_landmarks.landmark[BODYINDEX].y*frameinput.shape[0])
			self.correction_point = (detX,detY)
			if DebugPlotAllPose:
				mp_drawing.draw_landmarks(overlayframe, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
		else:
			self.correction_point = (0,0)
	
	#Ivan: Added 5 Sept#
	def YOLODetector(self, frameinput, overlayframe, infer=True, share_yolo_centroid_results=[]):
		#infer argument decides if we need to run yolo detector again to get newest locations#
		#This is to circumvent running detection twice for each tracker#
		#This improved fps from 8-->12 under extreme case (run detection every frame)#
		# 12 Sept, after meeting there is no 2 tracker scenario, so always leave infer=True, 
		# thus "share_yolo_centroid_results" is useless for now#

		if (infer):
			blob = cv2.dnn.blobFromImage(frameinput, 1 / 255.0, (self.yolo_input_size, self.yolo_input_size), swapRB=False, crop=False)
			self.net.setInput(blob)
			layerOutputs = self.net.forward(self.outNames)
			self.yolo_centroid_results = postprocess_YOLO_outputs(layerOutputs,overlayframe, self.yolo_confi_thresh,self.yolo_nms_thresh)
		else:
			self.yolo_centroid_results = share_yolo_centroid_results
			return
		
		if self.swapTargetFlag and len(self.yolo_centroid_results)>1:
			self.correction_point = getFarthestPoint(self.yolo_centroid_results,self.curr_trk_point)
			self.swapTargetFlag = False
			self.trkPatchVal = -1
			return
			#Get farthest point and end this function call, else update track point on same target#

		#update track point on same target#
		if len(self.yolo_centroid_results)>0:
			#self.correction_point = getNearestPoint(self.yolo_centroid_results,self.curr_trk_point)
			#Ivan 29Oct, use "best" point, has to be near + has some intersection#
			self.correction_point = getBestPoint(self.yolo_centroid_results, self.curr_trk_point, frameinput)
		else:
			self.correction_point = (0,0)
		
	#Ivan: Added 12 Sept#
	def swapTrackViaYOLO(self):
		self.swapTargetFlag = True
		#Not a good idea to directly run YOLO detect on cue, what if it's two consecutive frames of YOLO detect, video will lag...#


		#This function helps to swap track to farthest target#
		#There is a risk that yolo detects nothing, and swap target fails#
		#When that happens it will return False, so in the main program, if detection fails#
		#There may be some lag in the video in the worst case scenario: Call YOLO @  --> fail#
		# blob = cv2.dnn.blobFromImage(frameinput, 1 / 255.0, (self.yolo_input_size, self.yolo_input_size), swapRB=False, crop=False)
		# self.net.setInput(blob)
		# layerOutputs = self.net.forward(self.outNames)
		# self.yolo_centroid_results = postprocess_YOLO_outputs(layerOutputs,frameinput, self.yolo_confi_thresh,self.yolo_nms_thresh)
		# if len(self.yolo_centroid_results)>1: #At least two detections required to reliable get a different far Target#
		# 	self.correction_point = getFarthestPoint(self.yolo_centroid_results,self.curr_trk_point)
		# 	self.swapTargetFlag = False
		# else:
		# 	self.correction_point = (0,0)
		# 	self.swapTargetFlag = True
		# 	#Since swapTargetFlag = True, the next YOLODetector function call will attempt this until it succeeds and change swapTargetFlag to False#

# NOSE = 0
# LEFT_EYE_INNER = 1
# LEFT_EYE = 2
# LEFT_EYE_OUTER = 3
# RIGHT_EYE_INNER = 4
# RIGHT_EYE = 5
# RIGHT_EYE_OUTER = 6
# LEFT_EAR = 7
# RIGHT_EAR = 8
# MOUTH_LEFT = 9
# MOUTH_RIGHT = 10
# LEFT_SHOULDER = 11
# RIGHT_SHOULDER = 12
# LEFT_ELBOW = 13
# RIGHT_ELBOW = 14
# LEFT_WRIST = 15
# RIGHT_WRIST = 16
# LEFT_PINKY = 17
# RIGHT_PINKY = 18
# LEFT_INDEX = 19
# RIGHT_INDEX = 20
# LEFT_THUMB = 21
# RIGHT_THUMB = 22
# LEFT_HIP = 23
# RIGHT_HIP = 24
# LEFT_KNEE = 25
# RIGHT_KNEE = 26
# LEFT_ANKLE = 27
# RIGHT_ANKLE = 28
# LEFT_HEEL = 29
# RIGHT_HEEL = 30
# LEFT_FOOT_INDEX = 31
# RIGHT_FOOT_INDEX = 32


#Ivan: defunct, reference material for smaller search area tracker#
# def trkUpdate(currFrame, prev_trkX, prev_trkY, template):
# 	lowerX = prev_trkX - SEARCH_AREA
# 	if lowerX <=0:
# 		lowerX = 1
# 	lowerY = prev_trkY - SEARCH_AREA
# 	if lowerY <=0:
# 		lowerY = 1
# 	upperX = prev_trkX + SEARCH_AREA
# 	if upperX>= currFrame.shape[1]:
# 		upperX = currFrame.shape[1]-1
# 	upperY = prev_trkY + SEARCH_AREA
# 	if upperY>= currFrame.shape[0]:
# 		upperY = currFrame.shape[0]-1
# 	searchROI = currFrame[lowerY:upperY,lowerX:upperX].copy()
# 	res = cv2.matchTemplate(searchROI, template, cv2.TM_CCORR_NORMED)
# 	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
# 	ok,bbox = Tracker.update(searchROI)
# 	print(ok,bbox)
# 	corr = max_val
# 	if ok:#max_val>0.96:
# 		trkStatus_flag = 2
# 		updatedTrkX = int(bbox[0]) + int(bbox[2] / 2) + lowerX  # int(max_loc[0] + template.shape[1] / 2 + lowerX)
# 		updatedTrkY = int(bbox[1]) + int(bbox[3] / 2) + lowerY  # int(max_loc[1] + template.shape[0] / 2 + lowerY)
# 		newtemplate = template
# 	#elif max_val>0.85: #still good but update template. May be too sticky!
# 		#cropX,cropY = check_n_ret_ROI(updatedTrkX,updatedTrkY,currFrame.shape)
# 		#newtemplate = cropPatch(currFrame,cropX,cropY)
# 	else:#lost track
# 		trkStatus_flag = 3
# 		updatedTrkX = prev_trkX
# 		updatedTrkY = prev_trkY
# 		corr = 0
# 		newtemplate =template
#
# 	return updatedTrkX,updatedTrkY, round(corr*100,2),newtemplate

# #Thresholds for absdiff detector#
# BINARYTHRESHOLD = 70 #if diff>BINARYTHRESHOLD but less than 255#
# CC_AREA_THRESHOLD = 100 #if area of enclosed polygon>CC_AREA_THRESHOLD ? shortlist centroid#
# NUM_OF_SHORLISTED_FEATURE_THRESHOLD = 3 #choose higher mumber for greater confidence#

# 	#Ivan: Added on 15June to use movement to find and correct target#
# 	#Ivan: On 15Sept: Terrible idea! #	
# 	def DetectorAbsDiff(self, currframe, prevframe, debugview=False):
# 		diffimg = cv2.absdiff(currframe,prevframe)
# 		_, diffimg = cv2.threshold(diffimg, BINARYTHRESHOLD, 255, cv2.THRESH_BINARY)
# 		diffimg = cv2.cvtColor(diffimg, cv2.COLOR_BGR2GRAY)
# 		analysis = cv2.connectedComponentsWithStats(diffimg,4,cv2.CV_32S)
# 		output_canvas = np.zeros(diffimg.shape, dtype="uint8")
# 		xlist = []
# 		ylist = []
# 		pt1_debugview = []
# 		pt2_debugview = []

# 		(totalLabels, label_ids, values, centroid) = analysis
# 		for i in range(1, totalLabels):
# 		# Area of the component
# 			area = values[i, cv2.CC_STAT_AREA]
# 			if (area > CC_AREA_THRESHOLD):
# 				componentMask = (label_ids == i).astype("uint8") * 255
# 				(X, Y) = centroid[i]     
# 				xlist.append(X)
# 				ylist.append(Y)
# 				if debugview: 
# 					# Bounding boxes for each component#
# 					x1 = values[i, cv2.CC_STAT_LEFT]
# 					y1 = values[i, cv2.CC_STAT_TOP]
# 					w = values[i, cv2.CC_STAT_WIDTH]
# 					h = values[i, cv2.CC_STAT_HEIGHT]
# 					# Coordinate of the bounding box
# 					pt1_debugview.append((x1, y1))
# 					pt2_debugview.append((x1+ w, y1+ h))	
# 					output_canvas = cv2.bitwise_or(output_canvas, componentMask)
# 		if debugview:
# 			output_view = cv2.merge([output_canvas,output_canvas,output_canvas])
# 			for i in range(len(pt1_debugview)):
# 				cv2.rectangle(output_view, pt1_debugview[i],pt2_debugview[i], (0, 255, 0), 3)
# 			output_view = cv2.addWeighted(output_view,0.7,currframe,0.3,-1)
# 		if len(xlist)>NUM_OF_SHORLISTED_FEATURE_THRESHOLD:
# 			self.correction_point = (int(np.mean(xlist)),int(np.mean(ylist)))
# 			if debugview:
# 				cv2.putText(output_view, "Number of parts shortlisted: " + str(len(xlist)),(5, 20), 1, 1.5, (0,255,255), 2, cv2.LINE_AA)
# 				cv2.circle(output_view, self.correction_point , 6, (0, 0, 255), -1)
# 				cv2.imshow("AbsDiffTracker", output_view)
# 		else:
# 			self.correction_point = (0,0)

# def FlipROI(img,pt_x,pt_y,horizontal = 1): #flips crop img, currently useless#
# 	pt1 = (pt_x-HALF_RECT_CROP_SIZE,pt_y-HALF_RECT_CROP_SIZE)
# 	pt2 = (pt_x + HALF_RECT_CROP_SIZE, pt_y + HALF_RECT_CROP_SIZE)
# 	roi = img[(pt_y-HALF_RECT_CROP_SIZE):(pt_y+HALF_RECT_CROP_SIZE),(pt_x-HALF_RECT_CROP_SIZE):(pt_x+HALF_RECT_CROP_SIZE)]
# 	roi_flipped = cv2.flip(roi, horizontal)
# 	img[(pt_y-HALF_RECT_CROP_SIZE):(pt_y+HALF_RECT_CROP_SIZE),(pt_x-HALF_RECT_CROP_SIZE):(pt_x+HALF_RECT_CROP_SIZE)] = roi_flipped
# 	return img



# class templateMatchingTracker:
# 	def __init__(self): #use this function to reset tracker#
# 		self.trkStatus = False
# 		self.crcm = 0.0
# 		self.bbox = (-1,-1,HALF_RECT_CROP_SIZE[0], HALF_RECT_CROP_SIZE[1])
# 		self.threshold = TmplTrk_Threshold
	
# 	def init(self,frame, bbox):
# 		centX = int(bbox[0]+bbox[2]/2)
# 		centY = int(bbox[1]+bbox[3]/2)
# 		self.template = cropPatch(frame,centX,centY,cropWH=(bbox[2],bbox[3]))
# 		self.bbox = bbox
# 		self.crcm = 0.0
# 		self.trkStatus = True

# 	def update(self, frame):
# 		res = cv2.matchTemplate(frame, self.template, cv2.TM_CCORR_NORMED)
# 		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# 		self.bbox = (max_loc[0],max_loc[1],HALF_RECT_CROP_SIZE[0], HALF_RECT_CROP_SIZE[1])
# 		self.crcm = max_val
# 		if self.crcm>self.threshold:
# 			self.trkStatus = True
# 		else:
# 			self.trkStatus = False
# 		return self.trkStatus, self.bbox, self.crcm



# def getPatchAvgValue(img,centroids,WH=(HALF_RECT_CROP_SIZE[0],HALF_RECT_CROP_SIZE[1])):
# 	for xy in centroids:
# 		X, Y = check_n_ret_ROI(xy[0], xy[1], img.shape,cropWH=WH)
# 		patch = cropPatch(img,X,Y,cropWH=WH)
# 		avg = np.mean(patch, axis=(0,1))
# 		avg = np.mean(avg)
# 	return avg