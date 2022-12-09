import cv2
import numpy as np
import time
import random
import tracker_utilsV6 as TU
from imutils.video import WebcamVideoStream
from imutils import resize

# import ThreadingVid as TV

# from ThreadingVid import resize

'''
Fix Brushstroke multi
Make the AM in Line more obvious
'''
Enigma_Fullscreen = True
'-------------------------States-------------------------'
AM_BPM = 0
AM_On_Click = 1
AM_forever = 2
Brushstroke = 3
Brushstroke_multi = 4
Line = 5
AM_Faded = 6
Nothing = 9
toggle_title = ["AM", "AM on click", "AM Forever", "Brush", "Brush_multi", "Line", "AM_Faded", "state_6", "state_7",
                "Nothing"]
state_count = 8
'----------------------- Constants-----------------------'

'---video-related---'
vidfontFace = cv2.FONT_HERSHEY_COMPLEX
video_capture = WebcamVideoStream(src=0).start()
# video_capture = WebcamVideoStream(src=0).start()

cam_res = (int(960), int(720))
col_conv = cv2.COLOR_BGR2RGB
col_conv_list = np.array([cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV], dtype=np.uint8)
refresh_beat_interval = 2
afterimage_count = 8  # Has to be >1 (Product of refresh r8 <15 qfor less lag for R-Pi4)
AM_weight = 0.6
time_to_wait = 3  # in seconds
movement_threshold = cam_res[0] * cam_res[1] * 1.50  # ranges from 1-5 million
brightness_threshold = 100
brushstroke_points = 20
brushstroke_maxsize = 20
brushstroke_colours = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 0)]
brushstroke_period = 1. / 15.
flow_interval_frame = 2
brushframe = 0
DETECT_RATE = 3
brushframe_multi = 0
DETECT_RATE_MULTI = 10
DET_ON_MULTI = False
brushstroke_multi_Y_offset = -20
brushstroke_multi_Y_offset_max = -brushstroke_multi_Y_offset+20
diverge_max_frame = 200
Body_Index_strings = [
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
    "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX"
]
#Body_Index_list = [15, 2, 12, 13, 2]  # [L. Wrist, Head,L shoulder, R. Shoulder, Head]
Body_Index_list = [16,15, 2, 11, 12, 2]  # [L. Wrist, Head,L shoulder, R. Shoulder, Head]
Body_Index_num = 0
mmf_threshold = 16  # no more than 32
bump_diff_hold_timer = 0.14
faded_bump_diff_hold_timer = bump_diff_hold_timer / 4.
forever_bump_diff_hold_timer = bump_diff_hold_timer * 2.
mytimer = bump_diff_hold_timer
no_click_pos = (-1, -1)
px_colour = (0, 0, 0)
prev_px_colour = (0, 0, 0)
bright_unit_vector = np.array([1, 1, 1])
bright_unit_vector = bright_unit_vector / np.linalg.norm(bright_unit_vector)
'---colourstate-related---'
# forever_frame = np.zeros((cam_res[1],cam_res[0],3),dtype=np.uint8)
# low_mask = np.zeros(3,dtype=np.uint8)
# high_mask = np.zeros(3,dtype=np.uint8)
colst_fade_factor = 0.8
low_mask_bound = 30
high_mask_bound = 255
colourstate_normal = 99
colourstate_red = 92
colst_red_mul = np.array([0., 0., 255.]) / 255
colourstate_green = 91
colst_green_mul = np.array([0., 255., 0.]) / 255
colourstate_blue = 90
colst_blue_mul = np.array([255., 0., 0.]) / 255
colourstate_skyblu = 93
colst_skyblu_mul = np.array([255., 165., 0.]) / 255
colourstate_yellow = 94
colst_yellow_mul = np.array([0., 255., 255.]) / 255
colourstate_darkviolet = 95
colst_darkviolet_mul = np.array([148., 0., 211.]) / 255
colst_random_mul_list = [colst_red_mul, colst_green_mul, colst_blue_mul, colst_skyblu_mul, colst_yellow_mul,
                         colst_darkviolet_mul]
colourstate_random = 96
rand_for_colst_list = np.arange(0, len(colst_random_mul_list), dtype=np.uint8)
rand_for_colst = random.choice(rand_for_colst_list)
colst_random_mul = colst_random_mul_list[rand_for_colst]  # np.random.rand(3)

'---GPIO-related---'
# switchPin = 8
# JoystickPin = 12
line_maxThick = 400
line_maxColor = 255
line_maxangle = np.pi
max_mag = 20
'''
Joystick_X_mid = 121
Joystick_X_max = 254
Joystick_Y_mid = 123
Joystick_Y_max = 254
Potentiometer_max = 253
ADCPinColor = 1
ADCPinThickness = 2
ADCPinAngle = 3
ADCPinJoystickY = 4
ADCPinJoystickX = 5
'''
Y_move_mag = int(cam_res[0] / 4 / 30)
X_move_mag = Y_move_mag
counts_of_8 = 4
max_beat = counts_of_8 * 8

keys = [AM_BPM, AM_On_Click, AM_forever, colourstate_normal,  # key code
        Brushstroke, Line, 5, colourstate_red,
        6, 7, 8, colourstate_blue,
        Nothing, 10, 11, colourstate_green]

'---text-related---'
file_1 = open("mytext.txt", "r")

longstr = file_1.read()

word_list = longstr.split()
word_list_len = len(word_list)
textfontScale = 2
textfontThick = 4
textfontColour = (0, 0, 0)
text_time_trigger = 0.1

'------------------------------ Setup----------------------------------'

'----------video-related---------------'
frame_count = afterimage_count + 1
afterimage_range = np.arange(0, afterimage_count, dtype=np.uint8)
placeholder_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
All_AM_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
All_AMF_weight = .8 - (afterimage_count * 0.02)
sum_frames_weight = 0.4
combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
combine_frame_16 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint16)
control_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
show_combine_frame = False
show_combine_window = False
# diverge_frame_saved = np.copy(combine_frame)
# diverge_frame_load = np.copy(combine_frame)
combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
diverge_frame_saved = np.zeros((diverge_max_frame + 1, cam_res[1], cam_res[0], 3), dtype=np.uint8)
diverge_frame_load = np.zeros((diverge_max_frame + 1, cam_res[1], cam_res[0], 3), dtype=np.uint8)
frame_to_load = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
converge_frame_bool = False
empty_frame_bool = False
converge_frame_counter = 0
converge_to_keep = 1
save_diverge_frame = False
save_diverge_frame_prev = False
diverge_save_trail = False
mystop = 0
div_frame_partial_idx = 0
toggle_mystop = True
load_diverge_frame = False
div_load_bool = False
div_timer_before = time.time()
div_timer_after = time.time()
diverge_frame_time_interval = bump_diff_hold_timer / 2.5
load_diverge_frame_timer_after = time.time()
load_diverge_frame_timer_before = time.time()
save_diverge_count = 0
load_diverge_count = 0
load_diverge_maxframes = 0
disp_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)

diff_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)
diff_frames_2 = np.zeros((afterimage_count - 1, cam_res[1], cam_res[0], 3), dtype=np.uint8)
faded_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)
oldest_diff_frame_idx = 0
AM_press_count = 0
forever_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
sum_frames = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
conv_cycle = np.shape(col_conv_list)[0]
# video_capture.set(3, cam_res[0])
# video_capture.set(4, cam_res[1])


text_position = (5, 50)

current_image = video_capture.read()  # comment_2#
current_image = resize(current_image, width=cam_res[0], height=cam_res[1])
# ret, current_image = video_capture.read()


brushstroke_colorcycle = [(0, 0, 0)] * len(brushstroke_colours)
colorcycle_counter = 0
col_cy_fact = int(len(brushstroke_colours))
n_factor = int(brushstroke_points / col_cy_fact)
offset_max = brushstroke_points - 1
offset = offset_max
flow_tick = 0
Body_Index = Body_Index_list[0]
refresh_trigger = time.time()
first_state_click = True

colst_red_mul_broadcast = np.broadcast_to(colst_red_mul, (cam_res[1], cam_res[0], 3)) * colst_fade_factor
colst_green_mul_broadcast = np.broadcast_to(colst_green_mul, (cam_res[1], cam_res[0], 3)) * colst_fade_factor
colst_blue_mul_broadcast = np.broadcast_to(colst_blue_mul, (cam_res[1], cam_res[0], 3)) * colst_fade_factor
colst_skyblu_mul_broadcast = np.broadcast_to(colst_skyblu_mul, (cam_res[1], cam_res[0], 3)) * colst_fade_factor
colst_yellow_mul_broadcast = np.broadcast_to(colst_yellow_mul, (cam_res[1], cam_res[0], 3)) * colst_fade_factor
colst_darkviolet_mul_broadcast = np.broadcast_to(colst_darkviolet_mul, (cam_res[1], cam_res[0], 3)) * colst_fade_factor
colst_random_mul_broadcast_list = [colst_red_mul_broadcast, colst_green_mul_broadcast, colst_blue_mul_broadcast, \
                                   colst_skyblu_mul_broadcast, colst_yellow_mul_broadcast,
                                   colst_darkviolet_mul_broadcast]
colst_random_mul_broadcast = random.choice(
    colst_random_mul_broadcast_list)  # np.broadcast_to(colst_random_mul,(cam_res[1],cam_res[0],3))*colst_fade_factor
low_mask = np.full(3, low_mask_bound, dtype=np.uint8)
high_mask = np.full(3, high_mask_bound, dtype=np.uint8)

'---Line related---'
lineMag = np.linalg.norm(np.array(cam_res)) * 1.05
lineMidpoint_initial = np.array(cam_res) / 2.
line_thickness_initial = int(cam_res[0] / 5)
line_thickness_mag_initial = int(cam_res[0] / 45)
line_angle_initial = np.pi / 2
line_angle_mag_initial = np.pi / 16
line_change_angle_bool = False
line_change_mag_bool = False
line_center = np.array([0, 0], dtype=np.uint16)
line_time_before = time.time()
line_time_after = time.time()
line_time_interval = 1. / 30.

'---Text-related---'
text_counter = 0
text_line_counter = 0
show_text = False
show_textlines = False
text_time_before = time.time()
text_time_after = time.time()
text_time_passed = 0.

'---save video-related---'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
videofile = cv2.VideoWriter("./Record/save_vid.mp4", fourcc, 20, cam_res)

'------Function(s)------'


def return_smaller(number1, number2):
    if number1 >= number2:
        return number2
    else:
        return number1


def return_midpoint(diff_grey, brightness_threshold, midpoint_prev):
    midpoint_new = np.zeros(2, dtype=np.uint16)
    midpoint_sum = np.where(diff_grey > brightness_threshold)
    if not np.shape(midpoint_sum)[1]:
        midpoint_new[0] = midpoint_prev[0]
        midpoint_new[1] = midpoint_prev[1]
    else:
        midpoint_new[0] = int(np.mean(midpoint_sum[1]))
        midpoint_new[1] = int(np.mean(midpoint_sum[0]))
    return midpoint_new


def create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_mul, colst_mul_broadcast):
    global control_frame
    placeholder_frame = cv2.cvtColor(diff_grey, cv2.COLOR_GRAY2RGB)
    placeholder_frame = placeholder_frame * colst_mul_broadcast
    low_mask = np.multiply(low_mask, colst_mul).astype(np.uint8)
    high_mask = np.multiply(high_mask, colst_mul).astype(np.uint8)
    mask = cv2.inRange(placeholder_frame, low_mask, high_mask)
    control_frame = np.copy(placeholder_frame)
    sum_frames[mask == 255] = placeholder_frame[mask == 255]
    return sum_frames


def convert_to_colour(movement_meas_frame, diff_grey, colourstate, sum_frames):
    if colourstate == colourstate_normal:
        forever_frame = movement_meas_frame
        sum_frames = cv2.addWeighted(sum_frames, 1.0, forever_frame, sum_frames_weight, 0)
    else:
        global placeholder_frame, low_mask, high_mask  # global (variables) is used to reference/change global variables' values
        if colourstate == colourstate_red:
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_red_mul,
                                                       colst_red_mul_broadcast)
        if colourstate == colourstate_blue:
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_blue_mul,
                                                       colst_blue_mul_broadcast)
        if colourstate == colourstate_green:
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_green_mul,
                                                       colst_green_mul_broadcast)
        if colourstate == colourstate_skyblu:
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_skyblu_mul,
                                                       colst_skyblu_mul_broadcast)
        if colourstate == colourstate_yellow:
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_yellow_mul,
                                                       colst_yellow_mul_broadcast)
        if colourstate == colourstate_darkviolet:
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_darkviolet_mul,
                                                       colst_darkviolet_mul_broadcast)
        if colourstate == colourstate_random:
            global colst_random_mul, colst_random_mul_broadcast, colst_random_mul_list, colst_random_mul_broadcast_list, rand_for_colst_list
            sum_frames = create_coloured_forever_frame(placeholder_frame, low_mask, high_mask, colst_random_mul,
                                                       colst_random_mul_broadcast)
            rand_for_colst = random.choice(rand_for_colst_list)
            colst_random_mul = colst_random_mul_list[rand_for_colst]
            colst_random_mul_broadcast = colst_random_mul_broadcast_list[
                rand_for_colst]
    return sum_frames


def get_bpm_period(bpm_array):
    del_t_array = bpm_array[1:] - bpm_array[:-1]
    bpm_period = np.mean(del_t_array)
    return bpm_period


def calc_LineEnds(lineMag, lineMidpoint, theta):
    x_mid, y_mid = lineMidpoint
    L0 = np.array([x_mid - 0.5 * lineMag * np.cos(theta), y_mid - 0.5 * lineMag * np.sin(theta)])
    L1 = L0 + np.array([lineMag * np.cos(theta), lineMag * np.sin(theta)])
    L0 = np.rint(L0)
    L1 = np.rint(L1)
    return L0, L1


def get_brushstroke_factor(brushstroke_maxsize, brushstroke_points):
    brushstroke_factor = int(np.rint(brushstroke_maxsize / brushstroke_points))
    if brushstroke_factor == 0:
        return 1
    else:
        return brushstroke_factor


def store_new_frame(stored_frame_0, stored_frame_1, stored_frame_2, current_image, first_state_click,
                    movement_meas_frame, clean_image=False):
    if first_state_click:
        stored_frame_0 = np.copy(current_image)
        stored_frame_1 = np.copy(current_image)
        stored_frame_2 = np.copy(current_image)
        first_state_click = False
    stored_frame_2 = np.copy(stored_frame_1)
    stored_frame_1 = np.copy(stored_frame_0)
    stored_frame_0 = np.copy(current_image)
    # movement_meas_frame = cv2.absdiff(current_image, stored_frame_1) ORIGINAL HERE
    stored_absdiff_0to1 = cv2.absdiff(current_image, stored_frame_1)
    stored_absdiff_1to2 = cv2.absdiff(stored_frame_1, stored_frame_2)
    stored_absdiff_0to1 = np.asarray(stored_absdiff_0to1, dtype=np.uint16)
    stored_absdiff_1to2 = np.asarray(stored_absdiff_1to2, dtype=np.uint16)
    # stored_absdiff_1to2 = cv2.cvtColor(stored_absdiff_1to2, cv2.COLOR_BGR2GRAY)
    movement_meas_frame = cv2.addWeighted(stored_absdiff_0to1, 1.0, stored_absdiff_1to2, -1.0, 0)
    movement_meas_frame = cv2.convertScaleAbs(movement_meas_frame)
    # movement_meas_frame = cv2.bitwise_not(stored_absdiff_0to1,stored_absdiff_0to1,mask=mymask)
    diff_grey = cv2.cvtColor(movement_meas_frame, cv2.COLOR_BGR2GRAY)
    if clean_image:
        ret1, threshold_frame = cv2.threshold(diff_grey, mmf_threshold, 255, cv2.THRESH_BINARY)
        movement_meas_frame = cv2.bitwise_and(movement_meas_frame, movement_meas_frame, mask=threshold_frame)
    # th3 = cv2.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,2)
    move_thresh = np.sum(diff_grey)
    return stored_frame_0, stored_frame_1, stored_frame_2, movement_meas_frame, diff_grey, move_thresh, first_state_click


def get_new_diff(diff_frames):
    for m in np.arange(afterimage_count - 1):
        diff_frames_2[m] = cv2.bitwise_and(diff_frames[m], diff_frames[m + 1])
    return diff_frames_2


def get_colandbright_change(prev_px_colour, px_colour):
    vector_chg = px_colour - prev_px_colour
    vector_chg = vector_chg / np.linalg.norm(vector_chg)
    # bright_chg = np.arccos(np.dot(prev_px_colour,px_colour)/np.linalg.norm(prev_px_colour)/np.linalg.norm(px_colour))
    bright_chg = np.arccos(
        np.dot(vector_chg, bright_unit_vector) / np.linalg.norm(vector_chg) / np.linalg.norm(bright_unit_vector))
    col_chg = np.sqrt((prev_px_colour[0] - px_colour[0]) ^ 2 + (prev_px_colour[1] - px_colour[1]) ^ 2 + (
                prev_px_colour[2] - px_colour[2]) ^ 2)
    return col_chg, bright_chg


'------Initialising------'
test_toggle = True
i = 0
j = 0
bump_diff = 0
bump_diff_hold = False
bump_diff_hold_before = time.time()
bump_diff_hold_after = time.time()
bump_diff_hold_passed = 0.
n = 0
midpoint_sum = 0
midpoint = np.full((brushstroke_points, 2), (20000),
                   dtype=np.uint16)  # np.zeros((brushstroke_points, 2), dtype=np.uint32)
midpoint_new = np.full(2, 20000, dtype=np.uint16)
show_line_diff_AM = False
AM_click = 0
colourstate_prev = colourstate_random
colourstate = colourstate_random

brushstroke_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
line_diff_AM = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
line_diff_last_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
mask_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
stored_frame_0 = np.copy(current_image)
stored_frame_1 = np.copy(current_image)
stored_frame_2 = np.copy(current_image)
stored_absdiff_0to1 = np.copy(current_image)
stored_absdiff_1to2 = np.copy(current_image)
brushstroke_factor = get_brushstroke_factor(brushstroke_maxsize, brushstroke_points)
brush_threshold = np.linalg.norm(cam_res) / 2
cv2.namedWindow("Control Frame")
cv2.setMouseCallback("Control Frame", TU.click_and_crop)
framenum = 0
# left_click_tracker = TU.tracker("left_click") using v4
left_click_tracker = TU.tracker("left_click", ("yolov4-tiny-3l-DanceTrack.weights", "yolov4-tiny-3l-DanceTrack.cfg"))

diff = np.copy(current_image)
movement_meas_frame = np.copy(current_image)
move_thresh = 0
time_passed = 0
diff_grey = np.zeros((cam_res[1], cam_res[0]), dtype=np.uint8)
threshold_frame = np.zeros((cam_res[1], cam_res[0]), dtype=np.uint8)

# ADCsetup()
# button_checker = np.ones(2,dtype=np.uint8)
# button_toggle_array = np.array([1,0],dtype=np.uint8)
state = 9
state_prev = state
beat_count = 0
beats_passed = 0
bpm_array = np.zeros(max_beat)
del_t_array = np.zeros(max_beat - 1)
bpm_period = 0

last_image = np.zeros((cam_res[1], cam_res[0]), dtype=np.uint8)
last_image_2 = np.zeros((cam_res[1], cam_res[0]), dtype=np.uint8)
last_image_3 = np.zeros((cam_res[1], cam_res[0]), dtype=np.uint8)
last_image_4 = np.zeros((cam_res[1], cam_res[0]), dtype=np.uint8)
midpoint_X_move = 0
midpoint_Y_move = 0
lineMidpoint = np.copy(lineMidpoint_initial)
line_change_angle = line_angle_mag_initial
line_thickness = line_thickness_initial
line_thickness_mag = line_thickness_mag_initial
line_change_mag = line_thickness_mag
line_angle = line_angle_initial
line_angle_mag = line_angle_mag_initial
line_center_y = line_center[1]
line_center_x = line_center[0]
'------Loop(s)------'

'''
print('Program start! Looking for BPM...')
for i in np.arange(0, max_beat):
    if not i:
        keypress = input("Waiting for %d x %d count..." % (1, 1))
    else:
        keypress = input("Waiting for %d x %d count..." % (beat_count / 8 + 1, beat_count % 8 + 1))
    if keypress == "":
        bpm_array[i] = time.time()
        beat_count += 1
'''
# bpm_period = get_bpm_period(bpm_array)
bpm_period = 0.1
refresh_timer = time.time()
refresh_trigger = refresh_timer + bpm_period
refresh_state = np.array([bpm_period, 0., 0., brushstroke_period])
adj_bright_timer = 0.5
adj_bright_start = time.time()
adj_bool = True
saveFrame = False
while True:

    last_image_4 = np.copy(last_image_3)
    last_image_3 = np.copy(last_image_2)
    last_image_2 = np.copy(last_image)
    last_image = np.copy(current_image)
    current_image = video_capture.read()  # comment_2#
    current_image = resize(current_image, width=cam_res[0], height=cam_res[1])
    framenum += 1

    # if not ret:
    #    break
    if adj_bool:
        if time.time() - adj_bright_start < adj_bright_timer:
            stored_frame_2 = np.copy(current_image)
            stored_frame_1 = np.copy(current_image)
            stored_frame_0 = np.copy(current_image)

        else:
            print("Light Adjustment Complete")
            adj_bool = False

    if state_prev != state:
        if state == AM_forever:
            mytimer = forever_bump_diff_hold_timer
        elif state == AM_Faded:
            mytimer = faded_bump_diff_hold_timer
        else:
            mytimer = bump_diff_hold_timer
        control_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
        print("State Changed to %s" % (toggle_title[state]))
        if state_prev == AM_BPM or state_prev == AM_On_Click or state_prev == AM_forever or state_prev == AM_Faded or \
                state == AM_BPM or state == AM_On_Click or state == AM_forever or state == AM_Faded:
            AM_press_count = 0
            first_state_click = True
            bump_diff = 0
            bump_diff_hold = False
            stored_frame_0 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            stored_frame_1 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            stored_frame_2 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            stored_absdiff_0to1 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            stored_absdiff_1to2 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            if not state_prev == AM_Faded and not state == AM_Faded:
                diff = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                movement_meas_frame = current_image
            if state_prev == AM_Faded or state == AM_Faded:
                faded_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)

            if state_prev == AM_forever or state == AM_forever:
                sum_frames = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                forever_frame = np.copy(placeholder_frame)

            if state_prev == AM_BPM or state_prev == AM_On_Click or \
                    state == AM_BPM or state == AM_On_Click:
                diff_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)

                # diff_frames_2 = get_new_diff(diff_frames)
                combine_frame_16 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint16)
                combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                All_AM_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
        if state == Brushstroke or state == Brushstroke_multi:
            if not (state_prev == Brushstroke or state_prev == Brushstroke_multi):
                midpoint_sum = 0
                brushframe = 0
                brushframe_multi = 0
                midpoint = np.full((brushstroke_points, 2), (20000), dtype=np.uint16)
                midpoint_new = np.full(2, 20000, dtype=np.uint16)
                left_click_tracker.trkState = 0
                left_click_tracker.curr_trk_point = (20000, 20000)
                DET_ON_MULTI = True
            brushstroke_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            mytimer = bump_diff_hold_timer
        if state_prev == Line or state == Line:
            lineMidpoint = np.copy(lineMidpoint_initial)
            line_thickness = line_thickness_initial
            line_angle = line_angle_initial
            line_center_y = line_center[1]
            line_center_x = line_center[0]
            line_change_mag_bool = False
            line_change_angle_bool = False
        state_prev = state
    if colourstate_prev != colourstate:
        forever_frame = placeholder_frame
        colourstate_prev = colourstate

    if state == AM_On_Click or state == AM_forever or AM_Faded:
        if bump_diff_hold:
            bump_diff_hold_after = time.time()
            bump_diff_hold_passed = bump_diff_hold_after - bump_diff_hold_before
            if bump_diff_hold_passed >= mytimer:
                AM_click = 1
                bump_diff_hold_before = time.time()
        if AM_click:
            AM_press_count += 1
            AM_click = 0
            bump_diff = 1
            stored_frame_0, stored_frame_1, stored_frame_2, movement_meas_frame, diff_grey, move_thresh, first_state_click = store_new_frame(
                stored_frame_0, stored_frame_1, stored_frame_2, current_image, first_state_click, movement_meas_frame)
            if state == AM_forever:
                sum_frames = convert_to_colour(movement_meas_frame, diff_grey, colourstate, sum_frames)
        if state == AM_forever:
            disp_frame = cv2.addWeighted(current_image, 1, sum_frames, 1, 0)
    if state == AM_BPM or state == Brushstroke:  # Time dependent trigger for store_new_frame
        refresh_timer = time.time()
        if refresh_timer >= refresh_trigger:  # Store
            refresh_trigger = refresh_timer + refresh_state[state]
            if state == AM_BPM:
                beats_passed += 1
            if state == Brushstroke or beats_passed >= refresh_beat_interval:
                if state == AM_BPM:
                    bump_diff = 1
                    beats_passed = 0
                stored_frame_0, stored_frame_1, stored_frame_2, movement_meas_frame, diff_grey, move_thresh, first_state_click = store_new_frame(
                    stored_frame_0, stored_frame_1, stored_frame_2, current_image, first_state_click,
                    movement_meas_frame)
                j += 1
            if j >= conv_cycle:
                j = 0
    if state == AM_Faded:
        if bump_diff == 1:
            faded_frames = np.concatenate((faded_frames[1:], placeholder_frame[np.newaxis]), axis=0)
            faded_frames[-1] = current_image
            bump_diff = 0
        K = return_smaller(AM_press_count, afterimage_count)
        if K == 0:  # button hasn;t been pressed before
            disp_frame = current_image
        else:
            for k in afterimage_range:
                combine_frame = cv2.addWeighted(combine_frame, 1.0, faded_frames[k], 1.0 / (K + 1),
                                                0)  # To view just AM, replace current_image with placeholder_frame
            if show_combine_frame:
                disp_frame = cv2.addWeighted(current_image, 1.0 / (K + 1), combine_frame, 1.0, 0)
            else:
                disp_frame = current_image
        control_frame = np.copy(combine_frame)
        combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    if state == AM_BPM or state == AM_On_Click:  # AM: Combine current_image and diff_frames into a single disp_frame image
        if bump_diff == 1:
            diff_frame_to_subtract = np.asarray(diff_frames[(oldest_diff_frame_idx + 1) % afterimage_count],
                                                dtype=np.uint16)
            combine_frame_16 = cv2.addWeighted(combine_frame_16, 1.0, diff_frame_to_subtract, -1.0, 0)
            if converge_frame_bool:
                if converge_frame_counter < afterimage_count - converge_to_keep:
                    diff_frames[oldest_diff_frame_idx] = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                else:
                    diff_frames[oldest_diff_frame_idx] = diff_frames[
                        (oldest_diff_frame_idx + converge_to_keep - 1) % afterimage_count]
                converge_frame_counter += 1
                if converge_frame_counter >= afterimage_count:  # 1 full cycle,
                    bump_diff_hold = False
                    converge_frame_bool = False
                    converge_frame_counter = 0
                    first_state_click = True
            elif empty_frame_bool:
                if converge_frame_counter < afterimage_count:
                    diff_frames[oldest_diff_frame_idx] = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                else:
                    diff_frames[oldest_diff_frame_idx] = diff_frames[(oldest_diff_frame_idx - 1) % afterimage_count]
                converge_frame_counter += 1
                if converge_frame_counter >= afterimage_count:  # 1 full cycle,
                    bump_diff_hold = False
                    empty_frame_bool = False
                    converge_frame_counter = 0
                    first_state_click = True

            else:
                diff_frames[oldest_diff_frame_idx] = movement_meas_frame
            diff_frame_to_add = np.asarray(diff_frames[oldest_diff_frame_idx], dtype=np.uint16)
            combine_frame_16 = cv2.addWeighted(combine_frame_16, 1.0, diff_frame_to_add, 1.0, 0)
            combine_frame = cv2.convertScaleAbs(combine_frame_16)
            oldest_diff_frame_idx += 1
            div_frame_partial_idx = oldest_diff_frame_idx
            if oldest_diff_frame_idx >= afterimage_count:
                oldest_diff_frame_idx = 0
            bump_diff = 0
        if not (save_diverge_frame or diverge_save_trail) or load_diverge_frame:
            control_frame = np.copy(combine_frame)
        if show_combine_frame:
            disp_frame = cv2.addWeighted(current_image, 1.0, combine_frame, AM_weight, 0)
        else:
            disp_frame = np.copy(current_image)
        # if show_combine_window:
        #    control_frame = np.copy(combine_frame)

        if (save_diverge_frame or diverge_save_trail) and bump_diff_hold:
            if save_diverge_count >= diverge_max_frame + 1:
                print("Max diverge framecount achieved. Stopping save_diverge_frame")
                load_diverge_maxframes = save_diverge_count
                # load_diverge_maxframes = np.shape(diverge_frame_load)[0]
                toggle_mystop = True
                save_diverge_count = 0
                save_diverge_frame = False
                diverge_save_trail = False
                diverge_frame_load = np.copy(diverge_frame_saved)
                diverge_frame_saved = np.zeros((diverge_max_frame + 1, cam_res[1], cam_res[0], 3), dtype=np.uint8)
                combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            else:
                if diverge_frame_time_interval <= div_timer_after - div_timer_before:
                    if diverge_save_trail:
                        if toggle_mystop:
                            mystop = oldest_diff_frame_idx
                            toggle_mystop = False
                            print("save idx at mystop = %d" % mystop)
                        print("Trailing, saving first %d frames" % diverge_save_trail_count)
                        combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                        for k in np.arange(0, diverge_save_trail_count):
                            combine_diverge_frame = cv2.addWeighted(combine_diverge_frame, 1.0,
                                                                    diff_frames[(mystop - k) % afterimage_count], 1.0,
                                                                    0)  # To view just AM, replace current_image with placeholder_frame
                        print("myidx = %d" % mystop)
                        diverge_save_trail_count -= 1
                        if diverge_save_trail_count <= -1:
                            toggle_mystop = True
                            load_diverge_maxframes = save_diverge_count
                            save_diverge_count = 0
                            mystop = 0
                            diverge_save_trail = False
                            diverge_frame_load = np.copy(diverge_frame_saved)
                            diverge_frame_saved = np.zeros((diverge_max_frame + 1, cam_res[1], cam_res[0], 3),
                                                           dtype=np.uint8)
                            combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                            combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                    elif save_diverge_count < afterimage_count:
                        combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                        if toggle_mystop:
                            div_frame_partial_idx = mystop
                            toggle_mystop = False
                        for k in np.arange(0, max(save_diverge_count + 2, div_frame_partial_idx - mystop + 1)):
                            combine_diverge_frame = cv2.addWeighted(combine_diverge_frame, 1.0, diff_frames[
                                (div_frame_partial_idx + k) % afterimage_count], 1.0, 0)  # To view just AM, replace current_image with placeholder_frame
                    '''
                    NEW CONSTANTS
                    div_start_frame_count = 1
                    div_start_timer_marker = time.time()
                    
                    
                    elif div_start_frame_count < afterimage_count:
                        combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                        if div_timer_after - div_start_timer_marker >= bump_diff_hold_timer:
                          div_start_frame_count+=1
                        
                        for k in np.arange(1, div_start_frame_count):
                            combine_diverge_frame = cv2.addWeighted(combine_diverge_frame, 1.0, diff_frames[
                                -k], 1.0, 0)  # To view just AM,      
                    '''        
                            print("In start loop = %d" % (div_frame_partial_idx,))
                            print(k)
                    elif diverge_max_frame - save_diverge_count <= afterimage_count:  # all the way to the end
                        if toggle_mystop:
                            mystop = oldest_diff_frame_idx
                            toggle_mystop = False
                        combine_diverge_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                        # for k in np.arange(0,diverge_max_frame-save_diverge_count-1):
                        #    combine_frame = cv2.addWeighted(combine_frame, 1.0, diff_frames[k], 1.0, 0)  # To view just AM, replace current_image with placeholder_frame
                        for k in np.arange(1, diverge_max_frame - save_diverge_count):
                            combine_diverge_frame = cv2.addWeighted(combine_diverge_frame, 1.0,
                                                                    diff_frames[(mystop - k) % afterimage_count], 1.0,
                                                                    0)  # To view just AM, replace current_image with placeholder_frame
                            print(oldest_diff_frame_idx)
                    else:  # running as per normal
                        toggle_mystop = True
                        combine_diverge_frame = np.copy(combine_frame)
                if save_diverge_frame or diverge_save_trail:
                    if save_diverge_count == 0:
                        div_timer_before = time.time()
                        diverge_frame_saved[save_diverge_count] = np.copy(combine_diverge_frame)
                        save_diverge_count += 1
                    elif diverge_frame_time_interval <= div_timer_after - div_timer_before:
                        print("Saving %d th frame after %3f seconds" % (
                        save_diverge_count, div_timer_after - div_timer_before))
                        div_timer_before = time.time()
                        diverge_frame_saved[save_diverge_count] = np.copy(combine_diverge_frame)
                        save_diverge_count += 1
                div_timer_after = time.time()
            control_frame = np.copy(diverge_frame_saved[save_diverge_count - 1])
            '''
            if save_diverge_count == 0:
                div_timer_before = time.time()
                diverge_frame_saved = combine_frame
                save_diverge_count += 1
            elif diverge_frame_time_interval<=div_timer_after-div_timer_before:
                print("Saving %d th frame after %3f seconds" % (save_diverge_count, div_timer_after - div_timer_before))
                div_timer_before = time.time()
                if save_diverge_count == 1:
                    diverge_frame_saved = np.concatenate((diverge_frame_saved[np.newaxis], combine_frame[np.newaxis]), axis=0)
                else: # 2<diverge_count<diverge_maxframe
                    diverge_frame_saved = np.concatenate((diverge_frame_saved, combine_frame[np.newaxis]), axis=0)
                save_diverge_count += 1
            div_timer_after = time.time()
            control_frame = np.copy(diverge_frame_saved[-1])
            '''

    if state == Brushstroke:  #
        debug_image = current_image.copy()
        debug_overlay_image = current_image.copy()
        debug_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        brushframe += 1
        if brushframe >= DETECT_RATE and left_click_tracker.trkState == 2:
            left_click_tracker.DetectorDNNKeypoint(debug_overlay_image, debug_image, BODYINDEX=Body_Index,
                                                   DebugPlotAllPose=False)
            brushframe = 0
            # print(left_click_tracker.correction_point)
        left_click_tracker.main_processTrk(debug_gray, debug_overlay_image, 0, TU.L_refPt, TU.L_newTrk_flag)
        control_frame = np.copy(debug_overlay_image)

        '''
        if not np.array_equal(left_click_tracker.curr_trk_point,no_click_pos):
            midpoint_new = np.array(left_click_tracker.curr_trk_point,dtype = np.uint16)
            midpoint = np.concatenate((midpoint[1:], midpoint_new[np.newaxis]), axis=0)
            midpoint_travel = midpoint[-1] - midpoint[-2]
        else:
            midpoint = np.concatenate((midpoint[1:], midpoint[-1][np.newaxis]), axis=0)
            '''
        if not np.array_equal(left_click_tracker.curr_trk_point, no_click_pos):
            midpoint_new = np.array(left_click_tracker.curr_trk_point, dtype=np.uint16)
            midpoint = np.concatenate((midpoint[1:], midpoint_new[np.newaxis]), axis=0)
            midpoint_travel = midpoint[-1] - midpoint[-2]
        else:
            midpoint = np.concatenate((midpoint[1:], midpoint[-1][np.newaxis]), axis=0)
        brushstroke_frame = np.copy(current_image)
        for m in np.arange(brushstroke_points - 1, 1, -1):  # +1):
            if midpoint[m - 1][0] >= 20000:
                break
            cv2.line(img=brushstroke_frame,
                     pt1=(int(midpoint[m - 1][0]), int(midpoint[m - 1][1])),
                     pt2=(int(midpoint[m][0]), int(midpoint[m][1])),
                     color=brushstroke_colorcycle[int(np.floor((m + offset) / n_factor)) % col_cy_fact],
                     thickness=m * brushstroke_factor)
        flow_tick += 1
        if flow_tick >= flow_interval_frame:  # Flow_tick and offset are for strobing effect
            flow_tick = 0
            offset -= 0.6
        if offset <= 0:
            offset = offset_max
        disp_frame = np.copy(brushstroke_frame)
        # disp_frame = cv2.addWeighted(current_image, 1.0, brushstroke_frame, 1.0, 0)

        '''
        for m in np.arange(brushstroke_points - 1, 1, -1):  # +1):
            if midpoint[m-1][0]>=20000:
                break
            cv2.line(img=brushstroke_frame,
                     pt1=(int(midpoint[m - 1][0]), int(midpoint[m - 1][1])),
                     pt2=(int(midpoint[m][0]), int(midpoint[m][1])),
                     color=brushstroke_colorcycle[int(np.floor((m + offset) / n_factor)) % col_cy_fact],
                     thickness=m * brushstroke_factor)
        flow_tick += 1
        if flow_tick >= flow_interval_frame:  # Flow_tick and offset are for strobing effect
            flow_tick = 0
            offset -= 0.6
        if offset <= 0:
            offset = offset_max
        #cv2.imshow("Brushstroke_frame", brushstroke_frame)
        disp_frame = cv2.addWeighted(current_image, 1.0, brushstroke_frame, 1.0, 0)
        brushstroke_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
        '''
    if state == Brushstroke_multi:
        debug_image = current_image.copy()
        debug_overlay_image = current_image.copy()
        debug_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        brushframe_multi += 1
        if brushframe_multi >= DETECT_RATE_MULTI and DET_ON_MULTI:
            left_click_tracker.YOLODetector(np.copy(current_image), control_frame)
            brushframe_multi = 0
            # The results you wanted#
            print(left_click_tracker.yolo_centroid_results)
        left_click_tracker.main_processTrk(debug_gray, debug_overlay_image, 0, TU.L_refPt, TU.L_newTrk_flag)
        control_frame = np.copy(debug_overlay_image)

        if not np.array_equal(left_click_tracker.curr_trk_point, no_click_pos):
            midpoint_new = np.array(left_click_tracker.curr_trk_point, dtype=np.uint16)
            if midpoint_new[1]>= brushstroke_multi_Y_offset_max:
                midpoint_new[1]+=brushstroke_multi_Y_offset
            midpoint = np.concatenate((midpoint[1:], midpoint_new[np.newaxis]), axis=0)
            midpoint_travel = midpoint[-1] - midpoint[-2]
        else:
            midpoint = np.concatenate((midpoint[1:], midpoint[-1][np.newaxis]), axis=0)
        brushstroke_frame = np.copy(current_image)
        for m in np.arange(brushstroke_points - 1, 1, -1):  # +1):
            if midpoint[m - 1][0] >= 20000:
                break
            cv2.line(img=brushstroke_frame,
                     pt1=(int(midpoint[m - 1][0]), int(midpoint[m - 1][1])),
                     pt2=(int(midpoint[m][0]), int(midpoint[m][1])),
                     color=brushstroke_colorcycle[int(np.floor((m + offset) / n_factor)) % col_cy_fact],
                     thickness=m * brushstroke_factor)
        flow_tick += 1
        if flow_tick >= flow_interval_frame:  # Flow_tick and offset are for strobing effect
            flow_tick = 0
            offset -= 0.6
        if offset <= 0:
            offset = offset_max
        disp_frame = np.copy(brushstroke_frame)
        # disp_frame = cv2.addWeighted(current_image, 1.0, brushstroke_frame, 1.0, 0)
        brushstroke_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    if state == Line:
        # button_checker,lineMidpoint = Joystick_button_check(JoystickPin,button_checker,lineMidpoint)
        mask_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
        'Line Functions'
        mycolor = (0, 0, 0)
        L0, L1 = calc_LineEnds(lineMag, lineMidpoint, line_angle)
        'AM Functions'
        # 'Frame lets us pick out where the line is'
        if line_thickness >= 1:
            frame_to_comp = np.copy(line_diff_last_frame)
        if not np.array_equal(frame_to_comp, current_image):
            line_diff_AM = cv2.absdiff(last_image_4, current_image)
        # TODO: Compute line thickness
        if line_change_mag_bool:
            line_thickness += line_change_mag
            if line_thickness < 0:
                line_thickness = 0
            elif line_thickness > 960:
                line_thickness = 960
                print(line_thickness)
        if line_thickness >= 1:
            line_diff_last_frame = np.copy(current_image)  # store an image before a line is drawn over it
            cv2.line(img=current_image, pt1=(int(L0[0]), int(L0[1])), pt2=(int(L1[0]), int(L1[1])), color=mycolor,
                     thickness=line_thickness)
        if show_line_diff_AM:
            disp_frame = np.copy(line_diff_AM)
        else:
            if line_thickness >= 1:
                cv2.line(img=mask_frame, pt1=(int(L0[0]), int(L0[1])), pt2=(int(L1[0]), int(L1[1])),
                         color=(255, 255, 255),
                         thickness=line_thickness)
            disp_frame = cv2.bitwise_and(line_diff_AM, mask_frame)
        control_frame = np.copy(line_diff_AM)  # np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
    if state == Nothing:
        disp_frame = np.copy(current_image)
        control_frame = np.copy(current_image)
    if show_text:
        str_to_show = word_list[text_counter]
        textsize = cv2.getTextSize(str_to_show, vidfontFace, textfontScale, textfontThick)
        np.array(cam_res)
        text_x = int((np.array(cam_res)[0] - textsize[0][0]) / 2)
        text_y = int((np.array(cam_res)[1] + textsize[1]) / 2)
        cv2.putText(img=disp_frame, text=str_to_show, org=(text_x, text_y), fontFace=vidfontFace,
                    fontScale=textfontScale, color=textfontColour, thickness=textfontThick)
        text_time_after = time.time()
        text_time_passed = text_time_after - text_time_before
        # control_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
        # add put text counter or smth to control_frame here
        if text_time_passed >= text_time_trigger:
            text_counter += 1
            text_time_before = time.time()
        if text_counter >= word_list_len:
            text_counter = 0
            show_text = False
            print("Show text complete.")

    if load_diverge_frame:  # and not save_diverge_frame:
        if diverge_frame_time_interval <= div_timer_after - div_timer_before:
            print("Loading %d th frame after %3f seconds." % (load_diverge_count, div_timer_after - div_timer_before))
            div_timer_before = time.time()
            if load_diverge_count == load_diverge_maxframes - afterimage_count and state == AM_On_Click:
                AM_press_count = 0
                first_state_click = True
                bump_diff = 0
                stored_frame_0 = np.copy(current_image)
                stored_frame_1 = np.copy(stored_frame_0)
                stored_frame_2 = np.copy(stored_frame_0)
                AM_click = 0
                bump_diff_hold = True
                bump_diff_hold_before = time.time()
            if load_diverge_count > load_diverge_maxframes:
                print("Load complete")
                load_diverge_frame = False
                load_diverge_count = 0
                load_diverge_count += 1
                frame_to_load = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            else:
                if load_diverge_count == afterimage_count:
                    bump_diff_hold = False
                    diff_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)
                    combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
                    combine_frame_16 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint16)
                    faded_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)
                load_diverge_count += 1
                if load_diverge_count < diverge_max_frame + 1:
                    frame_to_load = np.copy(diverge_frame_load[load_diverge_count])
        disp_frame = cv2.addWeighted(disp_frame, 1.0, frame_to_load, AM_weight, 0)
        control_frame = np.copy(frame_to_load)
        div_timer_after = time.time()

#    if saveFrame:
#        # cv2.imwrite("./RecFrames/frame_overlay"+str(framenum).zfill(6)+".jpg",debug_overlay_image)
#        cv2.imwrite("./RecFrames/frame" + str(framenum).zfill(6) + ".png", disp_frame)
#        # videofile.write(current_image)

    cv2.rectangle(control_frame, (5, 5), (120, 55), (255, 255, 255), -1)  # white rectangle behind
    cv2.putText(img=control_frame, text="save?: " + str(int(saveFrame)), org=(text_position[0], text_position[1] - 30),
                fontFace=vidfontFace, fontScale=0.5,
                color=(0, 0, 0), thickness=1)
    cv2.putText(img=control_frame, text=toggle_title[state], org=text_position, fontFace=vidfontFace, fontScale=0.5,
                color=(0, 0, 0), thickness=1)  # show state
    if Enigma_Fullscreen:
        cv2.namedWindow("Enigma", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Enigma", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Enigma", disp_frame)
    # cv2.moveWindow("Enigma", 15, 15)
    cv2.imshow("Control Frame", control_frame)
    cv2.moveWindow("Control Frame", 10, 10)

    key = cv2.waitKey(1)
    if key == 111:  # 'o'
        AM_click = 1
        if bump_diff_hold:
            bump_diff_hold = False
    if key == 105:  # 'i'
        bump_diff_hold = not bump_diff_hold
        if not bump_diff_hold:
            bump_diff_hold_before = time.time()
    if key == 101:  # 'e' to reset stuffs
        first_state_click = True
        if state == AM_On_Click:
            diff_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)
            faded_frames = np.zeros((afterimage_count, cam_res[1], cam_res[0], 3), dtype=np.uint8)
            combine_frame = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
            combine_frame_16 = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint16)
        elif state == Line:
            lineMidpoint = np.copy(lineMidpoint_initial)
            line_thickness = line_thickness_initial
            line_angle = line_angle_initial
            line_center_y = line_center[1]
            line_center_x = line_center[0]
            line_change_mag_bool = False
            line_change_angle_bool = False
        elif state == AM_forever:
            sum_frames = np.zeros((cam_res[1], cam_res[0], 3), dtype=np.uint8)
        if bump_diff_hold:
            bump_diff_hold = False
    if key == 108:  # 'l'
        converge_frame_bool = not converge_frame_bool
        if not bump_diff_hold:
            bump_diff_hold = True
    if key == 117:  # 'u'
        empty_frame_bool = not empty_frame_bool
        if not bump_diff_hold:
            bump_diff_hold = True
    if key == 45 and afterimage_count > 1:  # '-'
        afterimage_count -= 1
        diff_frames = (diff_frames[1:])
        faded_frames = faded_frames[1:]
        afterimage_range = np.arange(0, afterimage_count, dtype=np.uint8)
        print(afterimage_count)
    if key == 61:  # '='
        afterimage_count += 1
        diff_frames = np.concatenate((placeholder_frame[np.newaxis], diff_frames), axis=0)
        faded_frames = np.concatenate((placeholder_frame[np.newaxis], faded_frames), axis=0)
        # diff_frames = np.concatenate(diff_frames,placeholder_frame[np.newaxis], axis = 0)
        # diff_frames[:1] = diff_frames[1:]
        afterimage_range = np.arange(0, afterimage_count, dtype=np.uint8)
        print(afterimage_count)
    if key == 49:  # '1'
        state_prev = state
        state = AM_BPM
    if key == 50:  # '2'
        state_prev = state
        state = AM_On_Click
    if key == 51:  # '3'
        state_prev = state
        state = AM_forever
    if key == 52:  # '4'
        state_prev = state
        state = Brushstroke
        print(midpoint)
        print(midpoint_new)
    if key == 53:  # '5'
        state_prev = state
        state = Brushstroke_multi
        print(midpoint)
        print(midpoint_new)
    if key == 54:  # '6'
        state_prev = state
        state = Line
    if key == 55:  # '7'
        state_prev = state
        state = AM_Faded
    if key == 48:  # '0'
        state_prev = state
        state = Nothing
    if key == 113:  # 'q'
        colourstate_prev = colourstate
        colourstate = colourstate_normal
    if key == 97:  # 'a'
        colourstate_prev = colourstate
        colourstate = colourstate_blue
    if key == 115:  # 's'
        colourstate_prev = colourstate
        colourstate = colourstate_green
    if key == 100:  # 'd'
        colourstate_prev = colourstate
        colourstate = colourstate_red
    if key == 122:  # 'z'
        colourstate_prev = colourstate
        colourstate = colourstate_skyblu
    if key == 120:  # 'x'
        colourstate_prev = colourstate
        colourstate = colourstate_yellow
    if key == 99:  # 'c'
        colourstate_prev = colourstate
        colourstate = colourstate_darkviolet
    if key == 119:  # 'w'
        colourstate_prev = colourstate
        colourstate = colourstate_random
    if key == 112:  # 'p'
        show_text = True
        time_time_before = time.time()
    if key == 46:  # '>'
        if not refresh_beat_interval == 1:
            refresh_beat_interval -= 1
        print(refresh_beat_interval)
    if key == 47:  # '?'
        refresh_beat_interval += 1
        print(refresh_beat_interval)
    if key == 107:  # 'k'
        show_combine_frame = not show_combine_frame
        print("Show combine Frame set to %s" % show_combine_frame)
    if key == 106:  # 'j' increase colours for
        if state == Brushstroke or state == Brushstroke_multi:
            if colorcycle_counter < len(brushstroke_colours):
                brushstroke_colorcycle[len(brushstroke_colours) - 1 - colorcycle_counter] = brushstroke_colours[
                    colorcycle_counter]
            elif colorcycle_counter < 2 * len(brushstroke_colours) - 1:
                brushstroke_colorcycle[2 * len(brushstroke_colours) - 1 - colorcycle_counter] = (0, 0, 0)
            else:
                brushstroke_colorcycle = [(0, 0, 0)] * len(brushstroke_colours)
                colorcycle_counter = -1
            colorcycle_counter += 1
    if key == 109:  # 'm'
        if state == Brushstroke_multi:
            print("Swap target")
            if DET_ON_MULTI:
                left_click_tracker.swapTrackViaYOLO()  # Detector needs to be ON to swap successfully
            else:
                # TODO: Need to check if we need to run the DET here
                left_click_tracker.swapTrackViaYOLO()  # Detector needs to be ON to swap successfully#
                left_click_tracker.YOLODetector(np.copy(current_image), control_frame)
    if key == 44:  # '<'
        if state == Brushstroke_multi:
            DET_ON_MULTI = not DET_ON_MULTI
            print("DET_ON_MULTI changed to %s." % DET_ON_MULTI)

    if key == 103:  # 'g'
        lineMidpoint[1] -= Y_move_mag
    if key == 98:  # 'b'
        lineMidpoint[1] += Y_move_mag
    if key == 110:  # 'n'
        lineMidpoint[0] += X_move_mag
    if key == 118:  # 'v'
        lineMidpoint[0] -= X_move_mag
    if key == 104:  # 'h'
        if state == Line:
            if line_change_mag > 0:
                line_change_mag_bool = not line_change_mag_bool
            # line_thickness += line_thickness_mag
            line_change_mag = line_thickness_mag
        elif state == Brushstroke or state == Brushstroke_multi:
            if colorcycle_counter == 0:
                brushstroke_colorcycle[0] = brushstroke_colours[len(brushstroke_colours) - 1]
                colorcycle_counter = 2 * len(brushstroke_colours)
            elif colorcycle_counter <= len(brushstroke_colours):  # counter = 1-4
                brushstroke_colorcycle[len(brushstroke_colours) - colorcycle_counter] = (0, 0, 0)
            else:  # counter = 5-7
                brushstroke_colorcycle[2 * len(brushstroke_colours) - colorcycle_counter] = brushstroke_colours[
                    colorcycle_counter - 1 - len(brushstroke_colorcycle)]
            colorcycle_counter -= 1

    if key == 102:  # 'f'
        if line_change_mag < 0:
            line_change_mag_bool = not line_change_mag_bool
        # line_thickness -= line_thickness_mag
        line_change_mag = -1 * line_thickness_mag
        if line_change_mag_bool:
            print("Decreasing line mag by %d" % line_thickness_mag)
        print(line_change_mag_bool)
    if key == 121:  # 'y'
        line_angle += line_angle_mag
    if key == 114:  # 'r'
        line_angle -= line_angle_mag
    if key == 116:  # 't'
        show_line_diff_AM = not show_line_diff_AM
    if key == 93:  # '['
        Body_Index_num += 1
        Body_Index = Body_Index_list[Body_Index_num % len(Body_Index_list)]
        print("Now Tracking: " + Body_Index_strings[Body_Index])
    if key == 91:  # ']'
        if Body_Index_num >= 1:
            Body_Index_num -= 1
        else:  # Body_Index_num = 0
            Body_Index_num += (len(Body_Index_list) - 1)
        Body_Index = Body_Index_list[Body_Index_num % len(Body_Index_list)]
        print("Now Tracking: " + Body_Index_strings[Body_Index])

    if key == 59:  # ';'
        save_diverge_frame = not save_diverge_frame
        if save_diverge_frame:  # Trigger on
            diverge_frame_saved = np.zeros((diverge_max_frame + 1, cam_res[1], cam_res[0], 3), dtype=np.uint8)
            div_timer_before = time.time()
        else:  # Trigger off
            diverge_frames_left = diverge_max_frame - save_diverge_count
            if diverge_frames_left > afterimage_count:  # trail off to zero
                diverge_save_trail_count = afterimage_count
            else:
                diverge_save_trail_count = diverge_frames_left
            diverge_save_trail = True

    save_diverge_frame_prev = save_diverge_frame  # Do I need this?
    if key == 39:  # ' ' '
        load_diverge_frame = True
        div_timer_before = time.time()
#    if key == 32:
#        saveFrame = not saveFrame

    elif key == 27:  # Esc key to break
        videofile.release()
        break

    # exit()
video_capture.stream.release()
cv2.destroyAllWindows()
# GPIO.cleanup()
