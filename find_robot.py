from __future__ import division
import numpy as np
import imutils
import cv2


class HSVTrackbarWindow:
    default_hsv_range = [0, 0, 0, 255, 255, 255]
    switch = '0 : Lower \n1 : Upper'

    def __init__(self, window_name, H_l, S_l, V_l, H_u, S_u, V_u):
        self.H_l, self.S_l, self.V_l = H_l, S_l, V_l
        self.H_u, self.S_u, self.V_u = H_u, S_u, V_u
        self.window_name = window_name
        self.current_mode = 0
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("H", self.window_name, 0, 255, self.nothing)
        cv2.createTrackbar("S", self.window_name, 0, 255, self.nothing)
        cv2.createTrackbar("V", self.window_name, 0, 255, self.nothing)
        cv2.createTrackbar(HSVTrackbarWindow.switch, self.window_name, 0, 1, self.nothing)

    def nothing(self, x):
        print(x)
        pass

    def get_range(self):
        toggle_lower_upper = cv2.getTrackbarPos(HSVTrackbarWindow.switch, self.window_name)

        if toggle_lower_upper == 0:
            if self.current_mode == 1:
                cv2.setTrackbarPos('H', self.window_name, self.H_l)
                cv2.setTrackbarPos('S', self.window_name, self.S_l)
                cv2.setTrackbarPos('V', self.window_name, self.V_l)
                self.current_mode = 0
            self.H_l = cv2.getTrackbarPos('H', self.window_name)
            self.S_l = cv2.getTrackbarPos('S', self.window_name)
            self.V_l = cv2.getTrackbarPos('V', self.window_name)
        else:
            if self.current_mode == 0:
                cv2.setTrackbarPos('H', self.window_name, self.H_u)
                cv2.setTrackbarPos('S', self.window_name, self.S_u)
                cv2.setTrackbarPos('V', self.window_name, self.V_u)
                self.current_mode = 1
            self.H_u = cv2.getTrackbarPos('H', self.window_name)
            self.S_u = cv2.getTrackbarPos('S', self.window_name)
            self.V_u = cv2.getTrackbarPos('V', self.window_name)

        lower = (self.H_l, self.S_l, self.V_l)
        upper = (self.H_u, self.S_u, self.V_u)

        return lower, upper


"""

I'm still working on the math and everything, but when done, this will be able to be customized
to track any colored object we want and can then be used to track the angle of the robot using
two colored blobs on a sheet of paper placed on top of the bot.

"""

#rightLower = (50, 250, 60)
#rightUpper = (90, 255, 255)



def angle_of_normal_between(a, b, adjustment=90):
    try:
        radians = np.arctan2(a[1]-b[1], b[0]-a[0])
        degrees = (adjustment + np.rad2deg(radians)) % 360
        return degrees
    except ZeroDivisionError:
        return None

def get_largest_contour(image, hsv_trackbar_window):

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask

    hsv_lower, hsv_upper = hsv_trackbar_window.get_range()

    hsv_median = cv2.medianBlur(image, 15)
    mask = cv2.inRange(hsv_median, hsv_lower, hsv_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow(hsv_trackbar_window.window_name, mask)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    radius = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
    return center, radius


def draw_largest_contour(image, center, radius, min_radius=10, color=(0, 0, 0)):
    # only proceed if the radius meets a minimum size
    if radius > min_radius:
        # draw the circle and centroid on the frame,
        # then update the list of tracked points
        cv2.circle(image, center, radius, color, 2)
    return image

def draw_robot_graphics(image, text, pt1, pt2):
    cv2.putText(image, text, (20, 30), cv2.FONT_HERSHEY_PLAIN,
                2.3, (0, 0, 255))
    cv2.line(image, pt1, pt2, color=(0,255,0))
    return image


def main():
    camera = cv2.VideoCapture(0)
    leftDrawColor = (255, 0, 255)
    rightDrawColor = (0, 255, 0)

    left_hsv_window = HSVTrackbarWindow("Left Mask", *HSVTrackbarWindow.default_hsv_range)
    right_hsv_window = HSVTrackbarWindow("Right Mask", *HSVTrackbarWindow.default_hsv_range)



    while True:
        # grab the current frame
        grabbed, frame = camera.read()
        # Resize the frame, blur it, and convert it to the HSV color space.
        frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)
        left_center, left_radius = get_largest_contour(hsv, left_hsv_window)

        right_center, right_radius = get_largest_contour(hsv, right_hsv_window)

        if left_center is not None:
            left_frame = draw_largest_contour(frame, left_center, 20, color=leftDrawColor)
#            cv2.imshow("Left Tracker", left_frame)
        if right_center is not None:
            right_frame = draw_largest_contour(frame, right_center, 30, color=rightDrawColor)
#            cv2.imshow("Right Tracker", right_frame)
        if left_center is not None and right_center is not None:
            both_frames = draw_largest_contour(left_frame, right_center, right_radius, color=rightDrawColor)
            both_frames = draw_robot_graphics(both_frames, str(angle_of_normal_between(left_center, right_center)),
                                              left_center, right_center)
            cv2.imshow("Both Trackers", both_frames)

        # show the frame to our screen
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

main()