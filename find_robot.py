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
    pt1_arr = np.array(pt1)
    pt2_arr = np.array(pt2)
    middle = pt1_arr+(pt2_arr-pt1_arr)/2
    middle = tuple(map(int, middle))
    cv2.putText(image, text, middle, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (128, 255, 200))
    cv2.line(image, pt1, pt2, color=(0, 255, 0))
    return image


def main(show_left=False, show_right=False, show_both=True):
    left_draw_color = (255, 0, 255)
    right_draw_color = (0, 255, 0)
    left_hsv_window = HSVTrackbarWindow("Left Mask", *HSVTrackbarWindow.default_hsv_range)
    right_hsv_window = HSVTrackbarWindow("Right Mask", *HSVTrackbarWindow.default_hsv_range)
    # Declare the interface through which the camera will be accessed
    camera_index = 0
    camera = cv2.VideoCapture(camera_index)

    while True:
        # grab the current frame
        able_to_retrieve_frame, frame = camera.read()
        # Get a boolean value determining whether a frame was successfuly grabbed
        # from the camera and then the actual frame itself.
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break

        # Resize the frame, blur it, and convert it to the HSV color space.
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)
        left_center, left_radius = get_largest_contour(hsv, left_hsv_window)
        right_center, right_radius = get_largest_contour(hsv, right_hsv_window)

        if left_center is not None:
            left_frame = draw_largest_contour(frame.copy(), left_center, left_radius, color=left_draw_color)
            if show_left:
                cv2.imshow("Left Tracker", left_frame)
        if right_center is not None:
            right_frame = draw_largest_contour(frame.copy(), right_center, right_radius, color=right_draw_color)
            if show_right:
                cv2.imshow("Right Tracker", right_frame)
        if left_center is not None and right_center is not None:
            both_frames = draw_largest_contour(left_frame, right_center, right_radius, color=right_draw_color)
            angle = round(angle_of_normal_between(left_center, right_center), 2)
            both_frames = draw_robot_graphics(both_frames, str(angle), left_center, right_center)
            if show_both:
                cv2.imshow("Both Trackers", both_frames)

        # Get the value for the key we entered with '& 0xFF' for 64-bit systems
        key = cv2.waitKey(1) & 0xFF
        # Stop the program if the key was q/Q
        if key == ord('q') or key == ord('Q'):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
