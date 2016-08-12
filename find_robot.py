from __future__ import division
import cv2
import numpy as np


class MultiTrackbarWindow:
    switch = '0 : Lower \n1 : Upper'

    def __init__(self, window_name, trackbars=None, generic_trackbar_count=1,
                 use_switch=True, return_unzipped=True, return_generator=False):
        self.switch_enabled = use_switch
        self.returns_unzipped_results = return_unzipped
        self.returns_generators = return_generator
        self.window_name = window_name
        self.current_mode = 0
        self.trackbars = []
        if trackbars is None:
            for i in range(generic_trackbar_count):
                self.trackbars.append({"name": str(i),
                                       "window": self.window_name,
                                       "lower": 0,
                                       "upper": 255,
                                       "current lower": 0,
                                       "current upper": 255,
                                       "callback": self.nothing})
        else:
            for i, trackbar in enumerate(trackbars):
                if trackbar:
                    if "name" not in trackbar:
                        trackbar["name"] = str(i)
                    if "window" not in trackbar:
                        trackbar["window"] = self.window_name
                    if "lower" not in trackbar:
                        trackbar["lower"] = 0
                    if "upper" not in trackbar:
                        trackbar["upper"] = 255
                    if "callback" not in trackbar:
                        trackbar["callback"] = self.nothing
                    if "current lower" not in trackbar:
                        trackbar["current lower"] = 0
                    if "current upper" not in trackbar:
                        trackbar["current upper"] = 255
                    self.trackbars.append(trackbar)
                else:
                    self.trackbars.append({"name": str(i),
                                           "window": self.window_name,
                                           "lower": 0,
                                           "upper": 255,
                                           "current lower": 0,
                                           "current upper": 255,
                                           "callback": self.nothing})
        cv2.namedWindow(self.window_name)
        for trackbar in self.trackbars:
            cv2.createTrackbar(trackbar["name"], trackbar["window"],
                               trackbar["lower"], trackbar["upper"],
                               trackbar["callback"])
        if self.switch_enabled:
            cv2.createTrackbar(MultiTrackbarWindow.switch, self.window_name, 0, 1, self.nothing)

    def nothing(self, x):
        pass

    def get_range(self):
        if self.switch_enabled:
            toggle_lower_upper = cv2.getTrackbarPos(MultiTrackbarWindow.switch, self.window_name)
        else:
            toggle_lower_upper = 0

        if toggle_lower_upper == 0:
            if self.current_mode == 1:
                for trackbar in self.trackbars:
                    cv2.setTrackbarPos(trackbar["name"], trackbar["window"], trackbar["current lower"])
                self.current_mode = 0
            for trackbar in self.trackbars:
                trackbar["current lower"] = cv2.getTrackbarPos(trackbar["name"], trackbar["window"])
        else:
            if self.current_mode == 0:
                for trackbar in self.trackbars:
                    cv2.setTrackbarPos(trackbar["name"], trackbar["window"], trackbar["current upper"])
                self.current_mode = 1
            for trackbar in self.trackbars:
                trackbar["current upper"] = cv2.getTrackbarPos(trackbar["name"], trackbar["window"])
        if self.returns_unzipped_results:
            lower = (trackbar["current lower"] for trackbar in self.trackbars)
            upper = (trackbar["current upper"] for trackbar in self.trackbars)
            if not self.returns_generators:
                lower = tuple(lower)
                upper = tuple(upper)
            result = (lower, upper)
        else:
            result = ((trackbar["current lower"], trackbar["current upper"]) for trackbar in self.trackbars)
            if not self.returns_generators:
                result = tuple(result)

        return result


def angle_of_normal_between(a, b, adjustment=0):
    try:
        radians = np.arctan2(a[1]-b[1], b[0]-a[0])
        degrees = (adjustment + np.rad2deg(radians)) % 360
        return degrees
    except ZeroDivisionError:
        return None


def get_largest_contour(image, lower, upper, window_name=None):
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    if window_name is not None:
        cv2.imshow(window_name, image)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    radius = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
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
    draw_color = (0, 255, 0)
    middle = pt1[0]+(pt2[0]-pt1[0])//2, pt1[1]+(pt2[1]-pt1[1])//2
    # pt1_arr = np.array(pt1)
    # pt2_arr = np.array(pt2)
    # middle = pt1_arr+(pt2_arr-pt1_arr)/2
    # middle = tuple(map(int, middle))
    cv2.putText(image, text, middle, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (128, 255, 200))
    # if dynamic_color:
    #     pixel_color = image[middle[0], middle[1]]
    #     pixel_hsv = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2HSV)[0][0]
    #     complementary_pixel_hsv = [(pixel_hsv[0]-127)%255, pixel_hsv[1], pixel_hsv[2]]
    #     draw_color = tuple(map(int, cv2.cvtColor(np.uint8([[complementary_pixel_hsv]]), cv2.COLOR_HSV2BGR)[0][0]))
    cv2.line(image, pt1, pt2, color=draw_color)
    return image


def find_target(image_to_analyze, trackbar_window,
                image_to_draw_on=None, indicator_color=(0, 0, 0),
                adjustment_ratio=1):
    lower, upper = trackbar_window.get_range()
    median = cv2.medianBlur(image_to_analyze, 15)
    mask = cv2.inRange(median, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, None, iterations=2)
    cv2.imshow(trackbar_window.window_name, mask)

    # Find contours in the mask and initialize the current (x, y) center of the ball
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    radius = None

    # Proceed only if at least one contour was found.
    if len(contours) > 0:
        # Find the largest contour in the mask, then use it to
        # compute the minimum enclosing circle and centroid.
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        center = (int(adjustment_ratio*x), int(adjustment_ratio*y))
        radius = int(adjustment_ratio*radius)
    if center is not None and image_to_draw_on is not None:
        draw_largest_contour(image_to_draw_on, center, radius, color=indicator_color)
    return center, radius


def find_agent(image_to_analyze, image_to_draw_on, left_trackbar_window, right_trackbar_window,
               scale_up_ratio=1, show_result=True):
    left_center, left_radius = find_target(image_to_analyze, left_trackbar_window,
                                           image_to_draw_on, (255, 0, 255),
                                           adjustment_ratio=scale_up_ratio)
    right_center, right_radius = find_target(image_to_analyze, right_trackbar_window,
                                             image_to_draw_on, (0, 255, 0),
                                             adjustment_ratio=scale_up_ratio)

    if left_center is not None and right_center is not None:
        angle = round(angle_of_normal_between(left_center, right_center), 2)
        draw_robot_graphics(image_to_draw_on, str(angle), left_center, right_center)
    if show_result:
        cv2.imshow("Agent Location", image_to_draw_on)


def main(camera_index=0, scale_down_ratio=1, scale_up_ratio=None):
    if scale_up_ratio is None:
        scale_up_ratio = 1 / scale_down_ratio
    left_controller = MultiTrackbarWindow("Left Tracker", [{"name": "H"}, {"name": "S"}, {"name": "V"}])
    right_controller = MultiTrackbarWindow("Right Tracker", [{"name": "H"}, {"name": "S"}, {"name": "V"}])
    # Declare the interface through which the camera will be accessed
    camera = cv2.VideoCapture(camera_index)
    while True:
        # Grab the current frame
        able_to_retrieve_frame, frame = camera.read()
        # Get a boolean value determining whether a frame was successfully grabbed
        # from the camera and then the actual frame itself.
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break

        # Resize the frame, blur it, and convert it to the HSV color space.
        resized = cv2.resize(frame, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)
        blurred = cv2.GaussianBlur(resized, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # cv2.imshow("HSV", hsv)
        find_agent(hsv, frame, left_controller, right_controller, scale_up_ratio)
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
