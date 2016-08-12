import cv2
import imutils
import numpy as np


kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

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


def nothing(x):
    '''Placeholder function so the trackbar has an empty callback function'''
    pass


def get_maze_rectangle(image, threshold_window):
    hsv_lower, hsv_upper = threshold_window.get_range()
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    other_colorspace = cv2.cvtColor(yuv, cv2.COLOR_BGR2HSV_FULL)
    # cv2.imshow("YUV", yuv)
    # cv2.imshow("HSV", hsv)
    cv2.imshow("HSV Full before", other_colorspace)
    a, b, c = cv2.split(other_colorspace)
    # a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, kernel, iterations=2)
    # a = cv2.morphologyEx(a, cv2.MORPH_OPEN,  kernel, iterations=2)
    # a = cv2.GaussianBlur(a, (19, 19), 0)
    # b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kernel, iterations=2)
    # b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  kernel, iterations=2)
    # b = cv2.GaussianBlur(b, (3, 3), 0)
    # c = cv2.morphologyEx(c, cv2.MORPH_CLOSE, kernel, iterations=2)
    # c = cv2.morphologyEx(c, cv2.MORPH_OPEN,  kernel, iterations=2)
    # c = cv2.GaussianBlur(c, (11, 11), 0)
    cv2.imshow("Hue", a)
    cv2.imshow("Sat", b)
    cv2.imshow("Val", c)
    other_colorspace = cv2.merge((a, b, c))
    cv2.imshow("HSV Full after", other_colorspace)
    thresh = cv2.inRange(other_colorspace, hsv_lower, hsv_upper)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=2)
    # thresh = cv2.add(image, image, mask=thresh)
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5,10, 30)
    #gray = cv2.GaussianBlur(gray, (2, 2), 0)
    '''
    cv2.imshow("Threshold", thresh)
    tweedle_dee = cv2.getTrackbarPos("Thingy1", "Edged")  # 30
    tweedle_dum = cv2.getTrackbarPos("Thingy2", "Edged")  # 200
    edged = cv2.Canny(thresh, tweedle_dee, tweedle_dum)
    cv2.imshow("Edged", edged)
    (im, cnts, hierarchy) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        precision = cv2.getTrackbarPos("Precision", "Edged")
        precision /= 1000
        approx = cv2.approxPolyDP(c, precision * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4 or len(approx) == 12:
            screenCnt = approx
            break

    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("Game Boy Screen", image)

    return None


def skeleton(camera_index=0, show_occupancy=False,
             scale_down_ratio=0.75, scale_up_ratio=None):
    # Initialize windows to display frames and controls
    threshold_window = HSVTrackbarWindow("Threshold", *HSVTrackbarWindow.default_hsv_range)
    cv2.namedWindow("Edged")
    if scale_up_ratio is None:
        scale_up_ratio = 1 / scale_down_ratio
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    if show_occupancy:
        cv2.namedWindow("Occupancy")
        cv2.createTrackbar("Line Thickness", "Occupancy", 0, 25, nothing)
    cv2.createTrackbar("Precision", "Edged", 0, 1000, nothing)
    cv2.createTrackbar("Thingy1", "Edged", 0, 100, nothing)
    cv2.createTrackbar("Thingy2", "Edged", 0, 1000, nothing)
    # Declare the interface through which the camera will be accessed
    camera = cv2.VideoCapture(camera_index)

    while True:
        # Get a boolean value determining whether a frame was successfuly grabbed
        # from the camera and then the actual frame itself.
        able_to_retrieve_frame, frame = camera.read()
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break
        ratio = frame.shape[0] / 300.0
        orig = frame.copy()
        image = imutils.resize(frame, height=300)
        #frame = cv2.resize(frame, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)

        try:
            rect = get_maze_rectangle(image, threshold_window)
        except cv2.error:
            pass
        '''hsv_lower, hsv_upper = threshold_window.get_range()
        hsv_median = cv2.medianBlur(frame, 15)
        thresh = cv2.inRange(hsv_median, hsv_lower, hsv_upper)

        resized = cv2.resize(frame, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh_temp = cv2.resize(thresh, (0, 0), fx=scale_up_ratio, fy=scale_up_ratio)
        #cv2.imshow("Threshold", thresh_temp)
        '''
        # Get the value for the key we entered with '& 0xFF' for 64-bit systems
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
    # Clean up, go home
    camera.release()
    cv2.destroyAllWindows()


def find_lines(camera_index=0, storage_length=10, delete_length=1, use_rolling_average=True,
               scale_down_ratio=0.75, scale_up_ratio=None):
    assert storage_length >= delete_length
    if scale_up_ratio is None:
        scale_up_ratio = 1 / scale_down_ratio
    # Define constants here.
    kernel = np.ones((1, 1), np.uint8)
    # A BGR color value used to denote the color thresholded for the occupancy grid
    # This is also the color used to draw lines, thus how we get the ^^^^^^^^^^^^^^
    track_color = (0, 0, 255)
    if use_rolling_average:
        first_run = True
        rolling_average_image = None
        cv2.namedWindow("Rolling Average")
        cv2.createTrackbar("Frame Fade", "Rolling Average", 95, 100, nothing)
        cv2.createTrackbar("Detection Strength", "Rolling Average", 40, 100, nothing)
    cv2.namedWindow("Occupancy")
    cv2.createTrackbar("Line Thickness", "Occupancy", 0, 25, nothing)
    # Declare the interface through which the camera will be accessed
    camera = cv2.VideoCapture(camera_index)
    min_line_length = 0
    while True:
        # Get a boolean value determining whether a frame was successfully
        # grabbed from the camera and then the actual frame itself.
        able_to_retrieve_frame, img = camera.read()
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break
        resized = cv2.resize(img, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # blur = cv2.GaussianBlur(gray, (21, 21), 0)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow("Threshold", thresh)

        # Canny edge detection and Hough Line Transform
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        cv2.imshow("Canny edges", edges)
        min_line_length = 100  # 85
        max_line_gap = 85  # 100
        # Try to find points which look like lines according to our settings.
        lines = cv2.HoughLinesP(edges, 5, np.pi/180, 75, None, min_line_length, max_line_gap)

        # If we find some lines to draw, draw them and display them.
        if lines is not None:
            # This allows us to adjust the thickness of the lines on the fly
            line_thickness = cv2.getTrackbarPos("Line Thickness", "Occupancy")
            for x1, y1, x2, y2 in lines[0]:
                # Draw the lines based on the gathered points
                cv2.line(resized, (x1, y1), (x2, y2), track_color, line_thickness)
            # Display those happy little lines overlaid on our original image
            cv2.imshow("Image with Overlay", resized)
            # Threshold the image looking for those red lines we just drew and display the result
            occupancy_map = cv2.inRange(resized, track_color, track_color)
            cv2.imshow("Occupancy", occupancy_map)

            if use_rolling_average:
                if not first_run:
                    alpha = cv2.getTrackbarPos("Frame Fade", "Rolling Average") / 100.0
                    beta = cv2.getTrackbarPos("Detection Strength", "Rolling Average") / 100.0
                    rolling_average_image = cv2.addWeighted(rolling_average_image, alpha, occupancy_map, beta, 0)
                    occupancy_map = rolling_average_image
                    cv2.imshow("Rolling Average", rolling_average_image)
                    result = cv2.add(thresh, rolling_average_image)
                    cv2.imshow("Rolling Occupancy", result)
                else:
                    rolling_average_image = occupancy_map
                    first_run = False
        # Get the value for the key we entered with '& 0xFF' for 64-bit systems
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
    # Clean up, go home
    camera.release()
    cv2.destroyAllWindows()


def main(camera_index):
    find_lines(camera_index)

if __name__ == "__main__":
    main(camera_index=1)