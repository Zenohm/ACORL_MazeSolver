import cv2
import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from skimage import img_as_ubyte
from skimage.morphology import medial_axis
import matplotlib.pyplot as plt
import find_robot

"""
# Probablistic Hough Lines on Picture
img = cv2.imread("img2.jpg")
small = cv2.resize(img, (0, 0), fx=0.075, fy=0.075)
hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, (211, 77, 42), (214, 77, 44))
mask = cv2.erode(mask, None, iterations=2)
mask = cv2.dilate(mask, None, iterations=2)
gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

edges = cv2.Canny(im_bw, 50, 150, apertureSize = 3)


#plt.subplot(121), plt.imshow(small, cmap='gray')
#plt.subplot(122), plt.imshow(edges, cmap='gray')
#plt.show()


lines = cv2.HoughLinesP(edges, 1, np.pi/2, 20, 2, 17, 2)

#these numbers are black magic as of right now
#lines = cv2.HoughLinesP(edges, 1, np.pi/360, 20, None, 50, 5)

for line in lines[0]:
    pt1 = (line[0], line[1])
    pt2 = (line[2], line[3])
    cv2.line(small, pt1, pt2, (0, 0, 255), 1)



redFinder = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
redFinder = cv2.inRange(small, (0, 0, 255), (0, 0, 255))
lrg = cv2.resize(redFinder, (0, 0), fx=20, fy = 20)
cv2.imwrite("Lines.jpg", lrg)
"""

"""
Alright bear with a brotha for a second. I don't think we need any kind of line detection at all.
The thresheld image automatically maps out the free space for us (by what black magic I do not know).
All we need to do then is build a list of those points in the occupied (i.e. white) space that lie directly
next to a point in the free (i.e. black) space and those are our lines. That list we can then ship off to PAGI
world for maze construction. This dramatically cuts down on computational time. As it stands we're able to build
voronoi diagrams in real time, which is pretty great!
"""


def nothing(x):
    '''Placeholder function so the trackbar has an empty callback function'''
    pass


def voronoi(storage_length=10, delete_length=1, save_images=False,
            use_rolling_average=True, use_line_buffer=True, draw_voronoi=True, find_keypoints=True,
            scale_down_ratio=0.75, scale_up_ratio=None):
    assert storage_length >= delete_length
    if scale_up_ratio is None:
        scale_up_ratio = 1 / scale_down_ratio
    # Define constants here.
    kernel = np.ones((5, 5), np.uint8)
    # A BGR color value used to denote the color thresholded for the occupancy grid
    # This is also the color used to draw lines, thus how we get the ^^^^^^^^^^^^^^
    track_color = (0, 0, 255)
    if find_keypoints:
        # ORB keypoint detection
        orb = cv2.ORB()
    if use_line_buffer:
        list_of_lines = []
    if use_rolling_average:
        first_run = True
        rolling_average_image = None
        cv2.namedWindow("Rolling Average")
        cv2.createTrackbar("Frame Fade", "Rolling Average", 95, 100, nothing)
        cv2.createTrackbar("Detection Strength", "Rolling Average", 40, 100, nothing)
    cv2.namedWindow("Occupancy")
    cv2.createTrackbar("Line Thickness", "Occupancy", 0, 25, nothing)
    # Declare the interface through which the camera will be accessed
    camera_index = 0
    camera = cv2.VideoCapture(camera_index)

    while True:
        # Get a boolean value determining whether a frame was successfuly grabbed
        # from the camera and then the actual frame itself.
        able_to_retrieve_frame, img = camera.read()
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break
        resized = cv2.resize(img, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow("Threshold", thresh)

        # Canny edge detection and Hough Line Transform
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        min_line_length = 100  # 85
        max_line_gap = 85  # 100
        # Try to find points which look like lines according to our settings.
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, None, min_line_length, max_line_gap)

        # If we find some lines to draw, draw them and display them.
        if lines is not None:
            if use_line_buffer:
                if len(list_of_lines) >= storage_length:
                    del list_of_lines[:delete_length]
                list_of_lines.append(lines)
            # This allows us to adjust the thickness of the lines on the fly
            line_thickness = cv2.getTrackbarPos("Line Thickness", "Occupancy")
            if use_line_buffer:
                for each_set_of_lines in list_of_lines:
                    for x1, y1, x2, y2 in each_set_of_lines[0]:
                        # Draw the lines based on the gathered points
                        cv2.line(resized, (x1, y1), (x2, y2), track_color, line_thickness)
            else:
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
                    alpha = cv2.getTrackbarPos("Frame Fade", "Rolling Average")/100.0
                    beta = cv2.getTrackbarPos("Detection Strength", "Rolling Average")/100.0
                    rolling_average_image = cv2.addWeighted(rolling_average_image, alpha, occupancy_map, beta, 0)
                    occupancy_map = rolling_average_image
                    cv2.imshow("Rolling Average", rolling_average_image)
                    result = cv2.add(thresh, rolling_average_image)
                    cv2.imshow("Rolling Occupancy", result)
                else:
                    rolling_average_image = occupancy_map
                    first_run = False
            """
            # Consider using corners instead of keypoints to reduce jumpiness
            corners = cv2.goodFeaturesToTrack(trackColorMap, 25, 0.01, 10)
            corners = np.int0(corners)
            for i in corners:
                x,y = i.ravel()
                cv2.circle(gray, (x,y), 3, 255 ,-1)
            """

            if find_keypoints:
                # Find and draw the key points
                key_points = orb.detect(thresh, None)
                key_point_overlay = cv2.drawKeypoints(resized, key_points, color=(0, 255, 0), flags=0)
                cv2.imshow("Image with key points", key_point_overlay)

                # Format the occupancy map for Voronoi diagram
                occupied_space = []
                for each_key_point in key_points:
                    x, y = each_key_point.pt
                    occupied_space.append((x, y))

            if draw_voronoi:
                # Compute the Voronoi diagram and draw the relevant sections on the screen.
                # For more qhull options: http://www.qhull.org/html/qh-quick.htm#options
                voronoi_diagram = Voronoi(occupied_space, furthest_site=False, incremental=False, qhull_options="v s Pg Qz")
                voronoi_plot_2d(voronoi_diagram)
                vertices = voronoi_diagram.vertices

                voronoi_map = occupancy_map.copy()
                for v1, v2 in voronoi_diagram.ridge_vertices:
                    # This allows us to adjust the thickness of the lines on the fly
                    line_thickness = cv2.getTrackbarPos("Line Thickness", "Occupancy")
                    # Draw the lines based on the gathered points
                    if v1 != -1 or v2 != -1:
                        try:
                            cv2.line(voronoi_map,
                                     tuple(map(int, vertices[v1])),
                                     tuple(map(int, vertices[v2])),
                                     (255, 255, 0), line_thickness)
                        except OverflowError:
                            # I'm not sure what causes this. Research.
                            pass
                cv2.imshow("Ridge Points", voronoi_map)

        # Get the value for the key we entered with '& 0xFF' for 64-bit systems
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            if save_images:
                cv2.imwrite("occupancy.png", occupancy_map)
                cv2.imwrite("Threshold.png", thresh)
            break

    # Clean up, go home
    camera.release()
    cv2.destroyAllWindows()


def skeleton(use_median_filter=False, invert_threshold=False, show_threshold=False, show_skeleton=False,
             show_skeleton_threshold=True, show_occupancy=False, find_keypoints=True,
             scale_down_ratio=0.75, scale_up_ratio=None):
    # Initialize windows to display frames and controls
    if show_threshold:
        cv2.namedWindow("Threshold")
    if show_skeleton:
        cv2.namedWindow("Skeleton-ized")
    if show_skeleton_threshold:
        cv2.namedWindow("Skeleton with Threshold")
    if scale_up_ratio is None:
        scale_up_ratio = 1 / scale_down_ratio
    # Define constants here.
    orb = cv2.ORB()
    # Precompute a kernel. For information
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    if show_occupancy:
        cv2.namedWindow("Occupancy")
        cv2.createTrackbar("Line Thickness", "Occupancy", 0, 25, nothing)
    # Declare the interface through which the camera will be accessed
    camera_index = 0
    camera = cv2.VideoCapture(camera_index)

    while True:
        # Get a boolean value determining whether a frame was successfuly grabbed
        # from the camera and then the actual frame itself.
        able_to_retrieve_frame, frame = camera.read()
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break
        resized = cv2.resize(frame, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        ret, thresh = cv2.threshold(gray_blur, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        if show_threshold:
            thresh_temp = cv2.resize(thresh, (0, 0), fx=scale_up_ratio, fy=scale_up_ratio)
            cv2.imshow("Threshold", thresh_temp)

        # Invert our threshold image since this will skeleton-ize white areas
        if invert_threshold:
            thresh = cv2.bitwise_not(thresh)
        inverted_thresh = cv2.bitwise_not(thresh)

        if use_median_filter:
            # SciPy Method, slower but better defined.
            path_skeleton, distance = medial_axis(inverted_thresh, return_distance=True)
            dist_on_skel = path_skeleton * distance
            path_skeleton = img_as_ubyte(path_skeleton)
            critical_points = find_critical_points(dist_on_skel, number_of_points=10, edge_width=10)
            print(critical_points)
            critical_point_overlay = cv2.cvtColor(path_skeleton.copy(), cv2.COLOR_GRAY2BGR)
            #critical_point_overlay_temp = cv2.resize(critical_point_overlay, (0, 0),
            #                                         fx=scale_up_ratio, fy=scale_up_ratio)
            for each_point in critical_points:
                cv2.circle(critical_point_overlay, each_point, 5, (0, 0, 255), 2)
            cv2.imshow("Critical Points", critical_point_overlay)
        else:
            # OPENCV Method, faster, but not as smooth.
            # Erode the image gradually, subtracting away sections until the skeleton is one pixel wide
            # Create a blank matrix to draw the skeleton on.
            path_skeleton = np.zeros(inverted_thresh.shape, np.uint8)
            while cv2.countNonZero(inverted_thresh):
                # Erode the image.
                eroded = cv2.erode(inverted_thresh, kernel)
                # Dilate the eroded image.
                temp = cv2.dilate(eroded, kernel)
                # Take the difference between the modified image and the original image
                temp = cv2.subtract(inverted_thresh, temp)
                # Write that difference to the skeleton matrix
                path_skeleton = cv2.bitwise_or(path_skeleton, temp)
                # Set the eroded image as the current image.
                inverted_thresh = eroded.copy()
                # Check to see if the sections are one-pixel wide.

        if find_keypoints:
            # Find and draw the key points
            key_points = orb.detect(path_skeleton, None)
            key_point_overlay = cv2.drawKeypoints(path_skeleton, key_points, color=(0, 255, 0), flags=0)
            cv2.imshow("Image with key points", key_point_overlay)

            # Format the occupancy map for Voronoi diagram
            occupied_space = []
            for each_key_point in key_points:
                x, y = each_key_point.pt
                occupied_space.append((x, y))

        if show_skeleton:
            path_skeleton_temp = cv2.resize(path_skeleton, (0, 0), fx=scale_up_ratio, fy=scale_up_ratio)
            cv2.imshow("Skeleton-ized", path_skeleton_temp)
        threshold_and_skeleton = cv2.add(thresh, path_skeleton)
        threshold_and_skeleton = cv2.resize(threshold_and_skeleton, (0, 0), fx=scale_up_ratio, fy=scale_up_ratio)
        if show_skeleton_threshold:
            cv2.imshow("Skeleton with Threshold", threshold_and_skeleton)

        # Get the value for the key we entered with '& 0xFF' for 64-bit systems
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
    # Clean up, go home
    camera.release()
    cv2.destroyAllWindows()


def find_critical_points(distance_on_skeleton, number_of_points, edge_width=50):
    # This assumes this is not a ragged array.
    top = edge_width
    bottom = len(distance_on_skeleton) - edge_width
    left = edge_width
    # If this needs to work with ragged arrays, move this next line everywhere the length of the row may change.
    right = len(distance_on_skeleton[0]) - edge_width
    flattened_distances = [item for sublist in distance_on_skeleton[top:bottom]
                           for item in sublist[left:right] if item]
    sorted_distances = sorted(list(set(flattened_distances)))
    critical_numbers = sorted_distances[:number_of_points]
    critical_points = []
    # IT'S THE END TIMES! THE END TIIIIIMES!!!
    # O(n^3) eat my CPU out... And send pictures plzthxbai
    for critical_number in critical_numbers:
        found_num = False
        for x, row in enumerate(distance_on_skeleton):
            for y, value in enumerate(row):
                if value == critical_number and left <= x <= right and top <= y <= bottom:
                    critical_points.append((x, y))
                    found_num = True
                    break
            if found_num:
                break
    # Now that the end times are over, let's continue.
    return critical_points

if __name__ == "__main__":
    skeleton(use_median_filter=False, scale_down_ratio=0.5, scale_up_ratio=None, show_skeleton=True,
             show_skeleton_threshold=True, show_threshold=True, find_keypoints=False)
    voronoi(storage_length=20, delete_length=5, use_rolling_average=True,
            use_line_buffer=False, draw_voronoi=False, find_keypoints=False)
    find_robot.main(show_left=True, show_right=True, show_both=True)
