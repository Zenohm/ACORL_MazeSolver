import cv2
import numpy as np
from scipy.spatial import Voronoi
from skimage import img_as_ubyte
from skimage.morphology import medial_axis, skeletonize
import find_robot

# TODO: Work on integrating find_robot with the rest of the code.

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

"""
Alright bear with a brotha for a second. I don't think we need any kind of line detection at all.
The thresheld image automatically maps out the free space for us (by what black magic I do not know).
All we need to do then is build a list of those points in the occupied (i.e. white) space that lie directly
next to a point in the free (i.e. black) space and those are our lines. That list we can then ship off to PAGI
world for maze construction. This dramatically cuts down on computational time. As it stands we're able to build
voronoi diagrams in real time, which is pretty great!
"""


def nothing(x):
    """Placeholder function so the trackbar has an empty callback function"""
    pass


def get_threshold(image_to_analyze, morphology_iterations=2, blur_amount=13, kernel=None):
    """
    Convenience function for finding the threshold of an image.
    :param image_to_analyze: Image to manipulate in search of results.
    :param morphology_iterations: How many times to perform the specified morphological operations.
    :param blur_amount: How much to blur the inputted image before thresholding it.
    :param kernel: Used to influence how morphological manipulations of the image take place.
    :return: The thresholded image.
    """
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    gray = cv2.cvtColor(image_to_analyze, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_amount, blur_amount), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=morphology_iterations)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=morphology_iterations)
    return thresh


def display(window_name, image, resize_ratio=None):
    """
    Convenience function for scaling and displaying results.
    :param window_name: Name of the window to show the image in.
    :param image: Image to be displayed
    :param resize_ratio: What ratio to rescale outgoing results to before they're displayed.
    :return: None
    """
    if resize_ratio is not None:
        image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)
    cv2.imshow(window_name, image)


def find_keypoints(image_to_analyze, image_to_draw_on=None, keypoint_count=0, resize_ratio=1):
    """
    Find and draw the key points
    :param image_to_analyze: Image to manipulate in search of results.
    :param image_to_draw_on: Image passed in that is to be used for drawing the results of analysis.
    :param keypoint_count: Maximum number of keypoints to search for.
    :param resize_ratio: What ratio to rescale outgoing results to before they're displayed.
    :return: A set of points indicating the location of keypoints.
    """
    key_points = cv2.goodFeaturesToTrack(image_to_analyze, keypoint_count, 0.01, 10)
    # Format the space of occupied points for the Voronoi Diagram
    occupied_space = [tuple(resize_ratio*each_key_point[0]) for each_key_point in key_points]
    if image_to_draw_on is not None:
        # TODO: Work resize_ratio into the drawing.
        for point in occupied_space:
            cv2.circle(image_to_draw_on, point, 2, (0, 255, 0))
        cv2.imshow("Keypoints", image_to_draw_on)
    return occupied_space


def find_lines(image_to_analyze, image_to_draw_on, min_line_length=100, max_line_gap=85,
               rolling_average_image=None, draw_voronoi=True, use_keypoints_for_voronoi=True,
               resize_ratio=1):
    """
    Canny edge detection and Hough Line Transform with a rolling average for stability and the ability
    to draw Voronoi Diagrams with the results.
    :param image_to_analyze: Image to manipulate in search of results.
    :param image_to_draw_on: Image passed in that is to be used for drawing the results of analysis.
    :param min_line_length: Minimum line length. Line segments shorter than that are rejected.
    :param max_line_gap: Maximum allowed gap between points on the same line to link them.
    :param rolling_average_image: Holds an image that is to be averaged with incoming results.
    :param draw_voronoi: Whether a Voronoi Diagram will be computed.
    :param use_keypoints_for_voronoi: Whether keypoints of the threshold will be used as the input
                                      when computing the Voronoi Diagram. This speeds up the calculation.
    :param resize_ratio: What ratio to rescale outgoing results to before they're displayed.
    :return: An image that contains the current rolling average of all occupancy map results.
    """
    # TODO: Work resize_ratio into the drawing.
    if rolling_average_image is None:
        rolling_average_image = np.zeros(image_to_draw_on.shape, np.uint8)
    edges = cv2.Canny(image_to_analyze, 50, 150, apertureSize=3)
    # Try to find points which look like lines according to our settings.
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, None, min_line_length, max_line_gap)

    # If we find some lines to draw, draw them and display them.
    if lines is not None:
        occupancy_map = np.zeros(image_to_draw_on.shape, np.uint8)
        # This allows us to adjust the thickness of the lines on the fly
        line_thickness = cv2.getTrackbarPos("Line Thickness", "Occupancy")
        for each_point_pair in lines[0]:
            scaled_points = map(lambda n: int(resize_ratio*n), each_point_pair)
            x1, y1, x2, y2 = scaled_points
            # Draw the lines based on the gathered points
            cv2.line(occupancy_map, (x1, y1), (x2, y2), 255, line_thickness)
        display("Occupancy", occupancy_map, 1)

        # Do the rolling average image calculation
        alpha = cv2.getTrackbarPos("Frame Fade", "Rolling Average") / 100.0
        beta = cv2.getTrackbarPos("Detection Strength", "Rolling Average") / 100.0
        rolling_average_image = cv2.addWeighted(rolling_average_image, alpha, occupancy_map, beta, 0)
        cv2.imshow("Rolling Average", rolling_average_image)
        """
        # Consider using corners instead of keypoints to reduce jumpiness
        corners = cv2.goodFeaturesToTrack(trackColorMap, 25, 0.01, 10)
        corners = np.int0(corners)
        for i in corners:
            x,y = i.ravel()
            cv2.circle(gray, (x,y), 3, 255 ,-1)
        """
        if draw_voronoi:
            calculate_voronoi(image_to_analyze, image_to_draw_on.copy(), use_keypoints=use_keypoints_for_voronoi,
                              resize_ratio=resize_ratio)
        return rolling_average_image


def calculate_voronoi(image_to_analyze, image_to_draw_on=None, use_keypoints=True, resize_ratio=1):
    """
    Computes the Voronoi Diagram for a set of points, displays the
    results if you want, and returns the result of the computation.
    :param image_to_analyze: Image to manipulate in search of results.
    :param image_to_draw_on: Image passed in that is to be used for drawing the results of analysis.
    :param use_keypoints: Try to compute the Voronoi Diagram based just on the keypoints as an optimization.
    :param resize_ratio: What ratio to rescale outgoing results to before they're displayed.
    :return: The result of computing the Voronoi Diagram.
    """
    # TODO: Work resize_ratio into the drawing.
    if use_keypoints:
        occupied_space = find_keypoints(image_to_analyze=image_to_analyze, resize_ratio=resize_ratio)
    else:
        occupied_space = image_to_analyze
    # Compute the Voronoi diagram and draw the relevant sections on the screen.
    # For more qhull options: http://www.qhull.org/html/qh-quick.htm#options
    voronoi_diagram = Voronoi(occupied_space, furthest_site=False, incremental=False)
    vertices = voronoi_diagram.vertices

    if image_to_draw_on is not None:
        # TODO: Figure out what's causing those jumping lines and fix it.
        for v1, v2 in voronoi_diagram.ridge_vertices:
            # This allows us to adjust the thickness of the lines on the fly
            line_thickness = cv2.getTrackbarPos("Line Thickness", "Occupancy")
            # Draw the lines based on the gathered points.
            if v1 != -1 or v2 != -1:
                try:
                    cv2.line(image_to_draw_on,
                             tuple(map(int, vertices[v1])),
                             tuple(map(int, vertices[v2])),
                             (255, 255, 0), line_thickness)
                except OverflowError:
                    # TODO: I'm not sure what causes this. Research.
                    pass
        cv2.imshow("Voronoi Map", image_to_draw_on)
    return voronoi_diagram


def get_skeleton_of_maze(image_to_analyze, image_to_draw_on=None,
                         use_medial_axis=True, invert_threshold=False,
                         locate_critical_points=True, resize_ratio=1):
    """
    Computes, returns, and potentially displays the morphological skeleton of the given binary image.
    :param image_to_analyze: Image to manipulate in search of results.
    :param image_to_draw_on: Image passed in that is to be used for drawing the results of analysis.
    :param use_medial_axis: Whether to use an alternative method for finding the skeleton that
                              allows the computation of the critical points in the image.
    :param invert_threshold: Whether the threshold should be inverted before the skeleton is located
    :param locate_critical_points: Whether to find and draw critical points on the skeleton.
    :param resize_ratio: What ratio to rescale outgoing results to before they're displayed.
    :return: The skeleton of the image, and the critical points (If you chose to try to find them)
    """
    result = []
    # Invert our thresholded image since this will skeletonize white areas
    if invert_threshold:
        image_to_analyze = cv2.bitwise_not(image_to_analyze)
    image_to_analyze = cv2.bitwise_not(image_to_analyze)

    if use_medial_axis or locate_critical_points:
        # http://scikit-image.org/docs/dev/auto_examples/plot_medial_transform.html
        # In short, allows us to find the distance to areas on the skeleton.
        # This information can be used to find critical points in the skeleton, theoretically.
        path_skeleton, distance = medial_axis(image_to_analyze, return_distance=True)
        distance_on_skeleton = path_skeleton * distance
        path_skeleton = img_as_ubyte(path_skeleton)
        result.append(path_skeleton)

        if locate_critical_points:
            critical_points = find_critical_points(distance_on_skeleton, number_of_points=10,
                                                   minimum_thickness=50, edge_width=20,
                                                   image_to_draw_on=image_to_draw_on)
            result.append(critical_points)
    else:
        skeleton = skeletonize(image_to_analyze/255)
        path_skeleton = np.array(skeleton*255, np.uint8)
        result.append(path_skeleton)

    if image_to_draw_on is not None:
        path_skeleton_temp = cv2.cvtColor(path_skeleton, cv2.COLOR_GRAY2BGR)
        superimposed_skeleton = cv2.add(image_to_draw_on, path_skeleton_temp)
        display("Skeleton", superimposed_skeleton, resize_ratio)
    return result


def find_critical_points(distance_on_skeleton, number_of_points,
                         minimum_thickness=10, edge_width=20,
                         image_to_draw_on=None, resize_ratio=1):
    """
    Used to identify critical points in a skeleton, which can be
    used to divide the skeleton up into topological regions.
    :param distance_on_skeleton: Obtained by finding the medial axis of the image, contains information about
                                 the local width of the obtained skeleton which is useful for finding critical points.
    :param number_of_points: How many critical points you want to locate.
    :param minimum_thickness: Minimum local thickness of the skeleton for it to be
                              considered when searching for a critical point.
    :param edge_width: How far you want the results to be from the edges of the image.
    :param image_to_draw_on: Image passed in that is to be used for drawing the results of analysis.
    :param resize_ratio: What ratio to rescale outgoing results to before they're displayed.
    :return: The locations of the critical points
    """
    # This assumes this is not a ragged array.
    top = edge_width
    bottom = len(distance_on_skeleton) - edge_width
    left = edge_width
    # If this needs to work with ragged arrays, move this next line everywhere the length of the row may change.
    right = len(distance_on_skeleton[0]) - edge_width
    # Get all the notable distance values within the boundaries defined by edge_width
    flattened_distances = [item for sub_list in distance_on_skeleton[top:bottom]
                           for item in sub_list[left:right] if item]
    filtered_distances = filter(lambda n: not n % 2 and n > minimum_thickness, flattened_distances)
    sorted_distances = sorted(list(set(filtered_distances)))
    critical_numbers = sorted_distances[:number_of_points]
    critical_points_array = []
    critical_points = []
    # IT'S THE END TIMES! THE END TIIIIIMES!!!
    # O(n^3) eat my CPU out... And send pictures plzthxbai
    matrix = np.array(distance_on_skeleton)
    for critical_number in critical_numbers:
        critical_points_array.append(np.where(matrix == critical_number))
    # Now that the end times are over, let's continue.
    for detection in critical_points_array:
        x = detection[0]
        y = detection[1]
        for a, b in zip(x, y):
            if (a, b) != (None, None):
                critical_points.append((int(resize_ratio*b), int(resize_ratio*a)))

    if image_to_draw_on is not None:
        critical_point_overlay = image_to_draw_on
        # critical_point_overlay_temp = cv2.resize(critical_point_overlay, (0, 0),
        #                                          fx=scale_up_ratio, fy=scale_up_ratio)
        # TODO: Work resize_ratio into the drawing.
        for each_point in critical_points:
            cv2.circle(critical_point_overlay, each_point, 5, (0, 0, 255), 2)
        cv2.imshow("Critical Points", critical_point_overlay)
    return critical_points


def find_maze(camera_index=0, draw_skeleton=True, show_threshold=False,
              use_rolling_average=True, draw_voronoi=True,
              save_images=False, locate_keypoints=True,
              use_keypoints_for_voronoi=True, locate_critical_points=True,
              scale_down_ratio=1, scale_up_ratio=None,
              min_line_length=100, max_line_gap=85):
    """
    Organizes the logic of the various functions used to find a maze in the camera.
    :param camera_index: Which camera to grab frames from.
    :param draw_skeleton: Whether to compute and display the morphological skeleton.
    :param show_threshold: Whether to show the binary image that will be used for computation.
    :param save_images: Whether the occupancy map and the threshold should be saved when you press Q to quit.
    :param use_rolling_average: Determines if a rolling average will be used to stabilize the occupancy map.
    :param draw_voronoi: Whether a Voronoi Diagram will be computed
    :param use_keypoints_for_voronoi: Whether keypoints of the threshold will be used as the input
                                      when computing the Voronoi Diagram. This speeds up the calculation.
    :param locate_critical_points: Whether to find and draw critical points on the skeleton.
    :param locate_keypoints: Will keypoints be located and drawn on the image?
    :param scale_down_ratio: What ratio to scale incoming frames to prior to their processing.
    :param scale_up_ratio: What ratio to rescale outgoing results to before they're displayed.
    :param min_line_length: Minimum line length. Line segments shorter than that are rejected.
    :param max_line_gap: Maximum allowed gap between points on the same line to link them.
    :return: None
    """
    if scale_up_ratio is None:
        scale_up_ratio = 1 / scale_down_ratio
    # Initialize windows to display frames and controls
    if use_rolling_average:
        rolling_average_image = None
        cv2.namedWindow("Rolling Average")
        cv2.createTrackbar("Frame Fade", "Rolling Average", 95, 100, nothing)
        cv2.createTrackbar("Detection Strength", "Rolling Average", 40, 100, nothing)
        cv2.namedWindow("Occupancy")
        cv2.createTrackbar("Line Thickness", "Occupancy", 0, 25, nothing)
    if draw_skeleton:
        cv2.namedWindow("Skeleton")
    # Declare the interface through which the camera will be accessed
    camera = cv2.VideoCapture(camera_index)

    while True:
        # Get a boolean value determining whether a frame was successfully
        # grabbed from the camera and then the actual frame itself.
        able_to_retrieve_frame, frame = camera.read()
        if not able_to_retrieve_frame:
            print("Camera is not accessible. Is another application using it?")
            print("Check to make sure other versions of this program aren't running.")
            break
        resized = cv2.resize(frame, (0, 0), fx=scale_down_ratio, fy=scale_down_ratio)
        thresh = get_threshold(resized)
        if show_threshold:
            display("Threshold", thresh, scale_up_ratio)

        if use_rolling_average:
            rolling_average_image = find_lines(thresh, frame.copy(),
                                               min_line_length, max_line_gap,
                                               rolling_average_image, draw_voronoi=draw_voronoi,
                                               use_keypoints_for_voronoi=use_keypoints_for_voronoi,
                                               resize_ratio=scale_up_ratio)
        if draw_skeleton:
            get_skeleton_of_maze(thresh, frame.copy(), resize_ratio=scale_up_ratio,
                                 locate_critical_points=locate_critical_points)
        if locate_keypoints:
            find_keypoints(thresh, frame.copy(), keypoint_count=0, resize_ratio=scale_up_ratio)

        # Get the value for the key we entered with '& 0xFF' for 64-bit systems
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            if save_images:
                cv2.imwrite("occupancy.png", rolling_average_image)
                cv2.imwrite("Threshold.png", thresh)
            break

    # Clean up, go home
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_maze(camera_index=1, scale_down_ratio=1, draw_skeleton=True, use_rolling_average=False,
              draw_voronoi=False, locate_keypoints=False)
    # find_robot.main(1, scale_down_ratio=0.5)
