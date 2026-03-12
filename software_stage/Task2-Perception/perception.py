import math
import cv2
import numpy as np
import sys


class RoboGambit_Perception:

    def __init__(self):
        # PARAMETERS - Camera intrinsics provided by organisers (DO NOT MODIFY)
        self.camera_matrix = np.array([
            [1030.4890823364258, 0, 960],
            [0, 1030.489103794098, 540],
            [0, 0, 1]
        ], dtype=np.float32)

        self.dist_coeffs = np.zeros((1, 5))

        # INTERNAL VARIABLES
        self.corner_world = {
            21: (350, 350),
            22: (350, -350),
            23: (-350, -350),
            24: (-350, 350)
        }
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []

        self.H_matrix = None

        self.board = np.zeros((6, 6), dtype=int)

        # ARUCO DETECTOR
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        print("Perception Initialized")


    # DO NOT MODIFY THIS FUNCTION
    def prepare_image(self, image):
        """
        DO NOT MODIFY.
        Performs camera undistortion and grayscale conversion.
        """
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        gray_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        return undistorted_image, gray_image


    def pixel_to_world(self, pixel_x, pixel_y):
        """
        Convert pixel coordinates into world coordinates using homography.
        Steps:
        1. Ensure homography matrix has been computed.
        2. Format pixel point for cv2.perspectiveTransform().
        3. Return transformed world coordinates.
        """
        if self.H_matrix is None:
            print("Homography matrix not computed yet.")
            return None, None

        # Format the pixel point as required by perspectiveTransform: shape (1, 1, 2)
        pixel_point = np.array([[[float(pixel_x), float(pixel_y)]]], dtype=np.float32)

        # Apply homography to transform pixel -> world
        world_point = cv2.perspectiveTransform(pixel_point, self.H_matrix)

        world_x = world_point[0][0][0]
        world_y = world_point[0][0][1]

        return world_x, world_y


    def process_image(self, image):
        """
        Main perception pipeline.
        """

        self.board[:] = 0

        # Reset internal state for fresh processing
        self.corner_pixels = {}
        self.pixel_matrix = []
        self.world_matrix = []
        self.H_matrix = None

        # Preprocess image (Do not modify)
        undistorted_image, gray_image = self.prepare_image(image)

        # ── 1. Detect ArUco markers ──────────────────────────────────────────
        corners, ids, rejected = self.detector.detectMarkers(gray_image)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(undistorted_image, corners, ids)
        else:
            print("No ArUco markers detected.")
            res = cv2.resize(undistorted_image, (1152, 648))
            cv2.imshow("Detected Markers", res)
            self.visualize_board()
            return

        # ── 2. Separate corner markers (21-24) and piece markers (1-10) ──────
        CORNER_IDS = {21, 22, 23, 24}
        PIECE_IDS  = set(range(1, 11))

        piece_markers = []  # piece_id -> (center_px, center_py)

        for i, marker_id in enumerate(ids.flatten()):
            # Compute the centre of the detected marker from its 4 corner pixels
            marker_corners = corners[i][0]  # shape (4, 2)
            cx = float(np.mean(marker_corners[:, 0]))
            cy = float(np.mean(marker_corners[:, 1]))

            if marker_id in CORNER_IDS:
                self.corner_pixels[marker_id] = (cx, cy)
            elif marker_id in PIECE_IDS:
                piece_markers.append((marker_id, cx, cy))

        # ── 3. Build pixel and world matrices ────────────────────────────────
        found_corners = [mid for mid in CORNER_IDS if mid in self.corner_pixels]
        print(f"Found corner markers: {found_corners}")

        if len(found_corners) < 4:
            print(f"Warning: only {len(found_corners)}/4 corner markers detected.")
            if len(found_corners) < 3:
                print("Not enough corner markers to compute homography.")
                res = cv2.resize(undistorted_image, (1152, 648))
                cv2.imshow("Detected Markers", res)
                self.visualize_board()
                return

        self.pixel_matrix = []
        self.world_matrix = []

        for mid in found_corners:
            px, py = self.corner_pixels[mid]
            wx, wy = self.corner_world[mid]
            self.pixel_matrix.append([px, py])
            self.world_matrix.append([wx, wy])

        pixel_pts = np.array(self.pixel_matrix, dtype=np.float32)
        world_pts = np.array(self.world_matrix, dtype=np.float32)

        # ── 4. Compute homography (pixel space -> world space) ────────────────
        self.H_matrix, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)

        if self.H_matrix is None:
            print("Homography computation failed.")
            res = cv2.resize(undistorted_image, (1152, 648))
            cv2.imshow("Detected Markers", res)
            self.visualize_board()
            return

        print(f"Homography computed successfully.")

        # ── 5. Convert piece marker centres to world coords & place on board ──
        for piece_id, px, py in piece_markers:
            wx, wy = self.pixel_to_world(px, py)
            if wx is None:
                continue
            print(f"Piece {piece_id}: pixel ({px:.1f}, {py:.1f}) -> world ({wx:.1f}, {wy:.1f})")
            self.place_piece_on_board(piece_id, wx, wy)

        print("Board state:")
        print(self.board)

        # Visualization (Do not modify)
        res = cv2.resize(undistorted_image, (1152, 648))
        cv2.imshow("Detected Markers", res)
        self.visualize_board()


    def place_piece_on_board(self, piece_id, x_coord, y_coord):
        """
        Places detected piece on the closest board square.

        Board definition:
            6x6 grid
            top-left corner = (300, 300) in world coords (mm)
            square size     = 100 mm

        World coordinate system (matches corner_world assignments):
            +X → right
            +Y → up  (row 0 is at the highest Y value)

        Square centres (col c, row r):
            world_x_centre = 300 - 50 - c*100  =  250 - c*100
            world_y_centre = 300 - 50 - r*100  =  250 - r*100

        Inverting:
            c = (250 - x_coord) / 100
            r = (250 - y_coord) / 100
        """

        SQUARE_SIZE = 100.0  # mm

        col = (250.0 - x_coord) / SQUARE_SIZE
        row = (250.0 - y_coord) / SQUARE_SIZE

        col_idx = int(round(col))
        row_idx = int(round(row))

        # Clamp to valid board indices [0, 5]
        col_idx = max(0, min(5, col_idx))
        row_idx = max(0, min(5, row_idx))

        self.board[row_idx][col_idx] = piece_id
        print(f"  -> Placed piece {piece_id} at board[{row_idx}][{col_idx}]")


    # DO NOT MODIFY THIS FUNCTION
    def visualize_board(self):
        """
        Draw a simple 6x6 board with detected piece IDs
        """
        cell_size = 80
        board_img = np.ones((6 * cell_size, 6 * cell_size, 3), dtype=np.uint8) * 255

        for r in range(6):
            for c in range(6):
                x1 = c * cell_size
                y1 = r * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                cv2.rectangle(board_img, (x1, y1), (x2, y2), (0, 0, 0), 2)

                piece = int(self.board[r][c])
                if piece != 0:
                    cv2.putText(board_img, str(piece), (x1 + 25, y1 + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Game Board", board_img)


# DO NOT MODIFY
def main():
    # To run code, use python/python3 perception.py path/to/image.png
    if len(sys.argv) < 2:
        print("Usage: python perception.py image.png")
        return

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    perception = RoboGambit_Perception()
    perception.process_image(image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()