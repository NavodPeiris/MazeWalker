import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import json

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Resize image if too large while maintaining aspect ratio
    max_dimension = 1000
    height, width = img.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Enhanced preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding instead of simple thresholding
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return img, binary, gray



def detect_grid_size_improved(binary_img, gray_img, debug=False):
    """
    Improved method to detect the maze's grid size (5x5, 9x9, or 16x16).
    Uses multiple approaches and cross-validates the results.
    """
    # Get image dimensions
    h, w = binary_img.shape
    
    # Method 1: Find contours to get the maze boundary
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    # Get the main contour (the maze boundary)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Crop the image to the maze boundaries
    maze_binary = binary_img[y:y+h, x:x+w].copy()
    maze_gray = gray_img[y:y+h, x:x+w].copy()
    
    # Method 2: Use edge detection and Hough transform for line detection
    edges = cv2.Canny(maze_binary, 50, 150, apertureSize=3)
    
    # Detect lines using probabilistic Hough transform
    min_line_length = min(w, h) // 16  # Make this smaller to detect more lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                           minLineLength=min_line_length, maxLineGap=10)
    
    if lines is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=min_line_length//2, maxLineGap=30)
        if lines is None:
            raise ValueError("Could not detect grid lines in the maze")
    
    # Method 3: Use adaptive thresholding for better edge detection
    if debug:
        # Draw all detected lines on a debug image
        debug_img = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite("detected_lines.jpg", debug_img)
    
    # Categorize lines as horizontal or vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Calculate angle of the line
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Horizontal lines (angle close to 0 or 180 degrees)
        if angle < 30 or angle > 150:
            horizontal_lines.append((y1 + y2) // 2)  # y-coordinate
        # Vertical lines (angle close to 90 degrees)
        elif 60 < angle < 120:
            vertical_lines.append((x1 + x2) // 2)  # x-coordinate
    
    # Group similar coordinates to handle slight variations
    def group_coordinates(coords, threshold=None):
        if not coords:
            return []
        if threshold is None:
            threshold = min(w, h) // 30  # Adaptive threshold based on image size
        coords = sorted(coords)
        groups = [[coords[0]]]
        for coord in coords[1:]:
            if coord - groups[-1][-1] <= threshold:
                groups[-1].append(coord)
            else:
                groups.append([coord])
        return [sum(group) // len(group) for group in groups]
    
    # Group similar coordinates
    horizontal_positions = group_coordinates(horizontal_lines)
    vertical_positions = group_coordinates(vertical_lines)
    
    if debug:
        # Create a debug image showing the grouped line positions
        debug_img2 = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        for y in horizontal_positions:
            cv2.line(debug_img2, (0, y), (w, y), (0, 255, 0), 1)
        for x in vertical_positions:
            cv2.line(debug_img2, (x, 0), (x, h), (255, 0, 0), 1)
        cv2.imwrite("grouped_lines.jpg", debug_img2)
    
    # Method 4: Cell counting approach
    # Calculate approximate cell size
    cell_width = w // len(vertical_positions) if vertical_positions else w
    cell_height = h // len(horizontal_positions) if horizontal_positions else h
    
    # Count number of grid cells
    num_cells_x = len(vertical_positions) + 1
    num_cells_y = len(horizontal_positions) + 1
    
    # Determine grid size (9x9, or 16x16)
    # Depending on which is more reliable, choose the max or min
    grid_size = max(num_cells_x, num_cells_y)
    
    # Additional method: Analyze the full image directly to count cells
    # Compute the cell size by dividing the maze dimensions by potential grid sizes
    potential_sizes = [9, 16]
    optimal_grid_size = None
    best_score = float('-inf')
    
    for size in potential_sizes:
        approx_cell_width = w / size
        approx_cell_height = h / size
        
        # Score how well this grid size matches the detected lines
        score = 0
        
        # Check horizontal lines
        for i in range(1, size):
            expected_y = int(y + i * approx_cell_height)
            # Find closest detected horizontal line
            closest_diff = min([abs(expected_y - pos) for pos in horizontal_positions], default=w)
            score -= closest_diff
        
        # Check vertical lines
        for i in range(1, size):
            expected_x = int(x + i * approx_cell_width)
            # Find closest detected vertical line
            closest_diff = min([abs(expected_x - pos) for pos in vertical_positions], default=h)
            score -= closest_diff
        
        if score > best_score:
            best_score = score
            optimal_grid_size = size
    
    # Final decision: Use optimal_grid_size if available, otherwise use the count-based approach
    final_grid_size = optimal_grid_size if optimal_grid_size is not None else grid_size
    
    # If number of detected lines suggests 9x9 or 16x16, prefer those over 5x5
    if num_cells_x > 6 or num_cells_y > 6:
        if num_cells_x <= 12 or num_cells_y <= 12:
            final_grid_size = 9
        else:
            final_grid_size = 16

    # Match to closest standard size if not already one of the standard sizes
    if final_grid_size not in [9, 16]:
        diffs = [(abs(final_grid_size - size), size) for size in [9, 16]]
        final_grid_size = min(diffs)[1]
    
    print(f"Detected grid dimensions: {num_cells_x}x{num_cells_y}")
    print(f"Determined grid size: {final_grid_size}x{final_grid_size}")
    
    # For debugging, draw a grid of the detected size
    if debug:
        debug_img3 = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        cell_size = w / final_grid_size
        for i in range(final_grid_size + 1):
            y_pos = int(i * cell_size)
            x_pos = int(i * cell_size)
            cv2.line(debug_img3, (0, y_pos), (w, y_pos), (0, 255, 255), 1)  # Horizontal
            cv2.line(debug_img3, (x_pos, 0), (x_pos, h), (0, 255, 255), 1)  # Vertical
        cv2.imwrite("determined_grid.jpg", debug_img3)
    
    return final_grid_size, maze_binary, (x, y, w, h)

def extract_cells(maze_img, grid_size, original_img, maze_bounds):
    """Extract individual cells from the maze."""
    h, w = maze_img.shape
    cell_h, cell_w = h // grid_size, w // grid_size
    
    # Draw grid on original image for debugging
    debug_img = original_img.copy()
    x0, y0, _, _ = maze_bounds
    
    cells = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            # Extract cell
            cell = maze_img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            row.append(cell)
            
            # Draw cell boundaries for debugging
            cv2.rectangle(debug_img, 
                         (x0 + j*cell_w, y0 + i*cell_h), 
                         (x0 + (j+1)*cell_w, y0 + (i+1)*cell_h), 
                         (0, 255, 0), 1)
        cells.append(row)
    
    return cells, debug_img

def analyze_cell_walls(cell, i, j, grid_size, debug=False):
    """
    Analyze a cell to detect which walls are present.
    In binary image, wall pixels are white (255) and open space is black (0).
    """
    h, w = cell.shape
    
    # Define narrow regions along each edge to check for walls
    # A wall is detected if there are enough white pixels in the region
    wall_thickness = max(2, min(h, w) // 10)  # Adjust based on cell size
    threshold_percentage = 0.4  # Percentage of white pixels needed to identify a wall
    
    # Check top wall
    top_region = cell[0:wall_thickness, :]
    top_wall = 1 if np.sum(top_region) / (top_region.size * 255) > threshold_percentage else 0
    
    # Check right wall
    right_region = cell[:, w-wall_thickness:w]
    right_wall = 1 if np.sum(right_region) / (right_region.size * 255) > threshold_percentage else 0
    
    # Check bottom wall
    bottom_region = cell[h-wall_thickness:h, :]
    bottom_wall = 1 if np.sum(bottom_region) / (bottom_region.size * 255) > threshold_percentage else 0
    
    # Check left wall
    left_region = cell[:, 0:wall_thickness]
    left_wall = 1 if np.sum(left_region) / (left_region.size * 255) > threshold_percentage else 0
    
    # Special handling for edge cells - outer boundary should always have walls
    if i == 0:  # Top row
        top_wall = 1
    if j == grid_size - 1:  # Rightmost column
        right_wall = 1
    if i == grid_size - 1:  # Bottom row
        bottom_wall = 1
    if j == 0:  # Leftmost column
        left_wall = 1
    
    if debug:
        # Create debug directory if it doesn't exist
        os.makedirs("cell_debug", exist_ok=True)
        
        # Create a debug view of the cell with detected walls
        debug_cell = cv2.cvtColor(cell.copy(), cv2.COLOR_GRAY2BGR)
        if top_wall:
            cv2.line(debug_cell, (0, wall_thickness//2), (w, wall_thickness//2), (0, 0, 255), 1)
        if right_wall:
            cv2.line(debug_cell, (w-wall_thickness//2, 0), (w-wall_thickness//2, h), (0, 0, 255), 1)
        if bottom_wall:
            cv2.line(debug_cell, (0, h-wall_thickness//2), (w, h-wall_thickness//2), (0, 0, 255), 1)
        if left_wall:
            cv2.line(debug_cell, (wall_thickness//2, 0), (wall_thickness//2, h), (0, 0, 255), 1)
        
        cv2.imwrite(f"cell_debug/cell_{i}_{j}.png", debug_cell)
    
    return [top_wall, right_wall, bottom_wall, left_wall]

def ensure_wall_consistency(walls, grid_size):
    """
    Ensure that shared walls between adjacent cells are consistent.
    If cell A has a right wall, cell B to its right must have a left wall, and vice versa.
    If cell A has a bottom wall, cell B below it must have a top wall, and vice versa.
    """
    # Convert the flat list of walls into a 2D grid for easier adjacency checks
    grid_walls = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            idx = i * grid_size + j
            row.append(walls[idx])
        grid_walls.append(row)
    
    # Fix inconsistencies
    for i in range(grid_size):
        for j in range(grid_size):
            # Check right wall consistency
            if j < grid_size - 1:
                # If cell has right wall, adjacent cell should have left wall
                if grid_walls[i][j][1] == 1:
                    grid_walls[i][j+1][3] = 1
                # If adjacent cell has left wall, cell should have right wall
                elif grid_walls[i][j+1][3] == 1:
                    grid_walls[i][j][1] = 1
            
            # Check bottom wall consistency
            if i < grid_size - 1:
                # If cell has bottom wall, cell below should have top wall
                if grid_walls[i][j][2] == 1:
                    grid_walls[i+1][j][0] = 1
                # If cell below has top wall, cell should have bottom wall
                elif grid_walls[i+1][j][0] == 1:
                    grid_walls[i][j][2] = 1
    
    # Convert back to flat list
    consistent_walls = []
    for i in range(grid_size):
        for j in range(grid_size):
            consistent_walls.append(grid_walls[i][j])
    
    return consistent_walls

def analyze_all_cells(cells, grid_size, debug=False):
    """Analyze all cells in the maze to detect walls."""
    walls = []
    
    # Create a debug grid image
    if debug:
        debug_grid = np.ones((grid_size * 50, grid_size * 50, 3), dtype=np.uint8) * 255
    
    for i in range(grid_size):
        for j in range(grid_size):
            cell_walls = analyze_cell_walls(cells[i][j], i, j, grid_size, debug=debug)
            walls.append(cell_walls)
            
            # Draw on debug grid
            if debug:
                cell_y, cell_x = i * 50, j * 50
                # Draw the cell
                cv2.rectangle(debug_grid, (cell_x, cell_y), (cell_x + 50, cell_y + 50), (200, 200, 200), 1)
                # Draw detected walls
                if cell_walls[0]:  # Top
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x + 50, cell_y), (0, 0, 0), 2)
                if cell_walls[1]:  # Right
                    cv2.line(debug_grid, (cell_x + 50, cell_y), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                if cell_walls[2]:  # Bottom
                    cv2.line(debug_grid, (cell_x, cell_y + 50), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                if cell_walls[3]:  # Left
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x, cell_y + 50), (0, 0, 0), 2)
                
                # Add labels
                cv2.putText(debug_grid, f"{i},{j}", (cell_x + 15, cell_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    # Apply wall consistency check
    walls = ensure_wall_consistency(walls, grid_size)
    
    # Update debug grid with consistent walls if in debug mode
    if debug:
        # Clear previous grid
        debug_grid = np.ones((grid_size * 50, grid_size * 50, 3), dtype=np.uint8) * 255
        
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                cell_walls = walls[idx]
                
                cell_y, cell_x = i * 50, j * 50
                # Draw the cell
                cv2.rectangle(debug_grid, (cell_x, cell_y), (cell_x + 50, cell_y + 50), (200, 200, 200), 1)
                # Draw consistent walls
                if cell_walls[0]:  # Top
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x + 50, cell_y), (0, 0, 0), 2)
                if cell_walls[1]:  # Right
                    cv2.line(debug_grid, (cell_x + 50, cell_y), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                if cell_walls[2]:  # Bottom
                    cv2.line(debug_grid, (cell_x, cell_y + 50), (cell_x + 50, cell_y + 50), (0, 0, 0), 2)
                if cell_walls[3]:  # Left
                    cv2.line(debug_grid, (cell_x, cell_y), (cell_x, cell_y + 50), (0, 0, 0), 2)
                
                # Add labels
                cv2.putText(debug_grid, f"{i},{j}", (cell_x + 15, cell_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imwrite("walls_debug_consistent.jpg", debug_grid)
        cv2.imwrite("walls_debug.jpg", debug_grid)
    
    return walls

def generate_output_file(grid_size, walls, output_path):
    """Generate the output text file."""
    with open(output_path, 'w') as f:
        # Write grid size
        f.write(f"{grid_size}\n")
        
        # Write wall data for each cell
        for wall in walls:
            f.write(f"{''.join(map(str, wall))}\n")

def visualize_detected_maze(grid_size, walls, output_path):
    """Generate a visualization of the detected maze."""
    # Create an image to visualize the maze
    cell_size = 30
    img_size = grid_size * cell_size
    maze_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    for idx, wall in enumerate(walls):
        i = idx // grid_size  # Row
        j = idx % grid_size   # Column
        
        y, x = i * cell_size, j * cell_size
        
        # Draw detected walls
        if wall[0]:  # Top
            cv2.line(maze_img, (x, y), (x + cell_size, y), (0, 0, 0), 2)
        if wall[1]:  # Right
            cv2.line(maze_img, (x + cell_size, y), (x + cell_size, y + cell_size), (0, 0, 0), 2)
        if wall[2]:  # Bottom
            cv2.line(maze_img, (x, y + cell_size), (x + cell_size, y + cell_size), (0, 0, 0), 2)
        if wall[3]:  # Left
            cv2.line(maze_img, (x, y), (x, y + cell_size), (0, 0, 0), 2)
    
    # Save the visualization
    vis_path = output_path.replace('.txt', '_visualization.jpg')
    cv2.imwrite(vis_path, maze_img)
    print(f"Maze visualization saved to {vis_path}")

def process_maze(image_path, debug=False):
    """Process a maze image and generate the output text file."""
    # Create debug directory if needed
    if debug:
        os.makedirs("debug", exist_ok=True)
    
    # Preprocess image
    original_img, binary_img, gray_img = preprocess_image(image_path)
    
    if debug:
        cv2.imwrite("debug/binary_image.jpg", binary_img)
    
    # Detect grid size using improved method
    grid_size, maze_img, maze_bounds = detect_grid_size_improved(binary_img, gray_img, debug=debug)
    
    # Extract cells
    cells, debug_img = extract_cells(maze_img, grid_size, original_img, maze_bounds)
    output_path = "maze.txt"

    # Save debug image
    if debug:
        debug_path = output_path.replace('.txt', '_debug.jpg')
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved to {debug_path}")
    
    # Analyze cells
    walls = analyze_all_cells(cells, grid_size, debug=debug)
    
    # Generate output file
    generate_output_file(grid_size, walls, output_path)
    print(f"Output file saved to {output_path}")
    
    # Create visualization
    #visualize_detected_maze(grid_size, walls, output_path)


def verify_grid_size(binary_img, detected_size):
    """Verify if the detected grid size makes sense based on image analysis."""
    h, w = binary_img.shape
    cell_h, cell_w = h // detected_size, w // detected_size
    
    # Check if cells are roughly square
    cell_ratio = cell_w / cell_h
    if not 0.8 <= cell_ratio <= 1.2:
        return False
    
    # Check if number of edges matches expected grid size
    edges = cv2.Canny(binary_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                           minLineLength=min(h,w)//16, maxLineGap=20)
    
    if lines is None:
        return False
    
    expected_lines = (detected_size + 1) * 2  # Horizontal + vertical lines
    actual_lines = len(lines)
    
    return abs(actual_lines - expected_lines) <= detected_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a maze image to a text file')
    
    parser.add_argument('--image_path', type=str, help='path to image')
    args = parser.parse_args()

    image_path = args.image_path
    
    process_maze(image_path)