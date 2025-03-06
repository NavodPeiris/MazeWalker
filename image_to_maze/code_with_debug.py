import cv2
import numpy as np
import os
import json

def detect_maze_corners(image):
    """Detect the four corners of a maze in an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Try to approximate with 4 points
    for approx_factor in [0.02, 0.05]:
        approx = cv2.approxPolyDP(largest_contour, approx_factor * perimeter, True)
        if len(approx) == 4:
            break
    
    # If still not 4 points, return None
    if len(approx) != 4:
        return None
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left will have smallest sum, bottom-right largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left
    rect[2] = pts[np.argmax(s)]  # Bottom-right
    
    # Top-right will have smallest difference, bottom-left largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left
    
    return rect

def perspective_correct_maze(image, corners=None, target_size=1000):
    """Apply perspective correction to make the maze rectangular."""
    if corners is None:
        corners = detect_maze_corners(image)
        if corners is None:
            return image
    
    # Define destination points (square)
    dst_pts = np.array([
        [0, 0],
        [target_size - 1, 0],
        [target_size - 1, target_size - 1],
        [0, target_size - 1]
    ], dtype="float32")
    
    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(image, M, (target_size, target_size), 
                               flags=cv2.INTER_LINEAR)
    
    return warped

def preprocess_image(image_path):
    """Load and preprocess the maze image."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Quick resize to detect grid size
    quick_resize = 1000
    height, width = img.shape[:2]
    scale = quick_resize / max(height, width)
    small_img = cv2.resize(img, None, fx=scale, fy=scale)

    # Detect grid size from the small image
    small_corrected = perspective_correct_maze(small_img)
    gray_small = cv2.cvtColor(small_corrected, cv2.COLOR_BGR2GRAY)
    blurred_small = cv2.GaussianBlur(gray_small, (5, 5), 0)
    binary_small = cv2.adaptiveThreshold(blurred_small, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

    estimated_grid_size, _, _ = detect_grid_size(binary_small, gray_small)

    # Choose adaptive resolution based on estimated grid size
    max_dimension = {
        9: 1000,
        16: 1300,
        21: 1500
    }.get(estimated_grid_size, 1000)  # Default to 1000

    # Resize the full-resolution image accordingly
    scale = max_dimension / max(height, width)
    corrected_img = cv2.resize(img, None, fx=scale, fy=scale)
    corrected_img = perspective_correct_maze(corrected_img, target_size=max_dimension)
    
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return corrected_img, binary, gray

def detect_grid_size(binary_img, gray_img, debug=False):
    """Detect the grid size of the maze (9x9, 16x16, or 21x21)."""
    # Get image dimensions
    h, w = binary_img.shape
    
    # Find maze boundary
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise ValueError("No contours found in the image")
    
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Crop to maze boundaries
    maze_binary = binary_img[y:y+h, x:x+w].copy()
    maze_gray = gray_img[y:y+h, x:x+w].copy()
    
    # Use edge detection for line detection
    edges = cv2.Canny(maze_binary, 50, 150, apertureSize=3)
    
    # Detect lines
    min_line_length = min(w, h) // 22
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, 
                           minLineLength=min_line_length, maxLineGap=10)
    
    if lines is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                               minLineLength=min_line_length//2, maxLineGap=30)
        if lines is None:
            raise ValueError("Could not detect grid lines in the maze")
    
    # Draw debug image if requested
    if debug:
        debug_img = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite("outputs/detected_lines.jpg", debug_img)
    
    # Categorize lines as horizontal or vertical
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:  # Horizontal lines
            horizontal_lines.append((y1 + y2) // 2)
        elif 60 < angle < 120:  # Vertical lines
            vertical_lines.append((x1 + x2) // 2)
    
    # Group similar coordinates
    horizontal_positions = group_coordinates(horizontal_lines, min(w, h) // 30)
    vertical_positions = group_coordinates(vertical_lines, min(w, h) // 30)
    
    if debug:
        debug_img2 = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        for y in horizontal_positions:
            cv2.line(debug_img2, (0, y), (w, y), (0, 255, 0), 1)
        for x in vertical_positions:
            cv2.line(debug_img2, (x, 0), (x, h), (255, 0, 0), 1)
        cv2.imwrite("outputs/grouped_lines.jpg", debug_img2)
    
    # Calculate cell counts
    num_cells_x = len(vertical_positions) + 1
    num_cells_y = len(horizontal_positions) + 1
    
    # Determine grid size by comparing with standard sizes
    potential_sizes = [9, 16, 21]
    best_score = float('-inf')
    optimal_grid_size = None
    
    for size in potential_sizes:
        approx_cell_width = w / size
        approx_cell_height = h / size
        score = 0
        
        # Score horizontal lines
        for i in range(1, size):
            expected_y = int(y + i * approx_cell_height)
            closest_diff = min([abs(expected_y - pos) for pos in horizontal_positions], default=w)
            score -= closest_diff
        
        # Score vertical lines
        for i in range(1, size):
            expected_x = int(x + i * approx_cell_width)
            closest_diff = min([abs(expected_x - pos) for pos in vertical_positions], default=h)
            score -= closest_diff
        
        if score > best_score:
            best_score = score
            optimal_grid_size = size
    
    # Choose final grid size
    final_grid_size = optimal_grid_size
    
    # Adjust based on cell counts if needed
    if num_cells_x > 6 or num_cells_y > 6:
        if num_cells_x <= 12 or num_cells_y <= 12:
            final_grid_size = 9
        elif num_cells_x <= 19 or num_cells_y <= 19:
            final_grid_size = 16
        else:
            final_grid_size = 21
    
    # Match to closest standard size
    if final_grid_size not in potential_sizes:
        diffs = [(abs(final_grid_size - size), size) for size in potential_sizes]
        final_grid_size = min(diffs)[1]
    
    print(f"Detected grid dimensions: {num_cells_x}x{num_cells_y}")
    print(f"Determined grid size: {final_grid_size}x{final_grid_size}")
    
    # Draw debug grid if requested
    if debug:
        debug_img3 = cv2.cvtColor(maze_binary.copy(), cv2.COLOR_GRAY2BGR)
        cell_size = w / final_grid_size
        for i in range(final_grid_size + 1):
            y_pos = int(i * cell_size)
            x_pos = int(i * cell_size)
            cv2.line(debug_img3, (0, y_pos), (w, y_pos), (0, 255, 255), 1)
            cv2.line(debug_img3, (x_pos, 0), (x_pos, h), (0, 255, 255), 1)
        cv2.imwrite("outputs/determined_grid.jpg", debug_img3)
    
    return final_grid_size, maze_binary, (x, y, w, h)

def group_coordinates(coords, threshold=None):
    """Group similar coordinates to handle slight variations."""
    if not coords:
        return []
    
    coords = sorted(coords)
    groups = [[coords[0]]]
    
    for coord in coords[1:]:
        if coord - groups[-1][-1] <= threshold:
            groups[-1].append(coord)
        else:
            groups.append([coord])
    
    return [sum(group) // len(group) for group in groups]

def extract_cells(maze_img, grid_size, original_img, maze_bounds):
    """Extract individual cells from the maze image."""
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
    """Analyze a single cell to detect its walls."""
    h, w = cell.shape
    
    # Define wall detection parameters
    wall_thickness = max(2, min(h, w) // 10)
    threshold_percentage = 0.4
    
    # Check walls in each direction
    regions = {
        'top': cell[0:wall_thickness, :],
        'right': cell[:, w-wall_thickness:w],
        'bottom': cell[h-wall_thickness:h, :],
        'left': cell[:, 0:wall_thickness]
    }
    
    walls = []
    for region in ['top', 'right', 'bottom', 'left']:
        has_wall = 1 if np.sum(regions[region]) / (regions[region].size * 255) > threshold_percentage else 0
        walls.append(has_wall)
    
    # Enforce boundary walls
    if i == 0:  # Top row
        walls[0] = 1
    if j == grid_size - 1:  # Rightmost column
        walls[1] = 1
    if i == grid_size - 1:  # Bottom row
        walls[2] = 1
    if j == 0:  # Leftmost column
        walls[3] = 1
    
    # Create debug visualization if requested
    if debug:
        os.makedirs("outputs/cell_debug", exist_ok=True)
        debug_cell = cv2.cvtColor(cell.copy(), cv2.COLOR_GRAY2BGR)
        
        # Draw detected walls
        wall_positions = [
            ((0, wall_thickness//2), (w, wall_thickness//2)),  # Top
            ((w-wall_thickness//2, 0), (w-wall_thickness//2, h)),  # Right
            ((0, h-wall_thickness//2), (w, h-wall_thickness//2)),  # Bottom
            ((wall_thickness//2, 0), (wall_thickness//2, h))   # Left
        ]
        
        for idx, has_wall in enumerate(walls):
            if has_wall:
                start, end = wall_positions[idx]
                cv2.line(debug_cell, start, end, (0, 0, 255), 1)
        
        cv2.imwrite(f"outputs/cell_debug/cell_{i}_{j}.png", debug_cell)
    
    return walls

def ensure_wall_consistency(walls, grid_size):
    """Ensure that shared walls between adjacent cells are consistent."""
    # Convert walls to 2D grid for easier adjacency checks
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
            # Check right-left wall consistency
            if j < grid_size - 1:
                if grid_walls[i][j][1] == 1:  # Right wall exists
                    grid_walls[i][j+1][3] = 1  # Set left wall of next cell
                elif grid_walls[i][j+1][3] == 1:  # Left wall of next cell exists
                    grid_walls[i][j][1] = 1  # Set right wall
            
            # Check bottom-top wall consistency
            if i < grid_size - 1:
                if grid_walls[i][j][2] == 1:  # Bottom wall exists
                    grid_walls[i+1][j][0] = 1  # Set top wall of cell below
                elif grid_walls[i+1][j][0] == 1:  # Top wall of cell below exists
                    grid_walls[i][j][2] = 1  # Set bottom wall
    
    # Convert back to flat list
    consistent_walls = []
    for i in range(grid_size):
        for j in range(grid_size):
            consistent_walls.append(grid_walls[i][j])
    
    return consistent_walls

def analyze_all_cells(cells, grid_size, debug=False):
    """Analyze all cells in the maze to detect walls."""
    walls = []
    
    # Create debug grid if requested
    if debug:
        debug_grid = np.ones((grid_size * 50, grid_size * 50, 3), dtype=np.uint8) * 255
    
    # Analyze each cell
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
                wall_positions = [
                    ((cell_x, cell_y), (cell_x + 50, cell_y)),  # Top
                    ((cell_x + 50, cell_y), (cell_x + 50, cell_y + 50)),  # Right
                    ((cell_x, cell_y + 50), (cell_x + 50, cell_y + 50)),  # Bottom
                    ((cell_x, cell_y), (cell_x, cell_y + 50))   # Left
                ]
                
                for idx, has_wall in enumerate(cell_walls):
                    if has_wall:
                        start, end = wall_positions[idx]
                        cv2.line(debug_grid, start, end, (0, 0, 0), 2)
                
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
                
                # Draw walls
                wall_positions = [
                    ((cell_x, cell_y), (cell_x + 50, cell_y)),  # Top
                    ((cell_x + 50, cell_y), (cell_x + 50, cell_y + 50)),  # Right
                    ((cell_x, cell_y + 50), (cell_x + 50, cell_y + 50)),  # Bottom
                    ((cell_x, cell_y), (cell_x, cell_y + 50))   # Left
                ]
                
                for idx, has_wall in enumerate(cell_walls):
                    if has_wall:
                        start, end = wall_positions[idx]
                        cv2.line(debug_grid, start, end, (0, 0, 0), 2)
                
                # Add labels
                cv2.putText(debug_grid, f"{i},{j}", (cell_x + 15, cell_y + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imwrite("outputs/walls_debug_consistent.jpg", debug_grid)
    
    return walls

def generate_output_file(grid_size, walls, output_path):
    """Generate the output text file with maze data."""
    with open(output_path, 'w') as f:
        # Write grid size
        f.write(f"{grid_size}\n")
        
        # Write wall data for each cell
        for wall in walls:
            f.write(f"{''.join(map(str, wall))}\n")

def visualize_detected_maze(grid_size, walls, output_path):
    """Create a visualization of the detected maze."""
    cell_size = 30
    img_size = grid_size * cell_size
    maze_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    for idx, wall in enumerate(walls):
        i = idx // grid_size  # Row
        j = idx % grid_size   # Column
        
        y, x = i * cell_size, j * cell_size
        
        # Draw walls
        wall_positions = [
            ((x, y), (x + cell_size, y)),  # Top
            ((x + cell_size, y), (x + cell_size, y + cell_size)),  # Right
            ((x, y + cell_size), (x + cell_size, y + cell_size)),  # Bottom
            ((x, y), (x, y + cell_size))   # Left
        ]
        
        for idx, has_wall in enumerate(wall):
            if has_wall:
                start, end = wall_positions[idx]
                cv2.line(maze_img, start, end, (0, 0, 0), 2)
    
    # Save the visualization
    vis_path = output_path.replace('.txt', '_visualization.jpg')
    cv2.imwrite(vis_path, maze_img)
    print(f"Maze visualization saved to {vis_path}")

def process_maze(image_path, output_path, debug=False):
    """Main function to process a maze image and generate output files."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    # Preprocess image
    original_img, binary_img, gray_img = preprocess_image(image_path)
    
    # Detect grid size
    grid_size, maze_img, maze_bounds = detect_grid_size(binary_img, gray_img, debug=debug)
    
    # Extract cells
    cells, debug_img = extract_cells(maze_img, grid_size, original_img, maze_bounds)
    
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
    visualize_detected_maze(grid_size, walls, output_path)

if __name__ == "__main__":
    # Load configuration from config.json
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    image_path = config.get('image_path')
    output_path = config.get('output_path')
    debug = config.get('debug')
    
    process_maze(image_path, output_path, debug)