import cv2
import pandas as pd
import numpy as np
import argparse
import os

class ConeOverlay:
    def __init__(self, video_path, csv1_path, csv2_path, cone_angle=45, cone_length=100, 
                 text_position=(50, 50), text_size=1.0, text_color=(255, 255, 255)):
        """
        Initialize the cone overlay visualization.
        
        Args:
            video_path: Path to the input video file
            csv1_path: Path to CSV file for rat 1 (cone source)
            csv2_path: Path to CSV file for rat 2 (intersection target)
            cone_angle: Half-angle of the cone in degrees (total cone angle = 2 * cone_angle)
            cone_length: Length of the cone vectors in pixels
            text_position: (x, y) tuple for text position when intersection occurs
            text_size: Size of the text (default: 1.0)
            text_color: Color of the text in BGR format (default: white)
        """
        self.video_path = video_path
        self.csv1_path = csv1_path
        self.csv2_path = csv2_path
        self.cone_angle = np.radians(cone_angle)  # Convert to radians
        self.cone_length = cone_length
        
        # Colors for cone (BGR format for OpenCV)
        self.color_normal = (0, 255, 0)      # Green when no intersection
        self.color_intersect = (0, 0, 255)   # Red when intersecting
        self.alpha = 0.3  # Transparency for filled area
        
        # Text settings
        self.text_position = text_position
        self.text_size = text_size
        self.text_color = text_color
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_thickness = 2
        
        # Load tracking data
        self.load_tracking_data()
        
    def load_tracking_data(self):
        """Load and process DeepLabCut CSV files."""
        print("Loading tracking data...")
        
        # Load CSV files
        self.df1 = pd.read_csv(self.csv1_path)
        self.df2 = pd.read_csv(self.csv2_path)
        
        print("CSV1 columns:", list(self.df1.columns))
        print("CSV2 columns:", list(self.df2.columns))
        print(f"CSV1 shape: {self.df1.shape}")
        print(f"CSV2 shape: {self.df2.shape}")
        
        # Get column names for head and body center coordinates
        # Adjust these based on your DeepLabCut model's body part names
        self.head_parts = ['head_1', 'head_2', 'head_3', 'head', 'Head', 'HEAD']  # Possible head part names
        self.body_parts = ['bodyCenter_1', 'bodyCenter_2', 'bodyCenter_3', 'bodyCenter', 'body_center', 'center']  # Possible body center names
        
        self.rat1_head_cols = self.find_bodypart_columns(self.df1, self.head_parts)
        self.rat1_body_cols = self.find_bodypart_columns(self.df1, self.body_parts)
        self.rat2_body_cols = self.find_bodypart_columns(self.df2, self.body_parts)
        
        print(f"Rat 1 head columns: {self.rat1_head_cols}")
        print(f"Rat 1 body columns: {self.rat1_body_cols}")
        print(f"Rat 2 body columns: {self.rat2_body_cols}")
        
    def find_bodypart_columns(self, df, part_names):
        """Find column names for specific body parts with _1 (x) and _2 (y) suffix convention."""
        columns = df.columns.tolist()
        
        # Look for bodypart_1 and bodypart_2 pattern (x and y coordinates)
        for part in part_names:
            # Remove any existing suffix to get base name
            base_part = part.replace('_1', '').replace('_2', '').replace('_3', '')
            
            x_col = f"{base_part}_1"  # x coordinate
            y_col = f"{base_part}_2"  # y coordinate
            
            if x_col in columns and y_col in columns:
                print(f"Found columns for {base_part}: {x_col}, {y_col}")
                return (x_col, y_col)
        
        # If exact match not found, try partial matching with _1 and _2 suffix
        for part in part_names:
            base_part = part.replace('_1', '').replace('_2', '').replace('_3', '')
            
            for col in columns:
                if base_part.lower() in col.lower() and col.endswith('_1'):
                    # Found x coordinate, look for corresponding y coordinate
                    y_col = col.replace('_1', '_2')
                    if y_col in columns:
                        print(f"Found columns for {base_part} (partial match): {col}, {y_col}")
                        return (col, y_col)
        
        # If still not found, try to find any columns with _1 and _2 pattern
        x_cols = [col for col in columns if col.endswith('_1')]
        y_cols = [col for col in columns if col.endswith('_2')]
        
        if x_cols and y_cols:
            # Try to find matching pairs
            for x_col in x_cols:
                base_name = x_col.replace('_1', '')
                y_col = f"{base_name}_2"
                if y_col in y_cols:
                    print(f"Warning: Using first available _1,_2 pair: {x_col}, {y_col}")
                    return (x_col, y_col)
        
        raise ValueError(f"Could not find coordinate columns for parts: {part_names}. Available columns: {columns}")
    
    def get_cone_direction(self, head_pos, body_pos):
        """Calculate the direction vector from body to head (forward direction)."""
        if head_pos is None or body_pos is None:
            return np.array([1, 0])  # Default direction
        
        # Direction should be from body toward head for forward-facing cone
        direction = np.array(head_pos) - np.array(body_pos)
        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([1, 0])  # Default direction if positions are the same
        return direction / norm
    
    def get_cone_vectors(self, head_pos, direction):
        """Calculate the two vectors defining the cone edges."""
        if head_pos is None:
            return None, None
        
        # Create rotation matrices for +/- cone_angle
        cos_angle = np.cos(self.cone_angle)
        sin_angle = np.sin(self.cone_angle)
        
        # Rotation matrix for positive angle
        rot_pos = np.array([[cos_angle, -sin_angle],
                           [sin_angle, cos_angle]])
        
        # Rotation matrix for negative angle
        rot_neg = np.array([[cos_angle, sin_angle],
                           [-sin_angle, cos_angle]])
        
        # Apply rotations to direction vector and scale by cone length
        vec1 = head_pos + np.dot(rot_pos, direction) * self.cone_length
        vec2 = head_pos + np.dot(rot_neg, direction) * self.cone_length
        
        return vec1, vec2
    
    def point_in_cone(self, point, head_pos, vec1, vec2):
        """Check if a point is inside the cone defined by head_pos, vec1, vec2."""
        if point is None or head_pos is None or vec1 is None or vec2 is None:
            return False
        
        # Vectors from head to cone edges
        edge1 = vec1 - head_pos
        edge2 = vec2 - head_pos
        point_vec = point - head_pos
        
        # Check if point is within the cone length first
        point_dist = np.linalg.norm(point_vec)
        if point_dist == 0 or point_dist > self.cone_length:
            return False
        
        # Use dot product method to check if point is within cone angle
        # Normalize vectors
        edge1_norm = edge1 / np.linalg.norm(edge1)
        edge2_norm = edge2 / np.linalg.norm(edge2)
        point_norm = point_vec / point_dist
        
        # Calculate the cone's center direction (bisector of the two edges)
        center_dir = (edge1_norm + edge2_norm) / 2
        center_dir = center_dir / np.linalg.norm(center_dir)
        
        # Check if point is within the cone angle from center direction
        dot_with_center = np.dot(point_norm, center_dir)
        
        # Also check using cross products for more robust detection
        cross1 = np.cross(edge1, point_vec)
        cross2 = np.cross(point_vec, edge2)
        
        # The point is inside if it's on the correct side of both edges
        # and within a reasonable angle from the center direction
        angle_threshold = np.cos(self.cone_angle)  # Convert angle to cosine threshold
        
        return (cross1 * cross2 >= 0) and (dot_with_center >= angle_threshold * 0.7)
    
    def draw_text(self, frame, text, position=None):
        """Draw text on the frame at specified position."""
        if position is None:
            position = self.text_position
        
        # Add background rectangle for better text visibility
        text_size = cv2.getTextSize(text, self.text_font, self.text_size, self.text_thickness)[0]
        
        # Background rectangle coordinates
        bg_x1 = position[0] - 5
        bg_y1 = position[1] - text_size[1] - 10
        bg_x2 = position[0] + text_size[0] + 5
        bg_y2 = position[1] + 5
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        cv2.putText(frame, text, position, self.text_font, self.text_size, 
                   self.text_color, self.text_thickness, cv2.LINE_AA)
        
        return frame
    
    def draw_cone(self, frame, head_pos, vec1, vec2, intersecting):
        """Draw the cone on the frame."""
        if head_pos is None or vec1 is None or vec2 is None:
            return frame
        
        color = self.color_intersect if intersecting else self.color_normal
        
        # Convert coordinates to integers
        head_pt = tuple(map(int, head_pos))
        vec1_pt = tuple(map(int, vec1))
        vec2_pt = tuple(map(int, vec2))
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Draw filled triangle (cone area)
        triangle_pts = np.array([head_pt, vec1_pt, vec2_pt], dtype=np.int32)
        cv2.fillPoly(overlay, [triangle_pts], color)
        
        # Draw cone edges
        cv2.line(overlay, head_pt, vec1_pt, color, 2)
        cv2.line(overlay, head_pt, vec2_pt, color, 2)
        cv2.line(overlay, vec1_pt, vec2_pt, color, 2)
        
        # Draw head position
        cv2.circle(overlay, head_pt, 5, (255, 255, 255), -1)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(frame, 1 - self.alpha, overlay, self.alpha, 0)
        
        # Add text if intersecting
        if intersecting:
            frame = self.draw_text(frame, "Looking at partner")
        
        return frame
    
    def get_coordinates(self, df, cols, frame_idx):
        """Extract coordinates from dataframe for given frame."""
        if frame_idx >= len(df):
            return None
        
        try:
            x = df.iloc[frame_idx][cols[0]]
            y = df.iloc[frame_idx][cols[1]]
            
            # Check if coordinates are valid (not NaN and within reasonable range)
            if pd.isna(x) or pd.isna(y):
                return None
            
            return np.array([float(x), float(y)])
        except Exception as e:
            print(f"Error getting coordinates for frame {frame_idx}: {e}")
            return None
    
    def process_video(self, output_path=None):
        """Process the video and add cone overlay."""
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        print(f"Video duration: {total_frames/fps:.2f} seconds")
        
        # Check CSV data length
        csv_frames = min(len(self.df1), len(self.df2))
        print(f"CSV data frames: {csv_frames}")
        
        # Determine if we need to handle frame rate mismatch
        frame_ratio = total_frames / csv_frames if csv_frames > 0 else 1
        print(f"Video to CSV frame ratio: {frame_ratio:.2f}")
        
        # Set up output video writer
        if output_path is None:
            base_name = os.path.splitext(self.video_path)[0]
            output_path = f"{base_name}_cone_overlay.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"Could not create output video writer for: {output_path}")
        
        frame_idx = 0
        csv_frame_idx = 0
        intersection_count = 0
        frames_written = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("No more frames to read from video")
                    break
                
                # Calculate corresponding CSV frame index
                # This handles cases where video and CSV have different frame rates
                if frame_ratio > 1:
                    # More video frames than CSV frames - interpolate or use nearest
                    csv_frame_idx = min(int(frame_idx / frame_ratio), csv_frames - 1)
                else:
                    # Equal or fewer video frames than CSV frames
                    csv_frame_idx = min(frame_idx, csv_frames - 1)
                
                # Get coordinates for current CSV frame
                rat1_head = self.get_coordinates(self.df1, self.rat1_head_cols, csv_frame_idx)
                rat1_body = self.get_coordinates(self.df1, self.rat1_body_cols, csv_frame_idx)
                rat2_body = self.get_coordinates(self.df2, self.rat2_body_cols, csv_frame_idx)
                
                if rat1_head is not None and rat1_body is not None:
                    # Calculate cone direction and vectors
                    direction = self.get_cone_direction(rat1_head, rat1_body)
                    vec1, vec2 = self.get_cone_vectors(rat1_head, direction)
                    
                    # Check if rat2 body is in cone
                    intersecting = self.point_in_cone(rat2_body, rat1_head, vec1, vec2)
                    
                    if intersecting:
                        intersection_count += 1
                    
                    # Debug output for first few intersections
                    if intersecting and intersection_count <= 5:
                        print(f"Frame {frame_idx} (CSV frame {csv_frame_idx}): INTERSECTION DETECTED!")
                        print(f"  Rat1 head: {rat1_head}")
                        print(f"  Rat1 body: {rat1_body}")
                        print(f"  Rat2 body: {rat2_body}")
                    
                    # Draw cone
                    frame = self.draw_cone(frame, rat1_head, vec1, vec2, intersecting)
                    
                    # Also draw rat2 position as a circle for debugging
                    if rat2_body is not None:
                        color = (255, 0, 255) if intersecting else (255, 255, 0)  # Magenta if intersecting, cyan otherwise
                        cv2.circle(frame, tuple(map(int, rat2_body)), 8, color, -1)
                
                # Write frame to output video
                out.write(frame)
                frames_written += 1
                
                frame_idx += 1
                
                # Progress indicator
                if frame_idx % 100 == 0:
                    print(f"Processed {frame_idx} / {total_frames} frames, Written: {frames_written}, Intersections: {intersection_count}")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"Output video saved to: {output_path}")
        print(f"Total frames processed: {frame_idx}")
        print(f"Total frames written: {frames_written}")
        print(f"Total intersections detected: {intersection_count}")
        
        # Verify output video
        verify_cap = cv2.VideoCapture(output_path)
        if verify_cap.isOpened():
            output_frames = int(verify_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = verify_cap.get(cv2.CAP_PROP_FPS)
            output_duration = output_frames / output_fps if output_fps > 0 else 0
            print(f"Output video verification: {output_frames} frames, {output_fps} FPS, {output_duration:.2f} seconds")
            verify_cap.release()
        
        return output_path

def process_cone_overlay(video_path, csv1_path, csv2_path, cone_angle=45, cone_length=100, 
                        output_path=None, text_position=(50, 50), text_size=1.0, text_color=(255, 255, 255)):
    """
    Main function to process cone overlay visualization.
    
    Args:
        video_path: Path to input video file (.mp4)
        csv1_path: Path to CSV file for rat 1 (cone source)
        csv2_path: Path to CSV file for rat 2 (intersection target)
        cone_angle: Half-angle of cone in degrees (default: 45)
        cone_length: Length of cone in pixels (default: 100)
        output_path: Output video path (optional)
        text_position: (x, y) tuple for text position when intersection occurs (default: (50, 50))
        text_size: Size of the text (default: 1.0)
        text_color: Color of the text in BGR format (default: white)
    
    Returns:
        Path to the output video file
    """
    # Create cone overlay processor
    processor = ConeOverlay(
        video_path,
        csv1_path,
        csv2_path,
        cone_angle=cone_angle,
        cone_length=cone_length,
        text_position=text_position,
        text_size=text_size,
        text_color=text_color
    )
    
    # Process video
    output_path = processor.process_video(output_path)
    print(f"Processing complete! Output saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Overlay cone visualization on DeepLabCut tracked video")
    parser.add_argument("video", help="Path to input video file (.mp4)")
    parser.add_argument("csv1", help="Path to CSV file for rat 1 (cone source)")
    parser.add_argument("csv2", help="Path to CSV file for rat 2 (intersection target)")
    parser.add_argument("--cone-angle", type=float, default=45, help="Half-angle of cone in degrees (default: 45)")
    parser.add_argument("--cone-length", type=int, default=100, help="Length of cone in pixels (default: 100)")
    parser.add_argument("--output", help="Output video path (default: input_cone_overlay.mp4)")
    parser.add_argument("--text-x", type=int, default=50, help="X position for intersection text (default: 50)")
    parser.add_argument("--text-y", type=int, default=50, help="Y position for intersection text (default: 50)")
    parser.add_argument("--text-size", type=float, default=1.0, help="Text size (default: 1.0)")
    
    args = parser.parse_args()
    
    # Use the main processing function
    process_cone_overlay(
        args.video,
        args.csv1,
        args.csv2,
        cone_angle=args.cone_angle,
        cone_length=args.cone_length,
        output_path=args.output,
        text_position=(args.text_x, args.text_y),
        text_size=args.text_size
    )

if __name__ == "__main__":
    main()

#%% Example usage:
from dlc_cone_fixed_v2 import process_cone_overlay

# Basic usage with default text position
# output_path = process_cone_overlay(
#     video_path="your_video.mp4",
#     csv1_path="rat1.csv",
#     csv2_path="rat2.csv"
# )

# Advanced usage with custom text settings
output_path = process_cone_overlay(
    video_path="E:/Jadhav lab data/Behavior/CohortAS1/Social W/50%/08-24-2023/log08-24-2023(1-XFN1-XFN3).1.mp4",
    csv1_path="E:/Jadhav lab data/Behavior/CohortAS1/Social W/50%/08-24-2023/log08-24-2023(1-XFN1-XFN3)-Rat1_corrected-AllTracking.csv",
    csv2_path="E:/Jadhav lab data/Behavior/CohortAS1/Social W/50%/08-24-2023/log08-24-2023(1-XFN1-XFN3)-Rat2_corrected-AllTracking.csv",
    cone_angle=10,           # Half-angle in degrees
    cone_length=400,         # Length in pixels
    output_path="my_output2.mp4",
    text_position=(100, 80), # Custom text position (x, y)
    text_size=1.5,           # Larger text
    text_color=(0, 255, 255) # Yellow text (BGR format)
)

print(f"Video saved to: {output_path}")