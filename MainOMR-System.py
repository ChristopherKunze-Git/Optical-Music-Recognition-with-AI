import numpy as np
from scipy.signal import find_peaks
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from midiutil import MIDIFile
import re

class Staff:
    def __init__(self, lines, center, upper_bound, lower_bound, unit_size, track=1, group=1):
        self.staffline_positions = lines
        self.center = center
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.unit_size = unit_size
        self.track = track
        self.group = group
        self.key_signature = {}
        self.bars = []
        
        # Pitch arrays based on track (treble or bass clef)
        if track == 1:  # Treble clef
            self.staffline_pitches = ['F5', 'D5', 'B4', 'G4', 'E4']  # Top to bottom
            self.ghost_pitches_above = ['C6', 'A5']  # Top to bottom
            self.ghost_pitches_below = ['C4', 'A3']  # Top to bottom
            
            # Initialize space pitches
            self.staff_spaces = ['E5', 'C5', 'A4', 'F4']  # Top to bottom
            self.ghost_spaces_above = ['B5']
            self.ghost_spaces_below = ['B3']
            self.transition_space_above_pitch = 'G5'
            self.transition_space_below_pitch = 'D4'
            
        else:  # Bass clef
            self.staffline_pitches = ['A3', 'F3', 'D3', 'B2', 'G2']  # Top to bottom
            self.ghost_pitches_above = ['E4', 'C4']  # Top to bottom
            self.ghost_pitches_below = ['E2', 'C2']  # Top to bottom
            
            # Initialize space pitches
            self.staff_spaces = ['G3', 'E3', 'C3', 'A2']  # Top to bottom
            self.ghost_spaces_above = ['D4']
            self.ghost_spaces_below = ['D2']
            self.transition_space_above_pitch = 'B3'
            self.transition_space_below_pitch = 'F2'
        
        # Calculate positions
        spacing = lines[1] - lines[0]
        
        # Ghost lines positions
        self.ghost_lines_above = [lines[0] - spacing * 2, lines[0] - spacing]
        self.ghost_lines_below = [lines[-1] + spacing, lines[-1] + spacing * 2]
        
        # Space positions
        self.ghost_space_positions_above = [(self.ghost_lines_above[i] + self.ghost_lines_above[i + 1]) / 2 
                                          for i in range(len(self.ghost_lines_above) - 1)]
        
        self.staff_space_positions = [(self.staffline_positions[i] + self.staffline_positions[i + 1]) / 2 
                                    for i in range(len(self.staffline_positions) - 1)]
        
        self.ghost_space_positions_below = [(self.ghost_lines_below[i] + self.ghost_lines_below[i + 1]) / 2 
                                          for i in range(len(self.ghost_lines_below) - 1)]
        
        # Transition space positions
        self.transition_space_above_pos = (self.ghost_lines_above[-1] + self.staffline_positions[0]) / 2
        self.transition_space_below_pos = (self.staffline_positions[-1] + self.ghost_lines_below[0]) / 2
   
    def add_key_signature(self, pitch, accidental_type):
        """Add an accidental to the key signature"""
        self.key_signature[pitch] = accidental_type
        print(f"Added {accidental_type} to key signature for pitch {pitch}")
        
    def apply_key_signature(self, key_objects):
        """Apply key signature effects to the staff"""
        self.key_signature.clear()
        
        for key_obj in key_objects:
            # Find closest staffline or space to the key symbol
            key_y = (key_obj['y1'] + key_obj['y2']) / 2
            pitch = self.get_pitch_at_position(key_y)
            
            if pitch:
                if key_obj['class'] == 'keyflat':
                    self.key_signature[pitch] = 'flat'
                elif key_obj['class'] == 'keysharp':
                    self.key_signature[pitch] = 'sharp'

    def get_pitch_at_position(self, y_position):
        """
        Get the pitch at a specific vertical position
        
        Args:
            y_position: Vertical position in the image
            
        Returns:
            pitch: String representing the pitch at that position
        """
        # First check if position is on a staffline
        for i, line_pos in enumerate(self.staffline_positions):
            if abs(y_position - line_pos) < self.unit_size / 3:
                return self.staffline_pitches[i]
        
        # Check if position is on a ghost line above
        for i, line_pos in enumerate(self.ghost_lines_above):
            if abs(y_position - line_pos) < self.unit_size / 3:
                return self.ghost_pitches_above[i]
        
        # Check if position is on a ghost line below
        for i, line_pos in enumerate(self.ghost_lines_below):
            if abs(y_position - line_pos) < self.unit_size / 3:
                return self.ghost_pitches_below[i]
        
        # Check if position is in a staff space
        for i, space_pos in enumerate(self.staff_space_positions):
            if abs(y_position - space_pos) < self.unit_size / 2:
                return self.staff_spaces[i]
        
        # Check if position is in a ghost space above
        for i, space_pos in enumerate(self.ghost_space_positions_above):
            if abs(y_position - space_pos) < self.unit_size / 2:
                return self.ghost_spaces_above[i]
        
        # Check if position is in a ghost space below
        for i, space_pos in enumerate(self.ghost_space_positions_below):
            if abs(y_position - space_pos) < self.unit_size / 2:
                return self.ghost_spaces_below[i]
        
        # Check transition spaces
        if abs(y_position - self.transition_space_above_pos) < self.unit_size / 2:
            return self.transition_space_above_pitch
        
        if abs(y_position - self.transition_space_below_pos) < self.unit_size / 2:
            return self.transition_space_below_pitch
        
        # Find the closest position
        all_positions = (
            self.staffline_positions + 
            self.ghost_lines_above + 
            self.ghost_lines_below +
            self.staff_space_positions +
            self.ghost_space_positions_above +
            self.ghost_space_positions_below +
            [self.transition_space_above_pos, self.transition_space_below_pos]
        )
        
        all_pitches = (
            self.staffline_pitches +
            self.ghost_pitches_above +
            self.ghost_pitches_below +
            self.staff_spaces +
            self.ghost_spaces_above +
            self.ghost_spaces_below +
            [self.transition_space_above_pitch, self.transition_space_below_pitch]
        )
        
        # Find the closest position
        distances = [abs(y_position - pos) for pos in all_positions]
        closest_idx = distances.index(min(distances))
        
        print(f"  DEBUG: Closest position to y={y_position:.1f} is {all_positions[closest_idx]:.1f} with pitch {all_pitches[closest_idx]}")
        
        return all_pitches[closest_idx]
    
# Bar class, which represent a measure
class Bar:
    def __init__(self, start_x, end_x, staff):
        self.start_x = start_x
        self.end_x = end_x
        self.staff = staff
        self.notes = []
        self.accidentals = {}  # Storing accidentals active in this bar
                
    def add_note(self, note):
        self.notes.append(note)
        self.notes.sort(key=lambda x: x['x'])
               
    def apply_accidental(self, accidental_obj):
        """Apply an accidental to the bar at a specific vertical position"""
        acc_y = (accidental_obj['y1'] + accidental_obj['y2']) / 2
        pitch = self.staff.get_pitch_at_position(acc_y)
        
        if pitch:
            if accidental_obj['class'] == 'accidentalflat':
                self.accidentals[pitch] = 'flat'
            elif accidental_obj['class'] == 'accidentalsharp':
                self.accidentals[pitch] = 'sharp'
            elif accidental_obj['class'] == 'accidentalnatural':
                self.accidentals[pitch] = 'natural'
                
    def get_note_pitch(self, note):
        """Get the actual pitch including any active accidentals"""
        base_pitch = note['pitch']
        
        # Check for accidental in this bar
        if base_pitch in self.accidentals:
            return f"{base_pitch}{self.accidentals[base_pitch]}"
        
        # Check for key signature
        if base_pitch in self.staff.key_signature:
            return f"{base_pitch}{self.staff.key_signature[base_pitch]}"
        
        return base_pitch       
    
def extract_stafflines(model, image_path, chunk_size=256, visualize=False):
    """Extract stafflines from a binary music score image."""
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = img.shape
    
    print(f"Processing image of size {original_height}x{original_width}")
    
    predictions = np.zeros((original_height, original_width))
    
    chunks = []
    chunk_preds = []
    
    # Process image in chunks
    for y in range(0, original_height, chunk_size):
        for x in range(0, original_width, chunk_size):
            # Extract chunk
            chunk = img[y:min(y+chunk_size, original_height), 
                       x:min(x+chunk_size, original_width)]
            
            chunks.append(chunk.copy())
            
            padded = np.zeros((chunk_size, chunk_size))
            padded[:chunk.shape[0], :chunk.shape[1]] = chunk
            
            padded = padded / 255.0
            padded = np.expand_dims(padded, axis=[0, -1])
            
            # Predict on chunk
            pred = model.predict(padded, verbose=0)
            pred = (pred > 0.5).astype(np.uint8)
            pred = np.squeeze(pred)
            
            # Store prediction
            chunk_preds.append(pred[:chunk.shape[0], :chunk.shape[1]])
            
            # Place prediction to the correct position
            predictions[y:min(y+chunk_size, original_height),
                      x:min(x+chunk_size, original_width)] = \
                pred[:min(original_height-y, chunk_size),
                     :min(original_width-x, chunk_size)]
    
    print("\nStaffline prediction complete")
    
    if visualize:
        plt.figure(figsize=(30,10))
        plt.imshow(predictions, cmap='gray')
        plt.title("Predicted Stafflines")
        plt.show()
    
    staffs = []
    
    # Use entire image width for analysis
    window = predictions
    
    # Pixels vertically
    row_sums = np.sum(window, axis=1)
    
    # Horizontal Sum Profile
    if visualize:
        plt.figure(figsize=(10, 30))
        plt.plot(row_sums, range(len(row_sums)))
        plt.gca().invert_yaxis()
        plt.title("Horizontal Pixel Sum Profile")
        plt.xlabel("Sum of pixels")
        plt.ylabel("Y position")
        plt.show()
    
    # Find peaks (stafflines)
    peaks, properties = find_peaks(row_sums, 
                                 distance=10,
                                 height=0.2*np.max(row_sums),
                                 width=1)
    
    if len(peaks) == 0:
        print("No peaks found")
        return [], predictions
        
    # Group stafflines into tracks (5 lines per staff)
    staff_groups = []
    current_group = []
    
    for i in range(len(peaks)-1):
        current_group.append(peaks[i])
        if peaks[i+1] - peaks[i] > 30:  # Large Pixel Gap indicates new staff (Not yet adaptive)
            if len(current_group) == 5:
                staff_groups.append(current_group)
            current_group = []

    if len(peaks) > 0:
        current_group.append(peaks[-1])
        if len(current_group) == 5:
            staff_groups.append(current_group)
            
    staff_centers = [np.mean(staff_lines) for staff_lines in staff_groups]
    
    # Sort tracks by vertical position
    sorted_indices = np.argsort(staff_centers)
    staff_groups = [staff_groups[i] for i in sorted_indices]
    staff_centers = [staff_centers[i] for i in sorted_indices]
    
    # Create staff objects in pairs
    for i in range(0, len(staff_groups), 2):
        if i + 1 >= len(staff_groups):
            break
            
        group_num = (i // 2) + 1
        
        # Create treble staff track(upper)
        upper_lines = staff_groups[i]
        unit_size = np.mean(np.diff(upper_lines))
        center = np.mean(upper_lines)
        
        treble_staff = Staff(
            lines=upper_lines,
            center=center,
            upper_bound=upper_lines[0] - unit_size,
            lower_bound=upper_lines[-1] + unit_size,
            unit_size=unit_size,
            track=1,
            group=group_num
        )
        
        # Treble staff attributes
        treble_staff.staffline_positions = upper_lines
        treble_staff.staffline_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        treble_staff.staffline_pitches = ['F5', 'D5', 'B4', 'G4', 'E4']  # Main staff lines (correct)
        treble_staff.ghost_lines_above = [upper_lines[0] - 2*unit_size, upper_lines[0] - unit_size]
        treble_staff.ghost_pitches_above = ['C6', 'A5']  
        treble_staff.ghost_lines_below = [upper_lines[-1] + unit_size, upper_lines[-1] + 2*unit_size]
        treble_staff.ghost_pitches_below = ['C4', 'A3'] 
        
        staffs.append(treble_staff)
        
        # Create bass staff (lower)
        lower_lines = staff_groups[i + 1]
        unit_size = np.mean(np.diff(lower_lines))
        center = np.mean(lower_lines)
        
        bass_staff = Staff(
            lines=lower_lines, 
            center=center,
            upper_bound=lower_lines[0] - unit_size,
            lower_bound=lower_lines[-1] + unit_size,
            unit_size=unit_size,
            track=2,
            group=group_num
        )
        
        # Bass staff attributes
        bass_staff.staffline_positions = lower_lines
        bass_staff.staffline_colors = [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255)]
        bass_staff.staffline_pitches = ['A3', 'F3', 'D3', 'B2', 'G2']  
        bass_staff.ghost_lines_above = [lower_lines[0] - 2*unit_size, lower_lines[0] - unit_size]
        bass_staff.ghost_pitches_above = ['E4', 'C4']  
        bass_staff.ghost_lines_below = [lower_lines[-1] + unit_size, lower_lines[-1] + 2*unit_size]
        bass_staff.ghost_pitches_below = ['E2', 'C2']  
        
        staffs.append(bass_staff)
        
        print(f"\nCreated staff pair group {group_num}")
        print(f"Treble staff at y={center:.1f}")
        print(f"Bass staff at y={np.mean(lower_lines):.1f}")
    
    # Sort staffs for visualization
    staffs.sort(key=lambda x: (x.group, x.track))
    
    # Visualize result
    if visualize:
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        ghost_color = (150, 150, 150) 
        
        for staff in staffs:
            # Draw bounds
            cv2.line(result_img, (0, int(staff.upper_bound)), (original_width, int(staff.upper_bound)), (128,128,128), 1)
            cv2.line(result_img, (0, int(staff.lower_bound)), (original_width, int(staff.lower_bound)), (128,128,128), 1)
            
            # Draw ghost lines above with pitches
            for ghost_y, ghost_pitch in zip(staff.ghost_lines_above, staff.ghost_pitches_above):
                cv2.line(result_img, (0, int(ghost_y)), (original_width, int(ghost_y)), ghost_color, 1)
                cv2.putText(result_img, f"{ghost_pitch} y={int(ghost_y)}", (10, int(ghost_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, ghost_color, 2)
                
            # Draw ghost lines below with pitches
            for ghost_y, ghost_pitch in zip(staff.ghost_lines_below, staff.ghost_pitches_below):
                cv2.line(result_img, (0, int(ghost_y)), (original_width, int(ghost_y)), ghost_color, 1)
                cv2.putText(result_img, f"{ghost_pitch} y={int(ghost_y)}", (10, int(ghost_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, ghost_color, 2)
            
            # Draw each staffline with unique color and pitch label
            for i, (pos, color, pitch) in enumerate(zip(staff.staffline_positions, staff.staffline_colors, staff.staffline_pitches)):
                cv2.line(result_img, (0, int(pos)), (original_width, int(pos)), color, 2)
                text_x = 50 + (i * 80)  
                text_y = int(pos)
                cv2.putText(result_img, f"{pitch} y={text_y}", (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add track and group label
            text_x = 10
            text_y = int(staff.center)
            cv2.putText(result_img, f"T{staff.track}G{staff.group}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        
        plt.figure(figsize=(30,20))
        plt.imshow(result_img)
        plt.title(f"Final Result - Found {len(staffs)} Staff Systems\nEach staffline and ghost line labeled with pitch and y-position\nGray lines show extended range", fontsize=16)
        plt.show()
    
    return staffs, predictions

######################################################## Start of step 2 ########################################################

def detect_and_process_noteheads(image_path, yolo_model, staffs, visualize=False):
    """
    Detect noteheads in an image by processing each staff group separately,
    filter overlapping boxes, assign pitches, and visualize results.
    
    Args:
        image_path: Path to the sheet music image
        yolo_model: YOLO model for notehead detection
        staffs: List of Staff objects with staffline information
        visualize: Whether to show visualization plots
        
    Returns:
        notehead_data: List of dictionaries containing information about each detected notehead
        result_img: Visualization image with detected noteheads
    """
    
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    result_img = img.copy()
    
    # Group staffs by group number
    staff_groups = {}
    for staff in staffs:
        if staff.group not in staff_groups:
            staff_groups[staff.group] = []
        staff_groups[staff.group].append(staff)
    
    # Sort staff groups by group number
    sorted_groups = sorted(staff_groups.items())
    
    # Function to calculate IoU (Intersection over Union)
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate areas
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    # Function to check if one box is completely inside another
    def is_box_contained(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return (x1_2 <= x1_1 <= x2_2 and 
                x1_2 <= x2_1 <= x2_2 and
                y1_2 <= y1_1 <= y2_2 and 
                y1_2 <= y2_1 <= y2_2)
    
    # Helper functions for pitch assignment
    def get_base_notehead_type(class_name):
        """Convert specific notehead classes to their base type"""
        if 'noteheadBlack' in class_name:
            return 'noteheadBlack'
        elif 'noteheadHalf' in class_name:
            return 'noteheadHalf'
        elif 'noteheadWhole' in class_name:
            return 'noteheadWhole'
        elif 'noteheadDoubleWhole' in class_name:
            return 'noteheadDoubleWhole'
        return class_name
    
    def get_duration_from_class(class_name):
        """Convert notehead class to duration in beats"""
        base_type = get_base_notehead_type(class_name)
        durations = {
            'noteheadWhole': 4.0,
            'noteheadDoubleWhole': 8.0,
            'noteheadHalf': 2.0,
            'noteheadBlack': 1.0
        }
        return durations.get(base_type, 1.0)
    
    # Process each staff group separately
    all_notehead_data = []
    
    for group_num, group_staffs in sorted_groups:
        print(f"\nProcessing staff group {group_num}...")
        
        group_staffs.sort(key=lambda x: x.track)
        
        # Find group boundaries
        min_y = max(0, int(min(staff.upper_bound for staff in group_staffs) - 50))
        max_y = min(img.shape[0], int(max(staff.lower_bound for staff in group_staffs) + 50))
        
        # Crop image to staff group region
        staff_group_img = img[min_y:max_y, :]
        
        # Save temporary cropped image
        temp_crop_path = f"temp_crop_group_{group_num}.jpg"
        cv2.imwrite(temp_crop_path, staff_group_img)
        
        # Run YOLO detection on cropped image
        results = yolo_model(temp_crop_path)
        
        result = results[0]
        
        # Get all boxes and filter overlapping ones
        filtered_boxes = []
        boxes_to_keep = []
        
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            # Adjust y coordinates to original image
            orig_y1 = y1 + min_y
            orig_y2 = y2 + min_y
            
            # Check overlap with existing boxes
            overlapping = False
            for j, existing_box in enumerate(filtered_boxes):
                iou = calculate_iou([x1, y1, x2, y2], existing_box[:4])
                # Check for complete containment
                box1_contains_box2 = is_box_contained([x1, y1, x2, y2], existing_box[:4])
                box2_contains_box1 = is_box_contained(existing_box[:4], [x1, y1, x2, y2])
                
                if iou > 0.5 or box1_contains_box2 or box2_contains_box1:
                    # Keep the box with higher confidence
                    if conf > existing_box[4]:
                        filtered_boxes.remove(existing_box)
                        boxes_to_keep.remove(j)
                        filtered_boxes.append((x1, y1, x2, y2, conf, class_id))
                        boxes_to_keep.append(i)
                    overlapping = True
                    break
            
            if not overlapping:
                # Additional size-based filtering
                box_width = x2 - x1
                box_height = y2 - y1
                aspect_ratio = box_width / box_height
                
                # Only keep boxes with reasonable size
                if 0.5 < aspect_ratio < 2.0 and 5 < box_width < 50 and 5 < box_height < 50:
                    filtered_boxes.append((x1, y1, x2, y2, conf, class_id))
                    boxes_to_keep.append(i)
        
        if visualize:
            for x1, y1, x2, y2, conf, class_id in filtered_boxes:
                # Adjust y coordinates to original image
                orig_y1 = y1 + min_y
                orig_y2 = y2 + min_y
            
                cv2.rectangle(result_img, (x1, orig_y1), (x2, orig_y2), (0, 255, 0), 1)
                
                label = f"C{class_id}({conf:.1f})"
                cv2.putText(result_img, label, (x1, orig_y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        
        # Calculate average box size for staff group
        total_area = 0
        box_count = 0
        all_boxes = []
        
        for x1, y1, x2, y2, conf, class_id in filtered_boxes:
            box_width = x2 - x1
            box_height = y2 - y1
            aspect_ratio = box_width / box_height
            
            if 0.5 < aspect_ratio < 2.0 and 5 < box_width < 50 and 5 < box_height < 50:
                area = box_width * box_height
                total_area += area
                box_count += 1
                all_boxes.append((x1, y1, x2, y2, conf, class_id, area))
        
        if box_count > 0:
            average_area = total_area / box_count
            print(f"  Average notehead area for group {group_num}: {average_area:.1f} pixels²")
        else:
            print(f"  No valid boxes found for average calculation in group {group_num}")
            average_area = 0
        
        # Process valid boxes
        group_notehead_data = []
        
        for x1, y1, x2, y2, conf, class_id, area in all_boxes:
            if average_area > 0 and area < average_area * 0.5:
                print(f"  Skipping small box with area {area:.1f} (less than half of average {average_area:.1f})")
                continue
            
            orig_y1 = y1 + min_y
            orig_y2 = y2 + min_y
            
            class_name = result.names[class_id]
            
            # Calculate notehead center
            notehead_x = (x1 + x2) / 2
            notehead_y = (orig_y1 + orig_y2) / 2
            
            # Find the closest staff within this group
            closest_staff = None
            min_distance = float('inf')
            
            for staff in group_staffs:
                if staff.upper_bound <= notehead_y <= staff.lower_bound:
                    closest_staff = staff
                    break
                else:
                    dist = min(abs(staff.upper_bound - notehead_y), 
                              abs(staff.lower_bound - notehead_y))
                    if dist < min_distance:
                        min_distance = dist
                        closest_staff = staff
            
            if closest_staff:
                # Calculate distance to each staffline and ghostline
                all_line_positions = (closest_staff.staffline_positions + 
                                    closest_staff.ghost_lines_above +
                                    closest_staff.ghost_lines_below)
                all_line_pitches = (closest_staff.staffline_pitches +
                                  closest_staff.ghost_pitches_above +
                                  closest_staff.ghost_pitches_below)
                
                distances = [abs(notehead_y - pos) for pos in all_line_positions]
                closest_line_idx = np.argmin(distances)
                min_distance = distances[closest_line_idx]
                
                # Get the base pitch from the closest line
                base_pitch = all_line_pitches[closest_line_idx]
                
                # Determine if the note is on a line or in a space
                unit_size = closest_staff.unit_size
                is_on_line = min_distance < unit_size / 3
                
                # If note is in a space, adjust the pitch accordingly
                if not is_on_line:
                    # Check if note is above or below the closest line
                    if notehead_y < all_line_positions[closest_line_idx]:
                        # Above
                        base_pitch = increment_pitch(base_pitch)
                    else:
                        # Below
                        base_pitch = decrement_pitch(base_pitch)
                
                group_notehead_data.append({
                    'x': notehead_x,
                    'y': notehead_y,
                    'staff': closest_staff,
                    'track': closest_staff.track,
                    'group': closest_staff.group,
                    'pitch': base_pitch,
                    'bbox': (x1, orig_y1, x2, orig_y2),
                    'class': class_name,
                    'has_flag': False,
                    'duration': get_duration_from_class(class_name)
                })
        
        # Add noteheads from this group to the overall list
        all_notehead_data.extend(group_notehead_data)
        
        try:
            os.remove(temp_crop_path)
        except:
            pass
    
    # Sort all noteheads by x position within each track and group
    all_notehead_data.sort(key=lambda x: (x['group'], x['track'], x['x']))
    
    print("\nNoteheads by Track and Group:")
    current_group = None
    current_track = None
    
    for note in all_notehead_data:
        if current_group != note['group'] or current_track != note['track']:
            print(f"\nGroup {note['group']}, Track {note['track']} ({'Treble' if note['track']==1 else 'Bass'}):")
            current_group = note['group']
            current_track = note['track']
        print(f"  Pitch: {note['pitch']}, X: {int(note['x'])}, Y: {int(note['y'])}")
    
    # Visuaize
    if visualize:
        track_vis_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        # Draw stafflines
        for staff in staffs:
            for pos, color in zip(staff.staffline_positions, staff.staffline_colors):
                cv2.line(track_vis_img, (0, int(pos)), (img.shape[1], int(pos)), color, 1)
        
        # Draw noteheads
        for note in all_notehead_data:
            x1, y1, x2, y2 = note['bbox']
            pitch = note['pitch']
            track = note['track']
            group = note['group']
            
            color = (0, 255, 0) if track == 1 else (255, 0, 0)  # Green for treble, Red for bass
            
            cv2.rectangle(track_vis_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            
            label = f"{pitch} (T{track}G{group})"
            cv2.putText(track_vis_img, label, 
                        (int(x2 + 5), int((y1 + y2) // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.4, color, 1)
        
        type_vis_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        type_colors = {
            'noteheadWhole': (255, 0, 0),        # Red
            'noteheadDoubleWhole': (0, 255, 0),  # Green
            'noteheadHalf': (0, 0, 255),         # Blue
            'noteheadBlack': (255, 165, 0)       # Orange
        }
        
        for note in all_notehead_data:
            base_type = get_base_notehead_type(note['class'])
            color = type_colors.get(base_type, (128, 128, 128))  #
            
            # Draw a circle at note position
            cv2.circle(type_vis_img,
                    (int(note['x']), int(note['y'])),
                    5, color, -1)
            
            label = f"{note['pitch']} ({base_type})"
            cv2.putText(type_vis_img,
                    label,
                    (int(note['x']), int(note['y'] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1)
            
        plt.figure(figsize=(45, 30))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Noteheads (Filtered for Overlaps)", fontsize=16)
        plt.show()
        
        plt.figure(figsize=(30, 20))
        plt.imshow(track_vis_img)
        plt.title("Detected Noteheads with Pitch, Track, and Group Assignment", fontsize=16)
        plt.show()
        
        plt.figure(figsize=(30, 20))
        plt.imshow(type_vis_img)
        plt.title("Notes with Updated Pitches and Classes\nRed=Whole, Green=Double Whole, Blue=Half, Orange=Black", fontsize=16)
        plt.show()
    
    return all_notehead_data, result_img

def increment_pitch(pitch):
    """Increment a pitch by one step (e.g., E4 -> F4)"""
    note = pitch[0]
    octave = int(pitch[-1])
    
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    idx = notes.index(note)
    
    if idx == len(notes) - 1:  # If it's B, go to C of next octave
        return f'C{octave + 1}'
    else:
        return f'{notes[idx + 1]}{octave}'

def decrement_pitch(pitch):
    """Decrement a pitch by one step (e.g., F4 -> E4)"""
    note = pitch[0]
    octave = int(pitch[-1])
    
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    idx = notes.index(note)
    
    if idx == 0:  # If it's C, go to B of previous octave
        return f'B{octave - 1}'
    else:
        return f'{notes[idx - 1]}{octave}'

def group_notes_into_chords(notehead_data, img=None, visualize=False):
    """
    Group detected notes into chords based on horizontal proximity.
    
    Args:
        notehead_data: List of dictionaries containing information about each detected notehead
        img: Original image for visualization (optional)
        visualize: Whether to show visualization plots
    
    Returns:
        note_groups: List of NoteGroup objects representing chords
        group_vis_img: Visualization image with note groups (if img is provided)
    """
    class NoteGroup:
        def __init__(self, group_id, track, staff_group):
            self.group_id = group_id
            self.track = track
            self.staff_group = staff_group
            self.notes = []
            
        def add_note(self, note):
            self.notes.append(note)
            
        def __str__(self):
            notes_str = "\n".join([f"        Pitch: {note['pitch']}, X: {int(note['x'])}, Y: {int(note['y'])}" 
                                  for note in self.notes])
            return f"Note Group No. {self.group_id} / Group: {self.staff_group} / Track: {self.track} :(\n    Note count: {len(self.notes)}\n    Notes:\n{notes_str})"

    # First, organize notes by staff group and track
    staff_groups = {}
    for note in notehead_data:
        key = (note['group'], note['track'])
        if key not in staff_groups:
            staff_groups[key] = []
        staff_groups[key].append(note)

    # Sort Notes
    for key in staff_groups:
        staff_groups[key].sort(key=lambda x: x['x'])

    note_groups = []
    current_group_id = 0

    for (staff_group, track), notes in sorted(staff_groups.items()):
        current_group = None
        
        for note in notes:
            x, y = note['x'], note['y']
            
            if current_group is None:
                current_group = NoteGroup(current_group_id, track, staff_group)
                current_group.add_note(note)
                note_groups.append(current_group)
                current_group_id += 1
                continue
                
            # Try to add to current group if close enough horizontally
            last_note = current_group.notes[-1]
            x_diff = abs(last_note['x'] - x)
            
            # Define threshold for grouping
            x_threshold = 20  # Horizontal distance threshold (Not yet adaptive)
            
            if x_diff <= x_threshold:
                current_group.add_note(note)
            else:
                current_group = NoteGroup(current_group_id, track, staff_group)
                current_group.add_note(note)
                note_groups.append(current_group)
                current_group_id += 1

    group_vis_img = None
    
    if img is not None and visualize:
        
        group_vis_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

        # Draw groups on the image
        for group in note_groups:
            color = (0, 255, 0) if group.track == 1 else (255, 0, 0)  # Green for treble, Red for bass
            
            min_x = min(note['x'] for note in group.notes)
            max_x = max(note['x'] for note in group.notes)
            min_y = min(note['y'] for note in group.notes)
            max_y = max(note['y'] for note in group.notes)
            
            # Draw rectangle around the group
            cv2.rectangle(group_vis_img, 
                         (int(min_x - 5), int(min_y - 5)), 
                         (int(max_x + 5), int(max_y + 5)), 
                         color, 1)
            
    
            label = f"G{group.group_id} (S{group.staff_group})" 
            cv2.putText(group_vis_img, label,
                        (int(min_x), int(min_y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1)

        plt.figure(figsize=(30,20))
        plt.imshow(group_vis_img)
        plt.title("Note Groups Visualization (Grouped by Staff)", fontsize=16)
        plt.show()

    return note_groups, group_vis_img

######################################################## Start of step 3 ########################################################

def detect_bar_lines(img, staffs, visualize=False):
    """
    Detect bar lines within each staff group.
    
    Args:
        img: Input image
        staffs: List of Staff objects
        visualize: Whether to show visualization plots
    
    Returns:
        bar_lines: List of dictionaries containing information about each detected bar line
        result_img: Visualization image with detected bar lines
    """
    bar_lines = []
    result_img = img.copy()
    
    # Group staffs by group number
    staff_groups = {}
    for staff in staffs:
        if staff.group not in staff_groups:
            staff_groups[staff.group] = []
        staff_groups[staff.group].append(staff)
    
    # Process each group
    for group_num, group_staffs in staff_groups.items():
    
        group_staffs.sort(key=lambda x: x.track)
        
        # Find group boundaries
        min_y = min(staff.upper_bound for staff in group_staffs)
        max_y = max(staff.lower_bound for staff in group_staffs)
        
        # Create a vertical projection of black pixels
        roi = img[int(min_y):int(max_y), :]
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement
        roi = cv2.equalizeHist(roi)
        
        # Create binary image with enhancement
        binary = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=2
        )
        
        # Convert to boolean array
        binary = binary > 0
        
        if visualize:    
            plt.figure(figsize=(15, 3))
            plt.imshow(binary, cmap='gray')
            plt.title(f"Enhanced Binary ROI for Group {group_num}")
            plt.show()
            
        vertical_projection = np.sum(binary, axis=0).flatten()
        
        # Find peaks in the vertical projection
        peaks, properties = find_peaks(vertical_projection,
                                     height=0.9 * np.max(vertical_projection),
                                     distance=30) #Not yet fully adaptive
        
        if visualize:
            print("\nPeak heights:")
            for idx, peak in enumerate(peaks):
                print(f"Bar line {idx + 1} at x={peak}: {vertical_projection[peak]} black pixels")
            
            # Visualize vertical projection
            plt.figure(figsize=(15, 5))
            plt.plot(vertical_projection)
            plt.axhline(y=0.9 * np.max(vertical_projection), color='r', linestyle='--', label='90% threshold')
            plt.title(f"Vertical Projection - Group {group_num}")
            plt.xlabel("X position")
            plt.ylabel("Black pixel count")
            plt.plot(peaks, vertical_projection[peaks], "x")
            
            for peak in peaks:
                plt.annotate(f'{int(vertical_projection[peak])}', 
                            (peak, vertical_projection[peak]),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center')
                
            plt.legend()
            plt.show()
        
        # Store bar lines
        for x in peaks:
            bar_lines.append({
                'x': x,
                'group': group_num,
                'min_y': min_y,
                'max_y': max_y
            })
        
        # Draw bar lines on result image
        for bar in bar_lines:
            if bar['group'] == group_num:
                cv2.line(result_img,
                        (int(bar['x']), int(bar['min_y'])),
                        (int(bar['x']), int(bar['max_y'])),
                        (0, 0, 255), 2)
    
    if visualize:
        plt.figure(figsize=(30, 20))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Bar Lines", fontsize=16)
        plt.show()
    
    return bar_lines, result_img

def debug_yolo_detections(img, yolo_results, title="YOLO Detections"):
    """
    Visualize all YOLO detections with detailed information
    
    Args:
        img: Input image
        yolo_results: Results from YOLO model
        title: Title for the visualization
        
    Returns:
        debug_img: Visualization image with detections
    """
    debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    print(f"\n{title}:")
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].item())
            class_name = result.names[cls]
            conf = box.conf[0].item()
        
            print(f"\nDetected {class_name}")
            print(f"  Position: x={x1:.1f}-{x2:.1f}, y={y1:.1f}-{y2:.1f}")
            print(f"  Confidence: {conf:.3f}")
            
            # Draw box and label on image
            cv2.rectangle(debug_img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (255, 0, 0), 2)
            
            # Calculate and draw center point
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(debug_img, (center_x, center_y), 5, (0, 255, 0), -1)
            
            # For flats, also draw the "belly" point (75% down from top)
            if 'flat' in class_name.lower():
                belly_y = int(y1 + (y2 - y1) * 0.75)
                cv2.circle(debug_img, (center_x, belly_y), 5, (0, 0, 255), -1)
                cv2.putText(debug_img, "belly", (center_x + 10, belly_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            label = f"{class_name} ({conf:.2f})"
            cv2.putText(debug_img,
                       label,
                       (int(x1), int(y1-5)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 0, 0),
                       1)
    
    plt.figure(figsize=(30,20))
    plt.imshow(debug_img)
    plt.title(title, fontsize=16)
    plt.show()
    
    return debug_img

def detect_accidentals(img, staffs, accidental_model, visualize=True):
    """
    Detect accidentals in the sheet music image
    
    Args:
        img: Input image
        staffs: List of Staff objects
        accidental_model: YOLO model for accidental detection
        visualize: Whether to show visualization plots
        
    Returns:
        accidentals: List of dictionaries containing information about each detected accidental
        result_img: Visualization image with detected accidentals
    """
    accidental_results = accidental_model(img)
    
    if visualize:
        print("\nAvailable accidental class names in YOLO model:")
        print(accidental_model.names)
        
        # Run the debug function
        debug_img = debug_yolo_detections(img, accidental_results, "Accidental Detections")
    
    # Process the results
    accidentals = []
    result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    for result in accidental_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0].item())
            class_name = result.names[cls]
            conf = box.conf[0].item()
            
            # Calculate center position
            center_x = (x1 + x2) / 2
            if 'flat' in class_name.lower():
                # For flats, use the "belly" position (about 75% down from the top)
                center_y = y1 + (y2 - y1) * 0.75
            else:
                # For other accidentals, use the normal center
                center_y = (y1 + y2) / 2
            
            closest_staff = None
            min_distance = float('inf')
            
            for staff in staffs:
                if staff.upper_bound <= center_y <= staff.lower_bound:
                    closest_staff = staff
                    break
                else:
                    dist = min(abs(staff.upper_bound - center_y), 
                              abs(staff.lower_bound - center_y))
                    if dist < min_distance:
                        min_distance = dist
                        closest_staff = staff
            
            if closest_staff:
                # Store accidental information
                accidental = {
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': [x1, y1, x2, y2],
                    'class': class_name,
                    'confidence': conf,
                    'staff': closest_staff,
                    'track': closest_staff.track,
                    'group': closest_staff.group
                }
                accidentals.append(accidental)
                
                color = (0, 0, 255)  # Red for accidentals
                cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 1)
                
                label = f"{class_name} (T{closest_staff.track}G{closest_staff.group})"
                cv2.putText(result_img, label, 
                            (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, color, 1)
    
    if visualize and accidentals:
        plt.figure(figsize=(30,20))
        plt.imshow(result_img)
        plt.title("Detected Accidentals with Staff Assignment", fontsize=16)
        plt.show()
    
        print("\nAccidentals by Track and Group:")
        current_group = None
        current_track = None
        
        for acc in sorted(accidentals, key=lambda x: (x['group'], x['track'], x['center_x'])):
            if current_group != acc['group'] or current_track != acc['track']:
                print(f"\nGroup {acc['group']}, Track {acc['track']} ({'Treble' if acc['track']==1 else 'Bass'}):")
                current_group = acc['group']
                current_track = acc['track']
            print(f"  {acc['class']}: X={int(acc['center_x'])}, Y={int(acc['center_y'])}")
    
    return accidentals, result_img

def process_key_signatures(staffs, accidentals, visualize=False):
    """
    Process key signatures at the start of each staff
    
    Args:
        staffs: List of Staff objects
        accidentals: List of detected accidentals
        visualize: Whether to show visualization plots
        
    Returns:
        None (updates staff objects directly)
    """
    print("\nProcessing key signatures...")
    
    # Define threshold for key signature region (Not really used, just to avoid key changes during the sheet music, not yet adaptive)
    key_signature_x_threshold = 1000 
    print(f"Key signature region threshold: {key_signature_x_threshold} pixels")
    
    # Filter accidentals
    key_accidentals = [acc for acc in accidentals 
                      if acc['center_x'] < key_signature_x_threshold 
                      and ('keyflat' in acc['class'].lower() or 'keysharp' in acc['class'].lower())]
    
    
    for staff in staffs:
        staff.key_signature = {}
    
    # Process each key signature accidental
    for acc in key_accidentals:
        staff = acc['staff']
        
        # Recalculate the detection point based on the bounding box
        box_height = acc['y2'] - acc['y1']
        
        # Determine accidental type and precise detection point
        if 'flat' in acc['class'].lower():
            accidental_type = 'flat'
            # For flats, use 75% down from the top of the bounding box
            detection_y = acc['y1'] + (box_height * 0.75)
        elif 'sharp' in acc['class'].lower():
            accidental_type = 'sharp'
            # For sharps, use the middle of the bounding box
            detection_y = acc['y1'] + (box_height * 0.50)
        else:
            continue
            
        print(f"\nFound {acc['class']} at y={detection_y:.1f}")
        print(f"  In staff {staff.track} (Group {staff.group})")
        print(f"  Bounding box: y1={acc['y1']:.1f}, y2={acc['y2']:.1f}, height={box_height:.1f}")
        
        # Get the pitch at this position
        pitch = staff.get_pitch_at_position(detection_y)
        
        print(f"  Detected pitch at position: {pitch}")
        
        # Verify pitch detection
        print(f"  Staff line positions:")
        for i, pos in enumerate(staff.staffline_positions):
            print(f"    Line {i+1}: y={pos:.1f}, pitch={staff.staffline_pitches[i]}")
        
        print(f"  Staff space positions:")
        for i, pos in enumerate(staff.staff_space_positions):
            print(f"    Space {i+1}: y={pos:.1f}, pitch={staff.staff_spaces[i]}")
        
        if pitch:
            note_letter = pitch[0]  # Get the letter part of the pitch
            
            # Get all pitches for this staff
            all_pitches = []
            
            # Add staffline pitches
            all_pitches.extend(staff.staffline_pitches)
            
            # Add staff space pitches
            all_pitches.extend(staff.staff_spaces)
            
            # Add ghost line pitches
            all_pitches.extend(staff.ghost_pitches_above)
            all_pitches.extend(staff.ghost_pitches_below)
            
            # Add ghost space pitches
            all_pitches.extend(staff.ghost_spaces_above)
            all_pitches.extend(staff.ghost_spaces_below)
            
            # Add transition space pitches
            all_pitches.append(staff.transition_space_above_pitch)
            all_pitches.append(staff.transition_space_below_pitch)
            
            # Apply accidental to all matching pitches
            for stored_pitch in all_pitches:
                if stored_pitch[0] == note_letter:
                    staff.key_signature[stored_pitch] = accidental_type
                    print(f"  Added {accidental_type} to key signature for pitch {stored_pitch}")
        else:
            print(f"  WARNING: No pitch detected at position y={detection_y:.1f}")
    
    # Final key signatures
    print("\nFinal Key Signatures:")
    for staff in staffs:
        print(f"\nStaff {staff.track} (Group {staff.group}):")
        if staff.key_signature:
            for pitch, accidental in sorted(staff.key_signature.items()):
                print(f"  {pitch}: {accidental}")
        else:
            print("  No key signature")

def visualize_key_signatures(img, staffs, visualize=False):
    """
    Visualize the stafflines and spaces affected by key signatures
    
    Args:
        img: Input image
        staffs: List of Staff objects
        visualize: Whether to show visualization plots
        
    Returns:
        result_img: Visualization image with key signature highlights
    """
    if not visualize:
        return None
        
    result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    for staff in staffs:

        # Staff lines
        for pos in staff.staffline_positions:
            cv2.line(result_img, (0, int(pos)), (img.shape[1], int(pos)), (200, 200, 200), 1)
        
        # Highlight positions affected by key signature
        for pitch, accidental in staff.key_signature.items():
            color = (255, 0, 0) if accidental == 'flat' else (0, 0, 255)
            
            # Check ghost lines above
            for i, ghost_pitch in enumerate(staff.ghost_pitches_above):
                if ghost_pitch[0] == pitch[0]:  # Match the letter part
                    y_pos = staff.ghost_lines_above[i]
                    cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                    cv2.putText(result_img, f"{ghost_pitch} {accidental}", (10, int(y_pos) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check ghost spaces above
            for i, ghost_pitch in enumerate(staff.ghost_spaces_above):
                if ghost_pitch[0] == pitch[0]:
                    y_pos = staff.ghost_space_positions_above[i]
                    cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                    cv2.putText(result_img, f"{ghost_pitch} {accidental}", (10, int(y_pos) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check transition space above
            if staff.transition_space_above_pitch[0] == pitch[0]:
                y_pos = staff.transition_space_above_pos
                cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                cv2.putText(result_img, f"{staff.transition_space_above_pitch} {accidental}", (10, int(y_pos) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check staff lines
            for i, staff_pitch in enumerate(staff.staffline_pitches):
                if staff_pitch[0] == pitch[0]:
                    y_pos = staff.staffline_positions[i]
                    cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                    cv2.putText(result_img, f"{staff_pitch} {accidental}", (10, int(y_pos) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check staff spaces
            for i, space_pitch in enumerate(staff.staff_spaces):
                if space_pitch[0] == pitch[0]:
                    y_pos = staff.staff_space_positions[i]
                    cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                    cv2.putText(result_img, f"{space_pitch} {accidental}", (10, int(y_pos) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check transition space below
            if staff.transition_space_below_pitch[0] == pitch[0]:
                y_pos = staff.transition_space_below_pos
                cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                cv2.putText(result_img, f"{staff.transition_space_below_pitch} {accidental}", (10, int(y_pos) - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check ghost lines below
            for i, ghost_pitch in enumerate(staff.ghost_pitches_below):
                if ghost_pitch[0] == pitch[0]:
                    y_pos = staff.ghost_lines_below[i]
                    cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                    cv2.putText(result_img, f"{ghost_pitch} {accidental}", (10, int(y_pos) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Check ghost spaces below
            for i, ghost_pitch in enumerate(staff.ghost_spaces_below):
                if ghost_pitch[0] == pitch[0]:
                    y_pos = staff.ghost_space_positions_below[i]
                    cv2.line(result_img, (0, int(y_pos)), (img.shape[1], int(y_pos)), color, 2)
                    cv2.putText(result_img, f"{ghost_pitch} {accidental}", (10, int(y_pos) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    plt.figure(figsize=(30,20))
    plt.imshow(result_img)
    plt.title("Stafflines and Spaces Affected by Key Signatures\nRed = Flat, Blue = Sharp", fontsize=16)
    plt.show()
    
    return result_img

def update_note_pitches_with_key_signatures(staffs, notehead_data):
    """
    Update the pitches of detected notes based on key signatures
    
    Args:
        staffs: List of Staff objects
        notehead_data: List of dictionaries containing information about each detected notehead
        
    Returns:
        None (updates notehead_data directly)
    """
    print("\nUpdating note pitches based on key signatures...")
    
    for note in notehead_data:
        # Find the corresponding staff
        staff = next((s for s in staffs 
                     if s.track == note['track'] 
                     and s.group == note['group']), None)
        
        if staff:
            # Get the base pitch
            base_pitch = note['pitch']
            note_letter = base_pitch[0]  # Extract note letter (e.g., 'B' from 'B4')
            
            # Extract the octave from the pitch
            octave_match = re.search(r'[A-G](\d+)', base_pitch)
            if not octave_match:
                continue 
                
            octave = octave_match.group(1)
            
            # Check if this note letter is affected by key signature
            for pitch, accidental in staff.key_signature.items():
                if pitch[0] == note_letter:  # If same note letter
                    # Update the pitch with the accidental
                    note['pitch'] = f"{base_pitch}{accidental}"
                    print(f"Updated note at x={note['x']:.1f}: {base_pitch} -> {note['pitch']}")
                    break
    
    print("\nVerification of updated notes:")
    for staff in staffs:
        print(f"\nStaff {staff.track} (Group {staff.group}):")
        staff_notes = [n for n in notehead_data 
                      if n['track'] == staff.track and n['group'] == staff.group]
        staff_notes.sort(key=lambda x: x['x']) 
        
        if staff_notes:
            for note in staff_notes:
                print(f"  x={note['x']:.1f}: {note['pitch']}")
        else:
            print("  No notes in this staff")

def visualize_updated_notes(img, notehead_data, visualize=False):
    """
    Visualize the notes with their updated pitches
    
    Args:
        img: Input image
        notehead_data: List of dictionaries containing information about each detected notehead
        visualize: Whether to show visualization plots
        
    Returns:
        result_img: Visualization image with updated note pitches
    """
    if not visualize:
        return None
        
    result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    for note in notehead_data:
        cv2.circle(result_img,
                  (int(note['x']), int(note['y'])),
                  5, (255, 0, 0), -1)
        
        cv2.putText(result_img,
                   note['pitch'],
                   (int(note['x']), int(note['y'] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (255, 0, 0),
                   1)
    
    plt.figure(figsize=(30,20))
    plt.imshow(result_img)
    plt.title("Notes with Updated Pitches", fontsize=16)
    plt.show()
    
    return result_img

def process_accidentals_within_measures(staffs, accidentals, bar_lines, notehead_data, img=None, visualize=False):
    """
    Process accidentals and their effects on notes within measures
    
    Args:
        staffs: List of Staff objects
        accidentals: List of detected accidentals
        bar_lines: List of detected bar lines
        notehead_data: List of dictionaries containing information about each detected notehead
        img: Original image for visualization (optional)
        visualize: Whether to show visualization plots
        
    Returns:
        notehead_data: Updated notehead data with accidentals applied
        debug_img: Visualization image if visualize is True
    """
    print("\nProcessing accidentals within measures...")
    
    # organize bars based on bar lines
    bars = []
    for staff_group in set(bar['group'] for bar in bar_lines):
        
        group_bars = [b for b in bar_lines if b['group'] == staff_group]
        group_bars.sort(key=lambda x: x['x'])
        
        # Create bars between consecutive bar lines
        for i in range(len(group_bars) - 1):
            bar_start = group_bars[i]['x']
            bar_end = group_bars[i + 1]['x']
            bars.append({
                'start_x': bar_start,
                'end_x': bar_end,
                'group': staff_group,
                'min_y': min(group_bars[i]['min_y'], group_bars[i+1]['min_y']),
                'max_y': max(group_bars[i]['max_y'], group_bars[i+1]['max_y']),
                'active_accidentals': {}  # Format: {(track, note_letter, octave): accidental_type}
            })
    
    # Calculate average note width for search window (adaptive)
    note_widths = [note.get('width', 20) for note in notehead_data]
    avg_note_width = sum(note_widths) / len(note_widths) if note_widths else 20
    search_window = avg_note_width * 1.5
    
    debug_img = None
    if visualize and img is not None:
        debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        # Draw bars
        for bar in bars:
            cv2.rectangle(debug_img,
                        (int(bar['start_x']), int(bar['min_y'])),
                        (int(bar['end_x']), int(bar['max_y'])),
                        (200, 200, 200), 1)
            
            cv2.putText(debug_img,
                       f"Bar {bars.index(bar)+1}",
                       (int((bar['start_x'] + bar['end_x'])/2 - 20), int(bar['min_y'] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (200, 200, 200),
                       1)
    
    measure_accidentals = [acc for acc in accidentals 
                          if not ('key' in acc['class'].lower())]
    
    # Process each accidental detection
    for acc in measure_accidentals:
        acc_x = acc['center_x']
        acc_y = acc['center_y']
        acc_class = acc['class'].lower()
        
        # Determine accidental type
        if 'flat' in acc_class:
            accidental_type = 'flat'
        elif 'sharp' in acc_class:
            accidental_type = 'sharp'
        elif 'natural' in acc_class:
            accidental_type = 'natural'
        else:
            print(f"Unknown accidental type: {acc_class}")
            continue
        
        # Find which bar this accidental belongs to
        bar_idx = None
        for i, bar in enumerate(bars):
            if (bar['group'] == acc['group'] and 
                bar['start_x'] <= acc_x <= bar['end_x']):
                bar_idx = i
                break
        
        if bar_idx is None:
            print(f"Accidental at x={acc_x:.1f} doesn't belong to any bar")
            continue
            
        # Calculate search area
        search_x_min = acc_x
        search_x_max = acc_x + search_window
        search_y_min = acc_y - 10
        search_y_max = acc_y + 10
        
        print(f"\nSearching for notes near accidental at x={acc_x:.1f}, y={acc_y:.1f}")
        print(f"Search window: x={search_x_min:.1f} to {search_x_max:.1f}, y={search_y_min:.1f} to {search_y_max:.1f}")
        
        # Find the closest note to the right of the accidental
        closest_note = None
        min_distance = float('inf')
        
        for note in notehead_data:
            note_x = note['x']
            note_y = note['y']
            
            # Calculate distance
            dx = note_x - acc_x
            dy = abs(note_y - acc_y)
            
            print(f"  Checking note at x={note_x:.1f}, y={note_y:.1f}")
            print(f"    Distance: dx={dx:.1f}, dy={dy:.1f}")
            
            if dx < 0:
                print("    Skipped: Note is not to the right of accidental")
                continue
                
            if dx > search_window:
                print("    Skipped: Note is beyond search window")
                continue
                
            if dy > 10:
                print("    Skipped: Note is too far vertically")
                continue
                
            # Calculate Euclidean distance
            distance = (dx**2 + dy**2)**0.5
            
            print(f"    Note is within search area! Distance: {distance:.1f}")
            
            # Update closest note
            if distance < min_distance:
                min_distance = distance
                closest_note = note
                print("    New closest note found!")
        
        if closest_note:
            print(f"\nFound closest note: x={closest_note['x']:.1f}, y={closest_note['y']:.1f}")
            
            # Get the staff and track
            staff = acc['staff']
            track = staff.track
            
            # Get the pitch at this position
            pitch = closest_note['pitch']
            note_letter = pitch[0]  # Get the letter part of the pitch
            
            # Extract the octave from the pitch (e.g., "C4" -> "4")
            octave_match = re.search(r'[A-G](\d+)', pitch)
            if octave_match:
                octave = octave_match.group(1)
            else:
                octave = "?"
            
            print(f"\nProcessing accidental:")
            print(f"  Type: {acc['class']}")
            if 'bbox' in acc:
                print(f"  Position: x={acc['bbox'][0]:.1f}-{acc['bbox'][2]:.1f}, y={acc['bbox'][1]:.1f}-{acc['bbox'][3]:.1f}")
            else:
                print(f"  Position: x={acc_x:.1f}, y={acc_y:.1f}")
            print(f"  Staff: Track {track}, Group {acc['group']}")
            print(f"  Affecting note: {pitch} at x={closest_note['x']:.1f}")
            print(f"  Bar: {bar_idx+1}")
            
            # Store the modification in the bar, including octave information
            bar = bars[bar_idx]
            bar['active_accidentals'][(track, note_letter, octave)] = accidental_type
            
            if visualize and debug_img is not None:
                color = (0, 255, 0) if 'sharp' in acc_class else \
                        (255, 0, 0) if 'flat' in acc_class else \
                        (0, 0, 255)  # natural
                
                if 'bbox' in acc:
                    x1, y1, x2, y2 = acc['bbox']
                    cv2.rectangle(debug_img,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color, 2)
                else:
                    cv2.circle(debug_img,
                              (int(acc_x), int(acc_y)),
                              10, color, 2)
                
                # Draw search area
                cv2.rectangle(debug_img,
                            (int(search_x_min), int(search_y_min)),
                            (int(search_x_max), int(search_y_max)),
                            (0, 255, 255), 1)
                
                # Draw line to note
                cv2.line(debug_img,
                        (int(acc_x), int(acc_y)),
                        (int(closest_note['x']), int(closest_note['y'])),
                        (255, 255, 0), 1)
                
                # Highlight note
                cv2.circle(debug_img,
                          (int(closest_note['x']), int(closest_note['y'])),
                          8, color, 2)
        else:
            print(f"No note found near accidental at x={acc_x:.1f}, y={acc_y:.1f}")
    
    # Apply accidentals to notes
    print("\nApplying note modifications:")
    for note in notehead_data:
        note_x = note['x']
        note_y = note['y']
        track = note['track']
        group = note['group']
        base_pitch = note['pitch']
        note_letter = base_pitch[0] 
        
        # Extract the octave from the pitch
        octave_match = re.search(r'[A-G](\d+)', base_pitch)
        if octave_match:
            octave = octave_match.group(1)
        else:
            octave = "?"  
        
        # Find which bar this note belongs to
        for bar_idx, bar in enumerate(bars):
            if (bar['group'] == group and 
                bar['start_x'] <= note_x <= bar['end_x']):
                
                # Check if this note letter and octave has an active accidental in this bar
                if (track, note_letter, octave) in bar['active_accidentals']:
                    accidental_type = bar['active_accidentals'][(track, note_letter, octave)]
                    
                    # Only apply to this note if it's at or after the accidental position
                    # Find the accidental's x position
                    accidental_found = False
                    accidental_x = 0
                    
                    # Look through all accidentals in this bar to find the one that affects this note
                    for acc in measure_accidentals:
                        if (acc['group'] == group and 
                            bar['start_x'] <= acc['center_x'] <= bar['end_x'] and
                            acc['staff'].track == track):
                            
                            # Check if this is the accidental for this note letter
                            acc_class = acc['class'].lower()
                            acc_type = 'flat' if 'flat' in acc_class else 'sharp' if 'sharp' in acc_class else 'natural'
                            
                            # Find the closest note to this accidental
                            closest_note = None
                            min_dist = float('inf')
                            for n in notehead_data:
                                # Extract octave from this note's pitch
                                n_octave_match = re.search(r'[A-G](\d+)', n['pitch'])
                                n_octave = n_octave_match.group(1) if n_octave_match else "?"
                                
                                if (n['group'] == group and 
                                    n['track'] == track and
                                    n['pitch'][0] == note_letter and
                                    n_octave == octave and  # Match octave too
                                    n['x'] > acc['center_x']):
                                    
                                    dist = n['x'] - acc['center_x']
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_note = n
                            
                            if closest_note and acc_type == accidental_type:
                                accidental_x = acc['center_x']
                                accidental_found = True
                                break
                    
                    # Only apply if the note is after the accidental
                    if accidental_found and note_x >= accidental_x:
                        # If the note already has an accidental from key signature
                        if 'flat' in base_pitch or 'sharp' in base_pitch:
                            # Natural cancels out key signature
                            if accidental_type == 'natural':
                                # Remove the accidental
                                note['pitch'] = note_letter + base_pitch[1:]
                                print(f"Bar {bar_idx+1}: {base_pitch} -> {note['pitch']} (natural)")
                            # Otherwise replace with the new accidental
                            else:
                                note['pitch'] = note_letter + base_pitch[1:] + accidental_type
                                print(f"Bar {bar_idx+1}: {base_pitch} -> {note['pitch']}")
                        else:
                            # If no existing accidental and not natural
                            if accidental_type != 'natural':
                                note['pitch'] = base_pitch + accidental_type
                                print(f"Bar {bar_idx+1}: {base_pitch} -> {note['pitch']}")
                
                break
    
    # Final visualization
    if visualize and img is not None:
        result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        for bar in bar_lines:
            cv2.line(result_img,
                    (int(bar['x']), int(bar['min_y'])),
                    (int(bar['x']), int(bar['max_y'])),
                    (128, 128, 128), 1)
        
        for note in notehead_data:
            cv2.circle(result_img,
                      (int(note['x']), int(note['y'])),
                      5, (255, 165, 0), -1)
            
            cv2.putText(result_img,
                       note['pitch'],
                       (int(note['x']), int(note['y'] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 165, 0),
                       1)
        
        plt.figure(figsize=(30,20))
        plt.imshow(result_img)
        plt.title("Final Note Assignments\nShowing modified pitches", fontsize=16)
        plt.show()
        
        if debug_img is not None:
            plt.figure(figsize=(30,20))
            plt.imshow(debug_img)
            plt.title("Accidental Processing Debug View", fontsize=16)
            plt.show()
    
    return notehead_data, debug_img

######################################################## Start of step 4 ########################################################
def detect_and_process_rests(img, staffs, rest_model, notehead_data, visualize=False):
    """
    Detect and process rests in the sheet music image by processing each staff group separately
    
    Args:
        img: Input image
        staffs: List of Staff objects
        rest_model: YOLO model for rest detection
        notehead_data: List of dictionaries containing information about each detected notehead
        visualize: Whether to show visualization plots
        
    Returns:
        rest_data: List of dictionaries containing information about each detected rest
        result_img: Visualization image with detected rests
    """
    print("\nDetecting and processing rests...")
    
    debug_img = None
    if visualize:
        debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    # Group staffs by group number
    staff_groups = {}
    for staff in staffs:
        if staff.group not in staff_groups:
            staff_groups[staff.group] = []
        staff_groups[staff.group].append(staff)
    
    sorted_groups = sorted(staff_groups.items())

    all_rest_data = []
    
    for group_num, group_staffs in sorted_groups:
        print(f"\nProcessing rests for staff group {group_num}...")
    
        group_staffs.sort(key=lambda x: x.track)
        
        # Find group boundaries
        min_y = max(0, int(min(staff.upper_bound for staff in group_staffs) - 50))
        max_y = min(img.shape[0], int(max(staff.lower_bound for staff in group_staffs) + 50))
        
        # Crop image to staff group region
        staff_group_img = img[min_y:max_y, :]
        
        # Save temporary cropped image
        temp_crop_path = f"temp_crop_rest_group_{group_num}.jpg"
        cv2.imwrite(temp_crop_path, staff_group_img)
        
        # Run yolo model
        results = rest_model(temp_crop_path)
        
        group_rest_data = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())
                class_name = result.names[cls]
                conf = box.conf[0].item()
                
                # Calculate center points
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                orig_center_y = center_y + min_y
                
                if visualize and debug_img is not None:
                    cv2.rectangle(debug_img,
                                (int(x1), int(y1 + min_y)),
                                (int(x2), int(y2 + min_y)),
                                (0, 255, 0), 2)
                    cv2.circle(debug_img,
                              (int(center_x), int(orig_center_y)),
                              5, (255, 0, 0), -1)
                
                # Find corresponding staff within this group
                closest_staff = None
                min_distance = float('inf')
                
                for staff in group_staffs:
                    if staff.upper_bound <= orig_center_y <= staff.lower_bound:
                        closest_staff = staff
                        break
                    else:
                        dist = min(abs(staff.upper_bound - orig_center_y), 
                                  abs(staff.lower_bound - orig_center_y))
                        if dist < min_distance:
                            min_distance = dist
                            closest_staff = staff
                
                if closest_staff:
                    # Determine duration based on rest type
                    duration = None
                    if 'whole' in class_name.lower():
                        duration = 4.0
                    elif 'half' in class_name.lower():
                        duration = 2.0
                    elif 'quarter' in class_name.lower():
                        duration = 1.0
                    elif '8th' in class_name.lower() or 'eighth' in class_name.lower():
                        duration = 0.5
                    elif '16th' in class_name.lower() or 'sixteenth' in class_name.lower():
                        duration = 0.25
                    
                    rest_info = {
                        'x': center_x,
                        'y': orig_center_y,
                        'type': class_name,
                        'duration': duration,
                        'track': closest_staff.track,
                        'group': closest_staff.group,
                        'confidence': conf,
                        'bbox': (x1, y1 + min_y, x2, y2 + min_y)
                    }
                    group_rest_data.append(rest_info)
                    
                    print(f"Found {class_name} rest at x={center_x:.1f}, y={orig_center_y:.1f}")
                    print(f"  Staff {closest_staff.track} (Group {closest_staff.group})")
                    print(f"  Duration: {duration}")
        
        # Add rests from this group to the overall list
        all_rest_data.extend(group_rest_data)
    
        try:
            os.remove(temp_crop_path)
        except:
            pass
    
    # Sort rests by x position within each staff
    all_rest_data.sort(key=lambda x: (x['group'], x['track'], x['x']))
    
    # Filter overlaps
    filtered_rest_data = []
    
    # Function to calculate IoU (Intersection over Union)
    def calculate_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate areas
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
  
    rest_groups = {}
    for rest in all_rest_data:
        key = (rest['group'], rest['track'])
        if key not in rest_groups:
            rest_groups[key] = []
        rest_groups[key].append(rest)
    
    # Process each group separately
    for (group, track), rests in rest_groups.items():
        rests.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Keep track of which rests to keep
        to_keep = [True] * len(rests)
        
        for i in range(len(rests)):
            if not to_keep[i]:
                continue
                
            for j in range(i + 1, len(rests)):
                if not to_keep[j]:
                    continue
                    
                # Calculate IoU between rests
                iou = calculate_iou(rests[i]['bbox'], rests[j]['bbox'])
                
                # If significant overlap, mark the lower confidence one to be removed
                if iou > 0.3: 
                    to_keep[j] = False
        
        # Add rests that should be kept
        for i, rest in enumerate(rests):
            if to_keep[i]:
                filtered_rest_data.append(rest)
    
    print("\nRest Detection Summary:")
    for staff in staffs:
        staff_rests = [r for r in filtered_rest_data 
                      if r['track'] == staff.track and r['group'] == staff.group]
        print(f"\nStaff {staff.track} (Group {staff.group}):")
        if staff_rests:
            for rest in staff_rests:
                print(f"  {rest['type']} rest at x={rest['x']:.1f}")
        else:
            print("  No rests detected")
    
    # Create visualization with both notes and rests
    if visualize:
        result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        
        # Draw notes
        for note in notehead_data:
            cv2.circle(result_img,
                      (int(note['x']), int(note['y'])),
                      5, (255, 0, 0), -1)
            cv2.putText(result_img,
                       note['pitch'],
                       (int(note['x']), int(note['y'] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (255, 0, 0),
                       1)
        
        # Draw rests
        for rest in filtered_rest_data:
            cv2.circle(result_img,
                      (int(rest['x']), int(rest['y'])),
                      5, (0, 255, 0), -1)
            cv2.putText(result_img,
                       f"{rest['type']} ({rest['duration']})",
                       (int(rest['x']), int(rest['y'] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 0),
                       1)
        
        plt.figure(figsize=(30,20))
        plt.imshow(result_img)
        plt.title("Notes (Red) and Rests (Green) in Score", fontsize=16)
        plt.show()
    
        if debug_img is not None:
            plt.figure(figsize=(30,20))
            plt.imshow(debug_img)
            plt.title("Rest Detection Debug View", fontsize=16)
            plt.show()
    
    return filtered_rest_data, debug_img

######################################################## Start of step 5 ########################################################

def detect_augmentation_dots(img, notehead_data, rest_data, staff_height=None, visualize=False):
    """
    Detect augmentation dots with adaptive search areas based on image resolution
    """
    debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Create binary image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate adaptive parameters based on image size or staff height
    if staff_height is None:
        # Estimate based on image height if staff_height not provided
        staff_height = img.shape[0] / 40  # Assuming roughly 8-10 staff systems per page
    
    # Define adaptive search area
    search_radius = int(staff_height * 0.8)    
    vertical_range = int(staff_height * 0.3)   
    min_dot_size = max(2, int(staff_height * 0.1))
    max_dot_size = max(6, int(staff_height * 0.25)) #not fully optimized, may cause issues in high resolution images 
    
    print(f"\nAdaptive search parameters:")
    print(f"Staff height estimate: {staff_height:.1f} pixels")
    print(f"Search radius: {search_radius} pixels")
    print(f"Vertical range: {vertical_range} pixels")
    print(f"Dot size range: {min_dot_size}-{max_dot_size} pixels")
    
    musical_symbols = []
    for note in notehead_data:
        musical_symbols.append({
            'type': 'note',
            'data': note,
            'x': note['x'],
            'y': note['y']
        })
    for rest in rest_data:
        musical_symbols.append({
            'type': 'rest',
            'data': rest,
            'x': rest['x'],
            'y': rest['y']
        })
    
    musical_symbols.sort(key=lambda x: x['x'])
    dot_detections = []
    
    search_areas_img = debug_img.copy()
    potential_dots_img = debug_img.copy()
    
    print("\nSearching for augmentation dots...")
    for symbol in musical_symbols:
        x, y = int(symbol['x']), int(symbol['y'])
        
        # Use adaptive search region
        search_x1 = x + min_dot_size 
        search_x2 = x + search_radius
        search_y1 = max(0, y - vertical_range)
        search_y2 = min(img.shape[0], y + vertical_range)
        
        # Draw search area
        overlay = search_areas_img.copy()
        cv2.rectangle(overlay,
                     (search_x1, search_y1),
                     (search_x2, search_y2),
                     (255, 255, 0), -1)
        cv2.addWeighted(overlay, 0.3, search_areas_img, 0.7, 0, search_areas_img)
        
        color = (255, 0, 0) if symbol['type'] == 'note' else (0, 0, 255)
        cv2.circle(search_areas_img, (x, y), 5, color, -1)
        
        # Extract and process search region
        search_region = binary[search_y1:search_y2, search_x1:search_x2]
        contours, _ = cv2.findContours(search_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x1, y1, w, h = cv2.boundingRect(contour)
            
            if (min_dot_size <= w <= max_dot_size and 
                min_dot_size <= h <= max_dot_size and
                abs(w - h) <= max(2, int(staff_height * 0.1))): 
                
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0 #check if circle formed
                
                if circularity > 0.75:
                    dot_x = search_x1 + x1 + w//2
                    dot_y = search_y1 + y1 + h//2
                    
                    cv2.circle(potential_dots_img,
                             (dot_x, dot_y),
                             max(w, h)//2,
                             (0, 255, 0), 2)
                    cv2.line(potential_dots_img,
                            (x, y),
                            (dot_x, dot_y),
                            (0, 255, 0), 1)
                    
                    dot_detections.append({
                        'bbox': [dot_x - w//2, dot_y - h//2, dot_x + w//2, dot_y + h//2],
                        'confidence': circularity,
                        'symbol_type': symbol['type'],
                        'symbol_pos': (x, y)
                    })
    
    # Visualization
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(30, 20))
        fig.suptitle("Augmentation Dot Detection Process", fontsize=16)
        
        axes[0, 0].imshow(debug_img)
        axes[0, 0].set_title("Original Image")
        
        axes[0, 1].imshow(search_areas_img)
        axes[0, 1].set_title("Search Areas\nYellow: Search regions, Blue: Notes, Red: Rests")
        
        axes[1, 0].imshow(potential_dots_img)
        axes[1, 0].set_title("Potential Dots\nGreen: Detected dots with connections to symbols")
        
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title("Binary Image Used for Detection")
        
        plt.tight_layout()
        plt.show()
    
    return dot_detections

def assign_augmentation_dots_to_notes_and_rests(notehead_data, rest_data, dot_detections, staffs, visualize=False):
    """
    Assign augmentation dots to their corresponding notes and rests and adjust durations
    using staff unit size for adaptive measurements
    """
    print("\nProcessing augmentation dots...")
    
    # Calculate average unit size across all staffs
    if staffs:
        unit_sizes = [staff.unit_size for staff in staffs]
        avg_unit_size = sum(unit_sizes) / len(unit_sizes)
        print(f"Average staff unit size: {avg_unit_size:.2f} pixels")
    else:
        avg_unit_size = 10
        print(f"Warning: No staffs available, using default unit size: {avg_unit_size}")
    
    # Define adaptive parameters based on unit size
    max_horizontal_distance = avg_unit_size * 3
    max_vertical_distance = avg_unit_size * 1
    
    print(f"Using adaptive distances based on unit size:")
    print(f"  Max horizontal distance: {max_horizontal_distance:.2f} pixels")
    print(f"  Max vertical distance: {max_vertical_distance:.2f} pixels")
    
    # Convert dot detections to center points
    dots = []
    for det in dot_detections:
        x1, y1, x2, y2 = det['bbox']
        dots.append({
            'x': (x1 + x2) / 2,
            'y': (y1 + y2) / 2,
            'confidence': det['confidence'],
            'assigned': False
        })
    
    # Combine notes and rests into a single list of musical symbols
    musical_symbols = []
    for note in notehead_data:
        # Initialize is_dotted field
        if 'is_dotted' not in note:
            note['is_dotted'] = False
        musical_symbols.append({
            'type': 'note',
            'data': note,
            'x': note['x'],
            'y': note['y'],
            'staff': next((s for s in staffs if s.track == note['track'] and s.group == note['group']), None)
        })
    for rest in rest_data:
        if 'is_dotted' not in rest:
            rest['is_dotted'] = False
        musical_symbols.append({
            'type': 'rest',
            'data': rest,
            'x': rest['x'],
            'y': rest['y'],
            'staff': next((s for s in staffs if s.track == rest['track'] and s.group == rest['group']), None)
        })
    
    musical_symbols.sort(key=lambda x: x['x'])
    
    # For each symbol, find the closest dot to its right
    for symbol in musical_symbols:
        symbol_x = symbol['x']
        symbol_y = symbol['y']
        
        if symbol['staff']:
            unit_size = symbol['staff'].unit_size
        else:
            unit_size = avg_unit_size
            
        symbol_max_h_distance = unit_size * 3.0
        symbol_max_v_distance = unit_size * 1.0
        
        closest_dot = None
        min_score = float('inf')  # Lower score is better
        
        for i, dot in enumerate(dots):
            if dot['assigned']:
                continue
                
            # Only consider dots to the right of the symbol
            dx = dot['x'] - symbol_x
            if dx <= 0:
                continue
            
            # Check if dot is within horizontal distance threshold
            if dx > symbol_max_h_distance:
                continue
                
            # Check vertical distance
            dy = abs(dot['y'] - symbol_y)
            if dy > symbol_max_v_distance:
                continue
            
            # Calculate a score that prioritizes:
            # 1. Dots that are closer horizontally (but not too close)
            # 2. Dots that are very close vertically
            # Ideal horizontal distance is about 0.75-1 unit
            h_score = abs(dx - unit_size * 0.85) / unit_size  # Normalized distance from ideal
            v_score = dy / unit_size  # Normalized vertical distance
            
            # Combined score (lower is better)
            score = h_score * 0.7 + v_score * 1.3  # Weight vertical alignment more
            
            if score < min_score:
                min_score = score
                closest_dot = (i, dot)
        
        # If dot found, assign to the symbol and adjust duration
        if closest_dot and min_score < 2.0:  # Score threshold
            dot_idx, dot = closest_dot
            dots[dot_idx]['assigned'] = True
            
            symbol_data = symbol['data']
            
            # Mark as dotted and increase duration
            symbol_data['is_dotted'] = True
            original_duration = symbol_data['duration']
            symbol_data['duration'] *= 1.5
            
            symbol_type = symbol['type']
            if symbol_type == 'note':
                print(f"Note at ({symbol_x:.1f}, {symbol_y:.1f}) - {symbol_data['pitch']}")
            else:
                print(f"Rest at ({symbol_x:.1f}, {symbol_y:.1f}) - {symbol_data['type']}")
            print(f"  Found dot at ({dot['x']:.1f}, {dot['y']:.1f})")
            print(f"  Score: {min_score:.3f} (lower is better)")
            print(f"  Duration: {original_duration} -> {symbol_data['duration']}")
    
    unassigned_dots = [d for d in dots if not d['assigned']]
    if unassigned_dots:
        print(f"\nWarning: {len(unassigned_dots)} dots were not assigned to any symbol")
        for dot in unassigned_dots:
            print(f"  Unassigned dot at ({dot['x']:.1f}, {dot['y']:.1f})")
    
    return notehead_data, rest_data

def visualize_augmented_notes_and_rests(img, notehead_data, rest_data, dot_detections):
    """Visualize notes and rests with their augmentation dots"""
    result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    # Draw notes
    for note in notehead_data:
        cv2.circle(result_img,
                  (int(note['x']), int(note['y'])),
                  5, (255, 0, 0), -1)
        
        label = f"{note['pitch']} ({note['duration']:.1f})"
        cv2.putText(result_img,
                   label,
                   (int(note['x']), int(note['y'] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (255, 0, 0),
                   1)
    
    # Draw rests
    for rest in rest_data:
        cv2.circle(result_img,
                  (int(rest['x']), int(rest['y'])),
                  5, (0, 0, 255), -1)
        
        label = f"Rest ({rest['duration']:.1f})"
        cv2.putText(result_img,
                   label,
                   (int(rest['x']), int(rest['y'] - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   (0, 0, 255),
                   1)
    
    # Draw dots
    for det in dot_detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(result_img,
                     (x1, y1),
                     (x2, y2),
                     (0, 255, 0),
                     1)
    
    plt.figure(figsize=(30,20))
    plt.imshow(result_img)
    plt.title("Notes and Rests with Augmentation Dots\nBlue: Notes, Red: Rests, Green: Augmentation dots", fontsize=16)
    plt.show()

def detect_and_process_rhythm_elements(img, notehead_data, staffs, rhythm_model, visualize=False):
    """Detect beams, flags and other rhythm elements and update note durations accordingly"""
    print("\nDetecting rhythm elements (beams, flags)...")
    
    note_width = calculate_average_note_width(notehead_data)
    
    # Crop image into staff groups for better detection
    cropped_images, crop_regions = crop_staff_groups(img, staffs)
    
    # Store all detections
    all_detections = []
    debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    # Process each staff group
    for crop_info in cropped_images:
        group_num = crop_info['group']
        cropped = crop_info['image']
        y_offset = crop_info['y_offset']
        
        print(f"\nProcessing staff group {group_num}")
        
        # Run model
        results = rhythm_model(cropped, conf=0.40)[0]
        
        group_detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                
                # Adjust coordinates to original image
                orig_y1 = y1 + y_offset
                orig_y2 = y2 + y_offset
                
                element_type = get_element_type(class_name)
                
                # Create detection object
                detection = {
                    'x1': x1,
                    'y1': orig_y1,
                    'x2': x2,
                    'y2': orig_y2,
                    'confidence': confidence,
                    'class': class_name,
                    'type': element_type,
                    'group': group_num
                }
                
                group_detections.append(detection)
        
        # Merge overlapping beams for this group
        group_detections = merge_overlapping_beams(group_detections)
        all_detections.extend(group_detections)
        
        if visualize:
            for det in group_detections:
                color = get_element_color(det['type'])
                cv2.rectangle(debug_img,
                            (int(det['x1']), int(det['y1'])),
                            (int(det['x2']), int(det['y2'])),
                            color, 1)
                
                label = f"G{group_num} {det['class']} ({det['confidence']:.2f})"
                cv2.putText(debug_img,
                           label,
                           (int(det['x1']), int(det['y1']-5)),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5,
                           color,
                           1)
    
    # Process the detections and update note durations
    updated_notehead_data = process_rhythm_elements(notehead_data, all_detections, staffs)
    
    if visualize:
        plt.figure(figsize=(30,20))
        plt.imshow(debug_img)
        plt.title("Rhythm Elements Detection", fontsize=16)
        plt.show()
        
        result_img = visualize_rhythm_analysis(img, updated_notehead_data, all_detections)
        
    return updated_notehead_data, all_detections, debug_img

def calculate_average_note_width(notehead_data):
    """Calculate the average width of notes directly from bounding box information"""
    
    # Extract width from bounding box information
    note_widths = []
    for note in notehead_data:
        if 'width' in note:
            note_widths.append(note['width'])
        elif 'bbox' in note:
            x1, _, x2, _ = note['bbox']
            width = x2 - x1
            note_widths.append(width)
    
    if note_widths:
        avg_width = sum(note_widths) / len(note_widths)
        print(f"Average note width: {avg_width:.2f} pixels")
        return avg_width
    else:
        print("Warning: No bounding box information found")
        return 10  # Default fallback

def get_element_type(class_name):
    """Determine the type of rhythm element from its class name"""
    class_name = class_name.lower()
    if 'beam' in class_name:
        return 'beam'
    elif 'flag' in class_name:
        return 'flag'
    elif 'tuplet' in class_name or 'triplet' in class_name:
        return 'tuplet'
    else:
        return 'other'

def get_element_color(element_type):
    """Return color for visualization based on element type"""
    colors = {
        'beam': (255, 0, 0),    # Red
        'flag': (0, 255, 0),    # Green
        'tuplet': (0, 0, 255),  # Blue
        'other': (128, 128, 0)  # Yellow
    }
    return colors.get(element_type, (255, 255, 255))

def crop_staff_groups(img, staffs):
    """Crop image into individual staff group regions with padding"""
    staff_groups = {}
    padding = 50  # Add padding above and below each group (Not yet adaptive)
    
    # Group staffs by group number
    for staff in staffs:
        if staff.group not in staff_groups:
            staff_groups[staff.group] = []
        staff_groups[staff.group].append(staff)
    
    # Crop image for each group
    cropped_images = []
    crop_regions = [] 
    
    for group_num, group_staffs in sorted(staff_groups.items()):
        # Find group boundaries
        min_y = max(0, int(min(staff.upper_bound for staff in group_staffs) - padding))
        max_y = min(img.shape[0], int(max(staff.lower_bound for staff in group_staffs) + padding))
        
        # Crop image
        cropped = img[min_y:max_y, :]
        cropped_images.append({
            'group': group_num,
            'image': cropped,
            'y_offset': min_y  
        })
        crop_regions.append({
            'group': group_num,
            'min_y': min_y,
            'max_y': max_y
        })
    
    return cropped_images, crop_regions

def merge_overlapping_beams(detections):
    """Merge overlapping beams and handle partial overlaps"""
    # Separate beams from other detections
    beams = [d for d in detections if d['type'] == 'beam']
    other_elements = [d for d in detections if d['type'] != 'beam']
    
    # Group beams by vertical position (y-coordinate)
    # This allows us to distinguish between multiple beams at different vertical positions
    beam_groups = {}
    for beam in beams:
        # Use the average y-position as the key for grouping
        y_pos = (beam['y1'] + beam['y2']) / 2
        # Round
        group_key = round(y_pos / 10) * 10
        
        if group_key not in beam_groups:
            beam_groups[group_key] = []
        beam_groups[group_key].append(beam)
    
    # Process each group of beams separately
    merged_beams = []
    for group_key, group_beams in beam_groups.items():
        # Sort beams by x position
        group_beams.sort(key=lambda x: x['x1'])
        
        # Merge overlapping beams
        if group_beams:
            current_beam = group_beams[0]
            
            for beam in group_beams[1:]:
                # Check if beams overlap horizontally
                if beam['x1'] <= current_beam['x2'] + 5:  # Allow small gap
                    # Merge beams
                    current_beam['x2'] = max(current_beam['x2'], beam['x2'])
                    # Take highest confidence
                    current_beam['confidence'] = max(current_beam['confidence'], beam['confidence'])
                else:
                    # No overlap, add current beam to results and start new one
                    merged_beams.append(current_beam)
                    current_beam = beam
            
            merged_beams.append(current_beam)
    
    # Combine merged beams with other elements
    return merged_beams + other_elements

def find_beams_for_note(note, rhythm_detections, group, note_width, staffs):
    """Count how many beams are connected to a note based on staff track"""
    beam_count = 0
    note_x, note_y = note['x'], note['y']
    note_track = note['track']
    note_group = note['group']
    
    # Find the staff for this note
    current_staff = None
    for staff in staffs:
        if staff.group == note_group and staff.track == note_track:
            current_staff = staff
            break
    
    if not current_staff:
        return 0
    
    # Use unit size for extension calculation
    ledger_extension = current_staff.unit_size * 2
    
    # Get all beams that might be connected to this note
    note_beams = []
    
    # Process each beam
    for det in rhythm_detections:
        if det['type'] != 'beam' or det['group'] != note_group:
            continue
            
        beam_y = (det['y1'] + det['y2']) / 2
        
        # Extended vertical range using unit size
        extended_upper = current_staff.upper_bound - ledger_extension
        extended_lower = current_staff.lower_bound + ledger_extension
        
        # Check if beam is within the extended staff area
        if not (extended_upper <= beam_y <= extended_lower):
            continue  # Skip beams that aren't in this staff's extended vertical range
        
        # Check horizontal alignment
        horizontal_padding = note_width * 2  
        if (det['x1'] - horizontal_padding <= note_x <= det['x2'] + horizontal_padding):
            note_beams.append(det)
    
    # Count beams at different vertical positions
    if note_beams:
        # Group beams by vertical position
        y_positions = set()
        for beam in note_beams:
            beam_y = (beam['y1'] + beam['y2']) / 2
            # Round to nearest 5 pixels to allow for slight variations
            y_pos = round(beam_y / 5) * 5
            y_positions.add(y_pos)
        
        # Number of distinct vertical positions = number of beams
        beam_count = len(y_positions)
        
        print(f"Note at x={note_x:.1f}, y={note_y:.1f} has {beam_count} beams at {len(note_beams)} positions")
    
    return beam_count

def find_closest_note(notehead_data, element, group):
    """Find the closest note to a given element"""
    min_dist = float('inf')
    closest_note = None
    element_x = (element['x1'] + element['x2']) / 2
    element_y = (element['y1'] + element['y2']) / 2
    
    for note in notehead_data:
        if note['group'] != group:
            continue
        
        dist = np.sqrt((note['x'] - element_x)**2 + (note['y'] - element_y)**2)
        if dist < min_dist:
            min_dist = dist
            closest_note = note
    
    return closest_note

def update_note_duration(note, beam_count):
    """Update note duration based on number of beams, preserving augmentation dots"""
    # Base duration (quarter note)
    base_duration = 1.0
    
    # Calculate new base duration based on beam count
    if beam_count == 1:
        # Single beam = eighth note
        new_duration = base_duration / 2  # 0.5
    elif beam_count == 2:
        # Double beam = sixteenth note
        new_duration = base_duration / 4  # 0.25
    elif beam_count == 3:
        # Triple beam = thirty-second note
        new_duration = base_duration / 8  # 0.125
    else:
        new_duration = base_duration
    
    # If the note is dotted, add half of the new duration
    if note.get('is_dotted', False):
        new_duration = new_duration * 1.5
    
    note['duration'] = new_duration
    return note

def process_rhythm_elements(notehead_data, rhythm_detections, staffs):
    """Process detected rhythm elements and update note durations"""
    print("\nProcessing rhythm elements...")
    
    # Calculate average note width once
    note_width = calculate_average_note_width(notehead_data)
    
    # Group rhythm elements by staff group
    elements_by_group = {}
    for det in rhythm_detections:
        group = det['group']
        if group not in elements_by_group:
            elements_by_group[group] = []
        elements_by_group[group].append(det)
    
    updated_notehead_data = notehead_data.copy()
    
    # Process each group
    for group, elements in elements_by_group.items():
        print(f"\nProcessing Group {group}")
        
        # Process each note to count its beams
        for note in updated_notehead_data:
            if note['group'] != group:
                continue
        
            beam_count = find_beams_for_note(note, elements, group, note_width, staffs)
            if beam_count > 0:
                update_note_duration(note, beam_count)
        
        # Process flags
        flags = [e for e in elements if e['type'] == 'flag']
        for flag in flags:
            # Find closest note to this flag
            note = find_closest_note(updated_notehead_data, flag, group)
            if note:
                # Update duration based on flag (treat as single beam)
                update_note_duration(note, 1)
    
    return updated_notehead_data

def visualize_rhythm_analysis(img, notehead_data, rhythm_detections):
    """Visualize notes with their durations and detected rhythm elements"""
    result_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    for det in rhythm_detections:
        if 'augmentationdot' in det['class'].lower():
            continue
            
        color = get_element_color(det['type'])
        cv2.rectangle(result_img,
                     (int(det['x1']), int(det['y1'])),
                     (int(det['x2']), int(det['y2'])),
                     color, 1)
  
        label = f"{det['class']}"
        if det['type'] == 'beam':
            y_offset = -20
        elif det['type'] == 'flag':
            y_offset = -35
        else:  # tuplet
            y_offset = -50
            
        cv2.putText(result_img,
                   label,
                   (int(det['x1']), int(det['y1'] + y_offset)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4,  # Smaller font size
                   color,
                   1)
    
    # Draw notes with their durations
    for note in notehead_data:
        # Draw note position
        cv2.circle(result_img,
                  (int(note['x']), int(note['y'])),
                  3,  # Smaller circle
                  (255, 0, 0),
                  -1)
        
        pitch_label = f"{note['pitch']}"
        duration_label = f"({note['duration']:.2f})"
        
        cv2.putText(result_img,
                   pitch_label,
                   (int(note['x'] - 20), int(note['y'] + 15)),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4,  
                   (255, 0, 0),
                   1)
        cv2.putText(result_img,
                   duration_label,
                   (int(note['x'] - 20), int(note['y'] + 30)),  
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4, 
                   (255, 0, 0),
                   1)
    
    legend_y = 30
    legend_x = 50
    cv2.putText(result_img,
               "Legend:",
               (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX,
               0.5,
               (0, 0, 0),
               1)
    
    legend_items = [
        ("Notes", (255, 0, 0)),
        ("Beams", get_element_color('beam')),
        ("Flags", get_element_color('flag')),
        ("Tuplets", get_element_color('tuplet'))
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y + 20 * (i + 1)
        cv2.putText(result_img,
                   f"• {label}",
                   (legend_x + 10, y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.4,
                   color,
                   1)
    
    plt.figure(figsize=(30,20))
    plt.imshow(result_img)
    plt.title("Rhythm Analysis Results", fontsize=16)
    plt.show()
    
    return result_img


######################################################## Start of step 6 ########################################################

def group_into_chords(events, staffs=None, threshold_multiplier=1.5):
    """
    Group notes or rests that occur at approximately the same time into chords
    
    Args:
        events: List of note or rest events to group
        staffs: List of Staff objects to determine unit size
        threshold_multiplier: Multiplier for unit size to determine threshold
        
    Returns:
        List of grouped events (chords)
    """
    if not events:
        return []
    
    # Determine appropriate threshold based on average staff unit size
    threshold = 15 
    
    if staffs and isinstance(staffs, list) and staffs:
        # Calculate average unit size across all staffs
        unit_sizes = [staff.unit_size for staff in staffs]
        avg_unit_size = sum(unit_sizes) / len(unit_sizes)
        threshold = avg_unit_size * threshold_multiplier
        print(f"Using average staff unit size for chord grouping: {threshold:.2f} pixels")
    else:
        print(f"No staff information available, using default threshold: {threshold} pixels")
    
    # Sort events by x position
    events = sorted(events, key=lambda x: x['x'])
    chords = []
    current_chord = [events[0]]
    
    for event in events[1:]:
        if abs(event['x'] - current_chord[0]['x']) <= threshold:
            current_chord.append(event)
        else:
            chords.append(current_chord)
            current_chord = [event]
    chords.append(current_chord)
    
    return chords

def validate_bar_durations(staffs, notehead_data, rest_data, bar_lines, time_signature=(3, 4), visualize=True):
    """
    Validate and normalize note/rest durations in each bar to match the expected duration
    based on the time signature and detected bar lines.
    """
    print("\nValidating bar durations...")
    
    # Calculate expected duration per bar in quarter notes
    numerator, denominator = time_signature
    expected_duration = (numerator * 4.0) / denominator
    print(f"Time signature: {numerator}/{denominator}")
    print(f"Expected duration per bar: {expected_duration} quarter notes")
    
    # Sort bar lines by x position within each group
    bar_lines_by_group = {}
    for bar in bar_lines:
        if bar['group'] not in bar_lines_by_group:
            bar_lines_by_group[bar['group']] = []
        bar_lines_by_group[bar['group']].append(bar['x'])
    
    for group in bar_lines_by_group:
        bar_lines_by_group[group].sort()
    
    # Group notes and rests by track
    track_contents = {}
    for staff in staffs:
        track_contents[staff.track] = {'notes': [], 'rests': []}
    
    for note in notehead_data:
        track_contents[note['track']]['notes'].append(note)
    for rest in rest_data:
        track_contents[rest['track']]['rests'].append(rest)
    
    validation_results = []
    
    visualization_img = None
    
    # Check each track
    for track, contents in track_contents.items():
        print(f"\nChecking Track {track}:")
        
        # Check each group 
        for group, bar_x_positions in bar_lines_by_group.items():
            # Create measure boundaries
            measures = []
            for i in range(len(bar_x_positions) - 1):
                measures.append((bar_x_positions[i], bar_x_positions[i + 1]))
            
            for measure_idx, (start_x, end_x) in enumerate(measures):
                # Get notes and rests within this measure
                measure_notes = [n for n in contents['notes'] 
                              if start_x <= n['x'] < end_x and n['group'] == group]
                measure_rests = [r for r in contents['rests'] 
                              if start_x <= r['x'] < end_x and r['group'] == group]
                
                # If measure is empty, add a full measure rest
                if not measure_notes and not measure_rests:
                    print(f"\nMeasure {measure_idx + 1}, Track {track} is empty - adding full measure rest")
                    
                    staff = next((s for s in staffs if s.track == track and s.group == group), None)
                    
                    if staff:
                        full_measure_rest = {
                            'x': (start_x + end_x) / 2,  
                            'y': staff.center,
                            'duration': expected_duration,
                            'type': 'whole',
                            'track': track,
                            'group': group,
                            'is_dotted': False
                        }
                        measure_rests.append(full_measure_rest)
                        rest_data.append(full_measure_rest)  # Add to rest data
    
                current_staff = next((s for s in staffs if s.track == track and s.group == group), None)
                
                # Combine notes and rests into events
                measure_events = []
                for note in measure_notes:
                    measure_events.append({
                        'type': 'note',
                        'data': note,
                        'x': note['x'],
                        'track': track,
                        'group': group
                    })
                for rest in measure_rests:
                    measure_events.append({
                        'type': 'rest',
                        'data': rest,
                        'x': rest['x'],
                        'track': track,
                        'group': group
                    })
                
                # Group events consistently
                event_groups = group_into_chords(measure_events, staffs)
                
                # Calculate total duration
                total_duration = 0
                for event_group in event_groups:
                    max_duration = 0
                    for event in event_group:
                        max_duration = max(max_duration, event['data']['duration'])
                    total_duration += max_duration
                
                # If duration is incorrect, normalize the durations
                if abs(total_duration - expected_duration) > 0.01:
                    print(f"\nNormalizing durations in Measure {measure_idx + 1}, Track {track}:")
                    print(f"Current total: {total_duration:.2f}, Expected: {expected_duration:.2f}")
                    
                    # Calculate scaling factor
                    if total_duration > 0:
                        scale_factor = expected_duration / total_duration
                        
                        for event_group in event_groups:
                            for event in event_group:
                                old_duration = event['data']['duration']
                                event['data']['duration'] *= scale_factor
                                event_type = "note" if event['type'] == 'note' else "rest"
                                event_desc = event['data'].get('pitch', event['data'].get('type', 'unknown'))
                                print(f"  Adjusted {event_type} {event_desc}: {old_duration:.2f} -> {event['data']['duration']:.2f}")
                    
                    # Recalculate total duration
                    total_duration = 0
                    for event_group in event_groups:
                        max_duration = 0
                        for event in event_group:
                            max_duration = max(max_duration, event['data']['duration'])
                        total_duration += max_duration
            
                for i, event_group in enumerate(event_groups):
                    group_duration = max(event['data']['duration'] for event in event_group)
                    notes_in_group = [event['data']['pitch'] for event in event_group if event['type'] == 'note']
                    rests_in_group = [event['data']['type'] for event in event_group if event['type'] == 'rest']
                    
                    if notes_in_group:
                        print(f"  Measure {measure_idx + 1} Event {i+1}: Notes {notes_in_group} (duration: {group_duration:.2f})")
                    if rests_in_group:
                        print(f"  Measure {measure_idx + 1} Event {i+1}: Rests {rests_in_group} (duration: {group_duration:.2f})")
                
                # Validate final duration
                is_valid = abs(total_duration - expected_duration) < 0.01

                note_chords = []
                rest_groups = []
                
                for event_group in event_groups:
                    notes = [event['data'] for event in event_group if event['type'] == 'note']
                    rests = [event['data'] for event in event_group if event['type'] == 'rest']
                    
                    if notes:
                        note_chords.append(notes)
                    if rests:
                        rest_groups.append(rests)
                
                result = {
                    'track': track,
                    'group': group,
                    'measure': measure_idx + 1,
                    'start_x': start_x,
                    'end_x': end_x,
                    'total_duration': total_duration,
                    'expected_duration': expected_duration,
                    'is_valid': is_valid,
                    'note_chords': note_chords,
                    'rest_groups': rest_groups
                }
                validation_results.append(result)
   
                print(f"\nMeasure {measure_idx + 1}, Track {track}:")
                print(f"Total duration: {total_duration:.2f} quarter notes")
                print(f"Status: {'✓ Valid' if is_valid else '✗ Invalid'}")
                
    return validation_results, notehead_data, rest_data, visualization_img

def create_midi_from_score(notehead_data, rest_data, output_filename="score.mid", staffs=None):
    """Convert detected notes and rests into a MIDI file"""
    print("\nCreating MIDI file...")
    
    # Create MIDI file with 2 tracks
    midi = MIDIFile(2)  # Two tracks: 0 for treble, 1 for bass
    tempo = 120
    volume = 100
    
    # Add tempo to both tracks
    midi.addTempo(0, 0, tempo)
    midi.addTempo(1, 0, tempo)
    
    def pitch_to_midi_note(pitch_str):
        """Convert pitch string (e.g., 'B4flat') to MIDI note number"""
        match = re.match(r'([A-G])(\d+)(sharp|flat)?', pitch_str)
        if not match:
            return None
        
        note, octave, accidental = match.groups()
        base_notes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
        midi_note = base_notes[note] + (int(octave) + 1) * 12
        
        if accidental == 'sharp':
            midi_note += 1
        elif accidental == 'flat':
            midi_note -= 1
            
        return midi_note

    def group_into_events(notes, rests, staffs=None):
        """Group notes and rests into sequential events based on x-position using adaptive threshold"""
        # Combine notes and rests into events
        events = []
        
        # Add notes as note events
        for note in notes:
            events.append({
                'type': 'note',
                'data': note,
                'x': note['x'],
                'track': note.get('track'),
                'group': note.get('group')
            })
            
        # Add rests as rest events
        for rest in rests:
            events.append({
                'type': 'rest',
                'data': rest,
                'x': rest['x'],
                'track': rest.get('track'),
                'group': rest.get('group')
            })
            
        # Sort all events by x position
        events.sort(key=lambda e: e['x'])
    
        return group_into_chords(events, staffs)
    
    # Group events by track and measure
    track_groups = {}
    for note in notehead_data:
        key = (note['track'] - 1, note['group']) 
        if key not in track_groups:
            track_groups[key] = {'notes': [], 'rests': []}
        track_groups[key]['notes'].append(note)
    
    for rest in rest_data:
        key = (rest['track'] - 1, rest['group']) 
        if key not in track_groups:
            track_groups[key] = {'notes': [], 'rests': []}
        track_groups[key]['rests'].append(rest)
    
    # Process each track and measure
    track_times = {0: 0.0, 1: 0.0}
    
    print("\nProcessing events by track and measure...")
    for (track, group), events in sorted(track_groups.items()):
        print(f"\nTrack {track+1}, Measure {group}:")
        
        # Group notes and rests into sequential events
        grouped_events = group_into_events(events['notes'], events['rests'], staffs)
        
        # Process each event group
        current_time = track_times[track]
        for event_group in grouped_events:
            print(f"\nEvent group at time {current_time:.2f}:")
            
            # Track max duration
            max_duration = 0
            
            # Process all events in the group
            for event in event_group:
                if event['type'] == 'note':
                    note = event['data']
                    midi_note = pitch_to_midi_note(note['pitch'])
                    if midi_note is None:
                        continue
                    
                    try:
                        midi.addNote(
                            track=track,
                            channel=0,
                            pitch=midi_note,
                            time=current_time,
                            duration=note['duration'],
                            volume=volume
                        )
                        print(f"  Added note {note['pitch']} (duration: {note['duration']})")
                        max_duration = max(max_duration, note['duration'])
                    except Exception as e:
                        print(f"  Error adding note {note['pitch']}: {e}")
                
                elif event['type'] == 'rest':
                    rest = event['data']
                    print(f"  Added rest (duration: {rest['duration']})")
                    max_duration = max(max_duration, rest['duration'])
            
            # Move time forward by the longest duration in the group !!!!
            if max_duration > 0:
                current_time += max_duration
        
        track_times[track] = current_time
    
    # Write file
    try:
        with open(output_filename, "wb") as f:
            midi.writeFile(f)
        print(f"\nSuccessfully created MIDI file: {output_filename}")
        
        print("\nMIDI Creation Summary:")
        print(f"Total notes processed: {len(notehead_data)}")
        print(f"Total rests processed: {len(rest_data)}")
        for track in [0, 1]:
            print(f"Track {track+1} duration: {track_times[track]:.2f} beats")
        
        return True
    except Exception as e:
        print(f"Error writing MIDI file: {e}")
        return False
    
######################################################## END ########################################################

def process_sheet_music(image_path, output_midi_path="output_score.mid", 
                        staffline_model_path="finaltest_unet_modelweighted500focaltverskyGAMMA.h5",
                        notehead_model_path="best.pt",
                        accidental_model_path="best_accidental.pt",
                        rhythm_model_path="best_rhythm.pt",
                        rest_model_path="best_rests.pt",
                        time_signature=(3, 4),
                        visualize=False):
    """
    Process a sheet music image and convert it to MIDI.
    
    Args:
        image_path: Path to the sheet music image
        output_midi_path: Path to save the output MIDI file
        staffline_model_path: Path to the U-Net model for staffline detection
        notehead_model_path: Path to the YOLO model for notehead detection
        accidental_model_path: Path to the YOLO model for accidental detection
        rhythm_model_path: Path to the YOLO model for rhythm element detection
        rest_model_path: Path to the YOLO model for rest detection
        time_signature: Tuple of (numerator, denominator) for time signature
        visualize: Whether to show visualization plots
    
    Returns:    
        True if successful, False otherwise
    """
    print(f"\n=== Processing sheet music: {image_path} ===")
    
    # Load the image and create inverted version
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    #img = cv2.resize(img, (1200, 1700))
    #img = cv2.resize(img, (1700, 2200))
    #img = cv2.resize(img, (2200, 2700))
    #resized_path = f"resized_{os.path.basename(image_path)}"
    resized_path = image_path
    #cv2.imwrite(resized_path, img)
    
    # Create inverted image for staffline detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)
    #inverted_img = cv2.equalizeHist(inverted_img)
    inverted_path = f"inverted_{os.path.basename(image_path)}"
    cv2.imwrite(inverted_path, inverted_img)
            # Apply contrast enhancement
    
        
    if visualize:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.subplot(122)
        plt.imshow(inverted_img, cmap='gray')
        plt.title("Inverted Image")
        plt.show()
    
    # Load models
    print("\nLoading models...")
    try:
        unet_model = load_model(staffline_model_path, compile=False)
        yolo_model = YOLO(notehead_model_path)
        model_accidental = YOLO(accidental_model_path)
        rhythm_model = YOLO(rhythm_model_path)
        rest_model = YOLO(rest_model_path)
        print("All models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
    
    # STEP 1: Removing the Stafflines and group the lines into Groups and assign Pitches to the lines
    print("\nExtracting stafflines...")
    staffs, pred_mask = extract_stafflines(unet_model, inverted_path, chunk_size=256, visualize=visualize)
    if not staffs:
        print("Error: No stafflines detected")
        return False
    
    print(f"Detected {len(staffs)} staffs")
    
    # STEP 2: Detect Note Symbols, assign pitches based on position.
    print("\nDetecting and processing noteheads...")
    notehead_data, notehead_img = detect_and_process_noteheads(resized_path, yolo_model, staffs, visualize)
    
    print("\nGrouping notes into chords...")
    note_groups, group_vis_img = group_notes_into_chords(notehead_data, img, visualize)
    print("\nDetected Note Groups:")
    for group in note_groups:
        print(group)
    
    # STEP 3: Change pitch of notes based on Key and Accidentals
    print("\nDetecting bar lines...")
    bar_lines, bar_line_img = detect_bar_lines(img, staffs, visualize)
    
    print("\nDetecting accidentals...")
    accidentals, accidental_img = detect_accidentals(img, staffs, model_accidental, visualize)

    print("\nProcessing key signatures...")
    process_key_signatures(staffs, accidentals, visualize)
    if visualize:
        print("\nVisualizing key signatures...")
        key_sig_img = visualize_key_signatures(img, staffs, visualize)
     
    print("\nUpdating note pitches based on key signatures...")
    update_note_pitches_with_key_signatures(staffs, notehead_data)
    
    if visualize:
        print("\nVisualizing updated notes...")
        updated_notes_img = visualize_updated_notes(img, notehead_data, visualize)
 
    print("\nProcessing accidentals within measures...")
    notehead_data, accidental_debug_img = process_accidentals_within_measures(
        staffs, accidentals, bar_lines, notehead_data, img, visualize)
    
    # STEP 4: Detect Break Symbols, assign durations to breaks and notes based on class type
    print("\nDetecting and processing rests...")
    rest_data, rest_img = detect_and_process_rests(img, staffs, rest_model, notehead_data, visualize)
    
    # STEP 5: Detect Rhythmic Element like Dots and Beams and assign them to their notes by updating their durations.
    print("\nDetecting and processing augmentation dots...")
    staff_height = staffs[0].unit_size * 4
    dot_detections = detect_augmentation_dots(img, notehead_data, rest_data, staff_height=staff_height)

    # Process the dots
    notehead_data, rest_data = assign_augmentation_dots_to_notes_and_rests(
        notehead_data, rest_data, dot_detections, staffs, visualize
    )
    if visualize:
        visualize_augmented_notes_and_rests(img, notehead_data, rest_data, dot_detections)
    # Detect Beams and Flags    
    updated_notehead_data, rhythm_detections, debug_img = detect_and_process_rhythm_elements(
        img, notehead_data, staffs, rhythm_model, visualize)

    # STEP 6: Validate durations and convert collected data to midi
    validation_results, final_notehead_data, final_rest_data, validation_img = validate_bar_durations(
        staffs, updated_notehead_data, rest_data, bar_lines, time_signature)
    
    # Create MIDI file
    print("\nCreating MIDI file...")
    midi_success = create_midi_from_score(final_notehead_data, final_rest_data, output_midi_path, staffs)
    
    if midi_success:
        print(f"\n=== Successfully processed sheet music and created MIDI file: {output_midi_path} ===")
        return True
    else:
        print("\n=== Failed to create MIDI file ===")
        return False
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sheet music and convert to MIDI')
    parser.add_argument('--image', type=str, default="TestImages/FINAL_ONE_PB_page-0001.jpg", help='Path to sheet music image')
    parser.add_argument('--output', type=str, default="OutputMidi/fobeamnew.mid", help='Path to output MIDI file')
    #parser.add_argument('--staffline_model', type=str, default="best_unet_model.h5", help='Path to staffline detection model')
    parser.add_argument('--staffline_model', type=str, default="finaltest_unet_modelweighted500focaltverskyGAMMA.h5", help='Path to staffline detection model')
    parser.add_argument('--notehead_model', type=str, default="best.pt", help='Path to notehead detection model')
    parser.add_argument('--accidental_model', type=str, default="best_accidental.pt", help='Path to accidental detection model')
    parser.add_argument('--rhythm_model', type=str, default="best_rythm.pt", help='Path to rhythm element detection model')
    parser.add_argument('--rest_model', type=str, default="best_rests.pt", help='Path to rest detection model')
    parser.add_argument('--time_signature', type=str, default="3/4", help='Time signature in format "numerator/denominator"')
    parser.add_argument('--no_visualize', action='store_true', help='Disable visualization plots')
    
    args = parser.parse_args()
    # get time signature
    try:
        num, denom = map(int, args.time_signature.split('/'))
        time_sig = (num, denom)
    except:
        print(f"Invalid time signature format: {args.time_signature}. Using default 3/4.")
        time_sig = (4, 4)

    # Process the image
    process_sheet_music(
        image_path=args.image,
        output_midi_path=args.output,
        staffline_model_path=args.staffline_model,
        notehead_model_path=args.notehead_model,
        accidental_model_path=args.accidental_model,
        rhythm_model_path=args.rhythm_model,
        rest_model_path=args.rest_model,
        time_signature=time_sig,
        visualize=False, #args.no_visualize
    )