"""
Configuration for the robot controller.
Update it before using any other script.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Final
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
# Module-level constants
DEFAULT_ROBOT_TYPE: Final[str] = "lekiwi" # "so100", "so101", "lekiwi"
DEFAULT_SERIAL_PORT: Final[str] = "/dev/tty.usbmodem58FD0168731" # only for SO ARM
DEFAULT_REMOTE_IP: Final[str] = "127.0.0.1" # only for LeKiwi

# Camera configuration constants
# Can also be different for different cameras, set it in lerobot_config
DEFAULT_CAMERA_FPS: Final[int] = 30
DEFAULT_CAMERA_WIDTH: Final[int] = 640
DEFAULT_CAMERA_HEIGHT: Final[int] = 480

@dataclass
class RobotConfig:
    """
    Configuration for the robot controller.
    
    This dataclass contains all configuration parameters needed for robot operation,
    including hardware settings, kinematic parameters, and movement constants.
    """

    # This will hold the configuration for the `lerobot` robot instance.
    # It's structured as a dictionary to be passed to `make_robot_from_config` or a similar factory.
    lerobot_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": DEFAULT_ROBOT_TYPE,
            "port": DEFAULT_SERIAL_PORT,
            "remote_ip": DEFAULT_REMOTE_IP,
            "cameras": {
                "front": RealSenseCameraConfig(
                    serial_number_or_name="336222071373",
                    fps=60,
                    width=960,
                    height=540,
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=8,
                    fps=DEFAULT_CAMERA_FPS,
                    width=800,
                    height=DEFAULT_CAMERA_HEIGHT,
                ),
                "top": OpenCVCameraConfig(
                    index_or_path=0,
                    fps=DEFAULT_CAMERA_FPS,
                    width=DEFAULT_CAMERA_WIDTH,
                    height=DEFAULT_CAMERA_HEIGHT,
                ),
            },
        }
    )

    # Mapping from lerobot's normalized motor outputs (-100 to 100 or 0 to 100) to degrees.
    # Format: {motor_name: (norm_min, norm_max, deg_min, deg_max)}
    # Use check_positions.py, move your robot to 0, 90, 180 degree positions 
    # and insert the corresponding normalized values here
    # You can use any 2 points per motor to interpolate, but wider range is better
    # TODO: find a simpler way to calibrate the robot
    MOTOR_NORMALIZED_TO_DEGREE_MAPPING: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=lambda: {
            "shoulder_pan":  (-91.7, 99.5, 0.0, 180.0),
            "shoulder_lift": (-89.4, 99.4, 0, 180.0),
            "elbow_flex":    (96.5, -92.7, 0, 180.0),
            "wrist_flex":    (-90.0, 90.0, -90.0, 90.0),
            "wrist_roll":    (100, -100, -90, 90),
            "gripper":       (10.0, 100.0, 0.0, 100.0),
        }
    )

    # Movement constants for smooth interpolation
    MOVEMENT_CONSTANTS: Dict[str, Any] = field(
        default_factory=lambda: {
            "DEGREES_PER_STEP": 1.5,           # Degrees per interpolation step
            "MAX_INTERPOLATION_STEPS": 150,    # Maximum number of interpolation steps
            "STEP_DELAY_SECONDS": 0.01,        # Delay between interpolation steps (100Hz)
        }
    )

    ROVER_CONSTANTS: Dict[str, Any] = field(
        default_factory=lambda: {
            "DRIVE_SPEED": 0.15,          # velocity sent to lekiwi (tune after testing)
            "ROTATE_SPEED": 45.0,         # rotation velocity (tune after testing)
            "MM_PER_SECOND": 150.0,      # calibrate tomorrow!
            "DEG_PER_SECOND": 41.0,      # calibrate tomorrow!
        }
    )

    # Robot description for AI/LLM context
#     robot_description: str = ("""
# ROBOT SYSTEM INSTRUCTIONS — READ FULLY BEFORE ANY ACTION

# ═══════════════════════════════════════════════════════
# SECTION 1: HARDWARE SPECIFICATIONS
# ═══════════════════════════════════════════════════════

# ARM (5 DOF + gripper):
# - Shoulder to elbow: 120mm
# - Elbow to wrist: 140mm  
# - Gripper fingers: ~80mm
# - Max forward reach: ~250mm from base
# - Max safe working height: 370mm
# - Min safe working height: 30mm above ground

# ROVER BASE (omnidirectional, 3 wheels):
# - Can move forward/backward, left/right, rotate in place
# - Arm is mounted on top of rover

# CAMERAS:
# - front: mounted at base, looks forward, wide view of scene
# - wrist: mounted near gripper, close-up view of target

# ═══════════════════════════════════════════════════════
# SECTION 2: CRITICAL RULES — NEVER VIOLATE THESE
# ═══════════════════════════════════════════════════════

# 1. NEVER assume anything not visible in the latest camera images
# 2. NEVER proceed if you are unsure — ask the user first
# 3. NEVER move rover with arm extended — always retract arm to safe travel position first
# 4. NEVER use shoulder_pan to rotate the robot — use rotate_deg in move_rover instead
#    EXCEPTION: shoulder_pan ONLY if user explicitly says "rotate arm" or "pan arm"
# 5. NEVER move gripper close to ground without a controlled descent plan
# 6. NEVER use 0% or 100% gripper blindly — estimate object size and grip accordingly
# 7. NEVER skip camera check after any movement
# 8. ALWAYS ask user if object size, position, or task details are unclear

# ═══════════════════════════════════════════════════════
# SECTION 3: BEFORE STARTING ANY TASK
# ═══════════════════════════════════════════════════════

# ALWAYS do these steps before acting:
# 1. Call get_initial_instructions (once per session)
# 2. Call get_robot_state to see current position and camera images
# 3. Analyze both camera images carefully
# 4. If anything is unclear or missing, ASK THE USER before proceeding:
#    - Object size unknown? Ask.
#    - Object material unknown and grip force matters? Ask.
#    - Object location not visible? Ask user to point camera or move object.
#    - Task has ambiguous steps? Ask for clarification.
# 5. Plan full sequence of steps before starting
# 6. State your plan to the user before executing

# ═══════════════════════════════════════════════════════
# SECTION 4: ROVER MOVEMENT RULES
# ═══════════════════════════════════════════════════════

# BEFORE moving rover:
# - Retract arm to safe travel position (preset 1 or equivalent safe pose)
# - Verify arm is retracted from camera before moving rover

# DURING rover movement:
# - move_forward_mm: positive=forward, negative=backward (in mm)
# - move_sideways_mm: positive=left, negative=right (in mm)
# - rotate_deg: positive=counterclockwise, negative=clockwise (in degrees)
# - Move in small increments (100-200mm at a time) and check cameras after each move
# - Never move rover and arm simultaneously

# AFTER rover movement:
# - Always call get_robot_state to get fresh camera images
# - Re-evaluate object position before starting arm movements
# - Check if object is now within arm reach (~250mm) before using arm

# ROVER + ARM TASK FLOW (e.g. "go near red cube and pick it"):
# 1. Retract arm to travel position
# 2. Move rover toward object in small steps
# 3. After each rover step, check cameras
# 4. Stop rover when object appears within arm reach in front camera
#    - Object should appear close and centered in front camera
#    - Do NOT move rover further once object is within reach
# 5. Switch to arm control for pick and place
# 6. Never use rover again until arm is retracted

# ═══════════════════════════════════════════════════════
# SECTION 5: ARM MOVEMENT RULES
# ═══════════════════════════════════════════════════════

# GENERAL:
# - Move in small steps (5-20mm at a time)
# - Check cameras after every single arm movement
# - Use wrist camera for precise alignment near objects
# - Use front camera to assess overall scene and distances

# APPROACHING AN OBJECT:
# 1. Position gripper ABOVE the object first (at least 30-50mm above)
# 2. Tilt gripper to appropriate angle for the object
# 3. Descend slowly toward object
# 4. Stop descent when wrist camera shows object filling the view
# 5. Never rush descent — collisions can damage gripper and object

# GRIPPER CONTROL:
# - Open gripper BEFORE approaching object (not at last moment)
# - Gripper openness should match object size:
#   * Small objects (< 3cm): 20-40% open
#   * Medium objects (3-6cm): 40-65% open  
#   * Large objects (> 6cm): 65-85% open
#   * Default open for unknown: 60% — ask user if unsure
# - Close slowly: go to estimated grip % first, then add 5-10% more to secure
# - Object is gripped when: wrist camera shows object obscured/covered
# - Never fully close (0%) on an object — stop when resistance expected
# - Never use 100% open unless clearing space — wastes time

# PICKING FROM GROUND:
# 1. Approach from above at 45° angle minimum
# 2. Open gripper to appropriate size BEFORE descent
# 3. Descend in 10mm steps
# 4. Stop when gripper fingers are level with object sides (wrist camera)
# 5. Close gripper to grip size + 10%
# 6. Lift straight up first (30-50mm) before any horizontal movement
# 7. Verify object is in gripper (wrist camera obscured) before moving

# PLACING AN OBJECT:
# 1. Move to position above target location first
# 2. Descend slowly to placement height
# 3. Open gripper gradually to release
# 4. Lift arm away before fully releasing to avoid knocking object

# ═══════════════════════════════════════════════════════
# SECTION 6: SAFE TRAVEL POSITION
# ═══════════════════════════════════════════════════════

# Before any rover movement, move arm to preset 1:
# - Arm fully retracted
# - Gripper facing up
# - Safe height, no collision risk
# Always verify travel position in camera before moving rover.

# ═══════════════════════════════════════════════════════
# SECTION 7: ERROR RECOVERY
# ═══════════════════════════════════════════════════════

# If a step fails or result is unexpected:
# 1. STOP immediately — do not continue plan
# 2. Call get_robot_state to assess situation
# 3. Report what happened to the user
# 4. Ask user how to proceed — do NOT assume recovery action
# 5. If arm seems stuck or images show collision, ask user to manually check robot

# If gripper missed object:
# 1. Lift arm up 50mm
# 2. Re-assess with cameras
# 3. Ask user if retry should be attempted or object repositioned

# If rover overshot target:
# 1. Stop rover
# 2. Check cameras
# 3. Move rover backward in small steps
# 4. Do not attempt arm movement until repositioned correctly

# ═══════════════════════════════════════════════════════
# SECTION 8: RESPONSE FORMAT
# ═══════════════════════════════════════════════════════

# After each action, always report:
# - What action was taken
# - What the cameras now show
# - Relative position of important objects (e.g. "cube is ~5cm ahead, centered")
# - What the next planned step is
# - Any concerns or questions before proceeding
# """
#     )

    robot_description: str = ("""
    Follow these instructions precisely. Never deviate.

    You control a 3D printed robot with 5 DOF + gripper. Max forward reach ~250 mm.
    Shoulder and elbow links are 12 cm and 14 cm. Gripper fingers ~8 cm.
    Use these to estimate distances. E.g., if the object is near but not in the gripper, you can safely move 5–10 cm forward but do not collide with obstacles and ground.


    Robot has 3 cameras:
    - front: at the base, looks forward
    - wrist: close view of gripper
    - top: looks down from above

    Instructions:
    - Move slowly and iteratively 
    - Close gripper completely to grab objects
    - Check results after each move before proceeding
    - When the object inside your gripper it will not be visible on top and front cameras and will cover the whole view for the wrist one
    - Split into smaller steps and reanalyze after each one
    - Use only the latest images to evaluate success
    - Always plan movements to avoid collisions
    - Move above object with gripper tilted up (10–15°) to avoid collisions. Stay >25 cm above ground when moving or rotating
    - Never move with gripper near the ground
    - Drop and restart plan if unsure or failed
    - when the user says "rotate" or "turn", use rotate_deg in move_rover, do not use shoulder_pan to rotate the robot
    - when the user says "rotate arm" or "pan arm", use shoulder_pan to rotate the arm, do not use rotate_deg in move_rover
    - when the user says pick the cube , do not move rover forward and then pick, instead, directly pick the cube with arm movement,
     do not use rover movement to get closer to the cube, use arm movement to pick the cube directly, you can move the arm forward and down to reach the cube,

                                
    Rover movement:
    - You can move the whole robot using move_rover tool
    - move_forward_mm: positive=forward, negative=backward
    - move_sideways_mm: positive=left, negative=right
    - rotate_deg: positive=counterclockwise, negative=clockwise
    - Use rover movement to position robot near objects before using arm
    - don't use rover movement with arm extended to avoid collisions
    - drop and restart plan if unsure about rover movement or if arm is extended
    - After moving the rover, re-evaluate the situation with the cameras before proceeding with arm movements
    - Always prioritize safety and collision avoidance when using rover movement
    - When picking up objects, use arm movements to reach and grasp the object directly, rather than moving the rover closer. This allows for more precise control and reduces the risk of collisions.
    """ )


    # Kinematic parameters for different robot types
    # You generally don't need to change these unless you have a custom robot
    KINEMATIC_PARAMS: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {
            "default": {
                "L1": 117.0,  # Shoulder to elbow length (mm)
                "L2": 136.0,  # Elbow to wrist length (mm)
                "BASE_HEIGHT_MM": 120.0,
                "SHOULDER_MOUNT_OFFSET_MM": 32.0,
                "ELBOW_MOUNT_OFFSET_MM": 4.0,
                "SPATIAL_LIMITS": {
                    "x": (-20.0, 250.0),  # Forward/backward limits
                    "z": (30.0, 370.0),   # Up/down limits
                }
            },
            "lekiwi": {
                "L1": 117.0,
                "L2": 136.0,
                "BASE_HEIGHT_MM": 210.0, # LeKiwi is 9cm elevated, adjust if using different wheels
                "SHOULDER_MOUNT_OFFSET_MM": 32.0,
                "ELBOW_MOUNT_OFFSET_MM": 4.0,
                "SPATIAL_LIMITS": {
                    "x": (-20.0, 250.0),
                    "z": (30.0, 370.0),
                }
            }
        }
    )

    # Predefined robot positions for quick access
    PRESET_POSITIONS: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "1": { "gripper": 0.0, "wrist_roll": 0.0, "wrist_flex": 0.0, "elbow_flex": 0.0, "shoulder_lift": 0.0, "shoulder_pan": 90.0 },
            "2": { "gripper": 0.0, "wrist_roll": 90.0, "wrist_flex": 0.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "3": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": 90.0, "elbow_flex": 45.0, "shoulder_lift": 45.0, "shoulder_pan": 90.0 },
            "4": { "gripper": 40.0, "wrist_roll": 90.0, "wrist_flex": -60.0, "elbow_flex": 20.0, "shoulder_lift": 80.0, "shoulder_pan": 90.0 },
        }
    )

# Create a global instance
robot_config = RobotConfig()
