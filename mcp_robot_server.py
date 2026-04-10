"""
MCP server to control the robot.
"""

from __future__ import annotations

import io
import logging
import concurrent.futures
from typing import List, Optional, Union

import numpy as np
from PIL import Image as PILImage

from mcp.server.fastmcp import FastMCP, Image

from robot_controller import RobotController
from config import robot_config

import atexit
import traceback
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s MCP_Server %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Initialise FastMCP server
# -----------------------------------------------------------------------------

mcp = FastMCP(
    name="SO-ARM100 robot controller",
    host="0.0.0.0",
    port=3001
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

_robot: Optional[RobotController] = None

# How long (seconds) to wait for robot hardware to respond before giving up.
# LeKiwiClient connects over TCP to the Pi — if the Pi is off or lekiwi_host.py
# isn't running, the old code hung forever. Now it fails fast with a clear message.
ROBOT_CONNECT_TIMEOUT = 15.0


class RobotConnectionError(RuntimeError):
    """Raised when robot hardware is unreachable or failed to connect."""
    pass


def _np_to_mcp_image(arr_rgb: np.ndarray) -> Image:
    """Convert a numpy RGB image to MCP image format."""
    pil_img = PILImage.fromarray(arr_rgb)
    with io.BytesIO() as buf:
        pil_img.save(buf, format="JPEG")
        raw_data = buf.getvalue()
    return Image(data=raw_data, format="jpeg")


def _error_response(message: str) -> List:
    """Return a clean error response dict the LLM can understand."""
    logger.error(f"Tool error response: {message}")
    return [{"status": "error", "message": message, "robot_state": {"human_readable_state": {}}}]


def get_robot() -> RobotController:
    """Lazy-initialise the global RobotController instance.

    We avoid creating the controller at import time so the MCP Inspector can
    start even if the hardware is not connected. The first tool/resource call
    that actually needs the robot will trigger the connection.

    Uses a thread + timeout so that if the Pi is off or lekiwi_host.py isn't
    running, the call fails in ~15s with a clear RobotConnectionError instead
    of hanging forever (which looked like the agent was just stuck with no output).
    """
    global _robot
    if _robot is None:
        logger.info(f"Connecting to robot hardware (timeout={ROBOT_CONNECT_TIMEOUT}s)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(RobotController)
            try:
                _robot = future.result(timeout=ROBOT_CONNECT_TIMEOUT)
                logger.info("RobotController initialized successfully.")
            except concurrent.futures.TimeoutError:
                msg = (
                    f"Robot hardware did not respond within {ROBOT_CONNECT_TIMEOUT}s. "
                    "Check that the LeKiwi Pi is powered on and reachable on the network, "
                    "and that lekiwi_host.py is running on the Pi."
                )
                logger.error(f"MCP: {msg}")
                raise RobotConnectionError(msg)
            except Exception as e:
                msg = f"Robot hardware connection failed: {e}"
                logger.error(f"MCP: {msg}", exc_info=True)
                raise RobotConnectionError(msg)
    return _robot


def get_state_with_images(result_json: dict, is_movement: bool = False) -> List[Union[Image, dict, list]]:
    """Combine robot state with camera images into a unified response format."""
    try:
        robot = get_robot()
    except RobotConnectionError as e:
        return [result_json, f"Warning: Robot disconnected: {str(e)}"]
    try:
        if is_movement:
            time.sleep(1.0)

        raw_imgs = robot.get_camera_images()

        if not raw_imgs:
            logger.warning("MCP: No camera images returned from robot controller.")
            return [result_json, "Warning: No camera images available."]

        mcp_images = [_np_to_mcp_image(img) for img in raw_imgs.values()]

        result_json["robot_state"] = result_json["robot_state"]["human_readable_state"]

        return [result_json] + mcp_images
    except Exception as e:
        logger.error(f"Error getting camera images: {str(e)}")
        logger.error(traceback.format_exc())
        return [result_json] + ["Error getting camera images"]


# -----------------------------------------------------------------------------
# Tools – read-only
# -----------------------------------------------------------------------------

@mcp.tool(description="Get a description of the robot and instructions for the user. Run it before using any other tool.")
def get_initial_instructions() -> str:
    return robot_config.robot_description


@mcp.tool(description="Get current robot state with images from all cameras. Returns list of objects: json with results of the move and current state of the robot and images from all cameras")
def get_robot_state():
    try:
        robot = get_robot()
    except RobotConnectionError as e:
        return _error_response(str(e))

    move_result = robot.get_current_robot_state()
    result_json = move_result.to_json()
    logger.info(f"MCP: get_robot_state outcome: {result_json.get('status', 'success')}, Msg: {move_result.msg}")
    return get_state_with_images(result_json, is_movement=False)


# -----------------------------------------------------------------------------
# Tools – actuation
# -----------------------------------------------------------------------------

@mcp.tool(
        description="""
        Move the robot with intuitive controls.
        Args:
            move_gripper_up_mm (float, optional): Distance to move gripper up (positive) or down (negative) in mm
            move_gripper_forward_mm (float, optional): Distance to move gripper forward (positive) or backward (negative) in mm
            tilt_gripper_down_angle (float, optional): Angle to tilt gripper down (positive) or up (negative) in degrees
            rotate_gripper_counterclockwise_angle (float, optional): Angle to rotate gripper counterclockwise (positive) or clockwise (negative) in degrees
            rotate_robot_left_angle (float, optional): Angle to rotate entire robot counterclockwise/left (positive) or clockwise/right (negative) in degrees
        Expected input format:
        {
            "move_gripper_up_mm": "10",
            "move_gripper_forward_mm": "-5",
            "tilt_gripper_down_angle": "10",
            "rotate_gripper_counterclockwise_angle": "-15",
            "rotate_robot_left_angle": "15"
        }
        Returns:
            list: List containing JSON with robot state and camera images
    """
        )
def move_robot(
    move_gripper_up_mm=None,
    move_gripper_forward_mm=None,
    tilt_gripper_down_angle=None,
    rotate_gripper_counterclockwise_angle=None,
    rotate_robot_left_angle=None
):
    try:
        robot = get_robot()
    except RobotConnectionError as e:
        return _error_response(str(e))

    logger.info(f"MCP Tool: move_robot received: up={move_gripper_up_mm}, fwd={move_gripper_forward_mm}, "
                f"tilt={tilt_gripper_down_angle}, grip_rot={rotate_gripper_counterclockwise_angle}, "
                f"robot_rot={rotate_robot_left_angle}")

    move_params = {
        "move_gripper_up_mm": float(move_gripper_up_mm) if move_gripper_up_mm is not None else None,
        "move_gripper_forward_mm": float(move_gripper_forward_mm) if move_gripper_forward_mm is not None else None,
        "tilt_gripper_down_angle": float(tilt_gripper_down_angle) if tilt_gripper_down_angle is not None else None,
        "rotate_gripper_counterclockwise_angle": float(rotate_gripper_counterclockwise_angle) if rotate_gripper_counterclockwise_angle is not None else None,
        "rotate_robot_left_angle": float(rotate_robot_left_angle) if rotate_robot_left_angle is not None else None,
    }

    actual_move_params = {k: v for k, v in move_params.items() if v is not None}

    if not actual_move_params:
        current_state_result = robot.get_current_robot_state()
        result_json = current_state_result.to_json()
        result_json["message"] = "No movement parameters provided to move_robot tool."
        return get_state_with_images(result_json, is_movement=False)

    move_execution_result = robot.execute_intuitive_move(**actual_move_params)
    result_json = move_execution_result.to_json()

    logger.info(f"MCP: move_robot final outcome: {result_json.get('status', 'success')}, "
                f"Msg: {result_json.get('message', '')}, Warnings: {len(result_json.get('warnings', []))}")

    return get_state_with_images(result_json, is_movement=True)


@mcp.tool(description="Control the robot's gripper openness from 0% (completely closed) to 100% (completely open). Expected input format: {gripper_openness_pct: '50'}. Returns list of objects: json with results of the move and current state of the robot and images from all cameras")
def control_gripper(gripper_openness_pct):
    try:
        robot = get_robot()
    except RobotConnectionError as e:
        return _error_response(str(e))

    try:
        openness = float(gripper_openness_pct)
        logger.info(f"MCP Tool: control_gripper called with openness={gripper_openness_pct}%")

        move_result = robot.set_joints_absolute({'gripper': openness})
        result_json = move_result.to_json()
        logger.info(f"MCP: control_gripper outcome: {result_json.get('status', 'success')}, "
                    f"Msg: {move_result.msg}, Warnings: {len(move_result.warnings)}")
        return get_state_with_images(result_json, is_movement=True)

    except (ValueError, TypeError) as e:
        logger.error(f"MCP: control_gripper received invalid input: {gripper_openness_pct}, error: {str(e)}")
        return _error_response(f"Invalid gripper openness value: {str(e)}")


@mcp.tool(
    description="""
    Move the robot's rover base (wheels).
    Use this to move the whole robot to a new position.
    Args:
        move_forward_mm (float, optional): Distance to move forward (positive) or backward (negative) in mm
        move_sideways_mm (float, optional): Distance to move left (positive) or right (negative) in mm
        rotate_deg (float, optional): Angle to rotate counterclockwise (positive) or clockwise (negative) in degrees
    Expected input format:
    {
        "move_forward_mm": "100",
        "move_sideways_mm": "-50",
        "rotate_deg": "90"
    }
    Returns:
        list: List containing JSON with status and current robot state, and camera images
    """
)
def move_rover(
    move_forward_mm=None,
    move_sideways_mm=None,
    rotate_deg=None
):
    try:
        robot = get_robot()
    except RobotConnectionError as e:
        return _error_response(str(e))

    logger.info(f"MCP Tool: move_rover called: forward={move_forward_mm}, sideways={move_sideways_mm}, rotate={rotate_deg}")

    move_params = {
        "move_forward_mm": float(move_forward_mm) if move_forward_mm is not None else None,
        "move_sideways_mm": float(move_sideways_mm) if move_sideways_mm is not None else None,
        "rotate_deg": float(rotate_deg) if rotate_deg is not None else None,
    }

    actual_params = {k: v for k, v in move_params.items() if v is not None}

    if not actual_params:
        current_state = robot.get_current_robot_state()
        result_json = current_state.to_json()
        result_json["message"] = "No movement parameters provided to move_rover tool."
        return get_state_with_images(result_json, is_movement=False)

    move_result = robot.move_rover(**actual_params)
    result_json = move_result.to_json()

    logger.info(f"MCP: move_rover outcome: {result_json.get('status', 'success')}")

    return get_state_with_images(result_json, is_movement=True)


# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------

def _cleanup():
    """Disconnect from hardware on server shutdown."""
    global _robot
    if _robot is not None:
        try:
            _robot.disconnect()
        except Exception as e_disc:
            logger.error(f"MCP: Exception during _robot.disconnect(): {e_disc}", exc_info=True)

atexit.register(_cleanup)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting MCP Robot Server...")
    try:
        mcp.run(transport="sse")
    except SystemExit as e:
        logger.error(f"MCP Server failed to start: {e}")
    except Exception as e_main:
        logger.error(f"MCP Server CRITICAL RUNTIME ERROR: {e_main}", exc_info=True)