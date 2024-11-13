import azure.functions as func # type: ignore
import logging
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def expand_mask(mask, expand_pixels=2):
    kernel = np.ones((expand_pixels*2+1, expand_pixels*2+1), np.uint8)
    expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    return expanded_mask

def process_image_with_target_color(target_hex_color, wall_mask, image_rgb):
    target_color = np.array(hex_to_rgb(target_hex_color), dtype=np.uint8)
    image_hsv = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2HSV)
    _, _, v_channel = cv2.split(image_hsv)
    v_rgb = cv2.cvtColor(v_channel, cv2.COLOR_GRAY2RGB)
    target_color_image = np.full(v_rgb.shape, target_color, dtype=np.uint8)
    normalized_v_channel = v_channel / 255.0
    enhancement = np.where(
        normalized_v_channel > 0.9, 0.0,
        np.where(
            normalized_v_channel < 0.3, 0.4,
            0.00 + (0.4 - 0.0) * (0.9 - normalized_v_channel) / (0.9 - 0.3)
        )
    )
    enhancement = enhancement[:, :, np.newaxis]
    blended_image = cv2.multiply(v_rgb, target_color_image, scale=1/255.0) * (1 + enhancement)
    blended_image = np.clip(blended_image, 0, 255).astype(np.uint8)

    expanded_mask = expand_mask(wall_mask)

    final_image = np.where(expanded_mask[:, :, np.newaxis], blended_image, np.array(image_rgb))
    return final_image

def apply_colors(image, segments):
    image_rgb = np.array(image.convert('RGB'))
    processed_image = image_rgb.copy()

    for segment_name, segment_data in segments.items():
        mask = np.frombuffer(base64.b64decode(segment_data['mask']), dtype=np.uint8).reshape(image_rgb.shape[:2])
        color = segment_data['color']
        processed_image = process_image_with_target_color(color, mask, processed_image)

    return Image.fromarray(processed_image)


@app.route(route="color__specific_segment")
def color__specific_segment(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        # Get the JSON data from the request
        req_body = req.get_json()
        
        # Extract data from the JSON
        original_image_data = req_body['original_image']
        selected_segments = req_body['selected_segments']
        
        # Decode the original image
        original_image = Image.open(io.BytesIO(base64.b64decode(original_image_data)))
        
        # Apply colors to the selected segments
        colored_image = apply_colors(original_image, selected_segments)
        
        # Convert the colored image to base64
        buffered = io.BytesIO()
        colored_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Return the colored image
        return func.HttpResponse(
            json.dumps({'colored_image': img_str}),
            mimetype="application/json",
            status_code=200
        )
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return func.HttpResponse(
            f"An error occurred: {str(e)}",
            status_code=500
        )