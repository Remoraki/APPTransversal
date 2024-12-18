from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
from scipy.interpolate import splprep, splev

from PIL import Image
import io
from shapely.geometry.polygon import Polygon, Point
from scipy.ndimage import label

import cv2


app = Flask(__name__)

# Endpoint to serve the main page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle received points and return interpolated path
@app.route('/interpolate', methods=['POST'])
def interpolate():
    global left_boundary, right_boundary

    points = request.json['points']
    
    if len(points) < 2:
        return jsonify({"error": "Need at least two points for interpolation"}), 400
    
    # Extract x and y coordinates from the points
    x = np.array([0]+[p[0] for p in points])
    y = np.array([points[0][1]]+[p[1] for p in points])

    # Perform cubic spline interpolation
    tck, u = splprep([x, y],s=0)
    
    
    # Generate a smooth path (interpolated points)
    u_new = np.linspace(0, 1, 100)
    x_new, y_new = splev(u_new, tck)
    
    # Prepare the interpolated path for the front-end
    interpolated_points = list(zip(x_new.tolist(), y_new.tolist()))

    # Calculate the difference between consecutive points
    vect_diff = np.array(interpolated_points)
    vect_diff = vect_diff[1:] - vect_diff[:-1]

    # Calculate the boundaries of the path
    angle = 90.
    theta = (angle/180.) * np.pi

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
    
    left_boundary = 300*np.dot(vect_diff, rotMatrix)/np.linalg.norm(vect_diff, axis=1)[:, None] + np.array(interpolated_points[:-1])

    angle = -90.
    theta = (angle/180.) * np.pi

    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)], 
                            [np.sin(theta),  np.cos(theta)]])
    right_boundary = 300*np.dot(vect_diff, rotMatrix)/np.linalg.norm(vect_diff, axis=1)[:, None] + np.array(interpolated_points[:-1])


    
    return jsonify({"interpolated_points": interpolated_points, "left_boundary": left_boundary.tolist(), "right_boundary": right_boundary.tolist()})


@app.route('/process_image', methods=['POST'])
def process_image():
    nb_ite_dilate = 0
    file = request.files['texture']
    if not file:
        return "No file provided", 400
    # Open the image with PIL
    image = Image.open(file)

    bg_file = request.files['background']
    if not bg_file:
        return "No file provided", 400
    bg_image = Image.open(bg_file)
    
    # Open the mask using filename
    mask_file = "./textures/" + file.filename.split('.')[0] + "_masque.png"
    mask = Image.open(mask_file).convert("L")  # Convert mask to grayscale
    
    
    
    # Create a blank canvas (3000x3000 pixels)
    canvas_size = (3000, 3000)
    canvas = Image.new("L", canvas_size, color=0)
    canvas_colored = Image.new("RGB", canvas_size, color=(0, 0, 0))
    canvas_background = Image.new("RGB", canvas_size, color=(0, 0, 0))

    # Define grid properties
    cell_size = 500  # Each image will be 500x500 pixels
    x_cells = canvas_size[0] // cell_size
    y_cells = canvas_size[1] // cell_size

    # Resize the uploaded image to fit the grid cell
    resized_image = image.resize((cell_size, cell_size))
    mask = mask.resize((cell_size, cell_size))
    resized_bg_image = bg_image.resize((cell_size, cell_size))

    # Define the path polygon
    path_polygon = Polygon(np.concatenate([left_boundary, np.flip(right_boundary, axis=0)]))

    # Place the image in the grid
    for y in range(y_cells):
        for x in range(x_cells):
            top_left_x = x * cell_size
            top_left_y = y * cell_size

            cell_polygon = Polygon([(top_left_x-cell_size, top_left_y-cell_size), 
                                    (top_left_x + cell_size*2, top_left_y-cell_size), 
                                    (top_left_x + cell_size*2, top_left_y + cell_size*2), 
                                    (top_left_x-cell_size, top_left_y + cell_size*2)])

            # Check if the cell center is inside the polygon
            if path_polygon.intersects(cell_polygon):
                canvas_colored.paste(resized_image, (top_left_x, top_left_y))
                canvas.paste(mask, (top_left_x, top_left_y))
            #paste the background
            canvas_background.paste(resized_bg_image, (top_left_x, top_left_y))

    # Convert the canvas to a numpy array for OpenCV processing
    canvas_array = np.array(canvas)

    # Invert the mask to make black regions white and white regions black
    inverted_mask = cv2.bitwise_not(canvas_array)

    # Dilate the inverted mask (expand black areas)
    dilated_mask = cv2.dilate(inverted_mask, np.ones((5, 5), np.uint8), iterations=nb_ite_dilate)

    # Re-invert the mask back to its original polarity
    canvas_array = cv2.bitwise_not(dilated_mask)


    # Detect contours in the mask
    contours, _ = cv2.findContours(canvas_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new blank canvas to draw the filtered contours
    filtered_canvas = np.zeros_like(canvas_array)

    # Filter and draw contours within the path polygon
    for contour in contours:
        # Convert contour to a Shapely Polygon
        contour_points = [tuple(point[0]) for point in contour]
        #contour_polygon = Polygon(contour_points)
        if len(contour_points) >= 3:  # A valid polygon requires at least 3 points
            contour_polygon = Polygon(contour_points)

            # Check if the contour is doesn't intersect the path polygon
            if path_polygon.intersects(contour_polygon):
                # Draw the contour onto the filtered canvas
                cv2.drawContours(
                    filtered_canvas, [np.array(contour_points, dtype=np.int32)], -1, (255, 0, 255), thickness=cv2.FILLED
                )
    # Dilate back the filtered canvas
    filtered_canvas = cv2.dilate(filtered_canvas, np.ones((5, 5), np.uint8), iterations=nb_ite_dilate)

    # Apply the final mask to the colored canvas by multiplying
    canvas_colored_array = np.array(canvas_colored)  # Convert colored canvas to numpy array
    texture_colored_array = cv2.bitwise_and(canvas_colored_array, canvas_colored_array, mask=filtered_canvas)

    #add the background in the inverse of the mask
    canvas_background_array = np.array(canvas_background)
    background_array = cv2.bitwise_and(canvas_background_array, canvas_background_array, mask=cv2.bitwise_not(filtered_canvas))

    # Combine the texture and background
    final_canvas_array = cv2.add(texture_colored_array, background_array)
    

    # Convert the final canvas back to a PIL image
    final_canvas = Image.fromarray(final_canvas_array)
    
    # Save the processed image to a BytesIO object
    img_io = io.BytesIO()
    final_canvas.save(img_io, 'PNG')
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
