
import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Function to perform object identification and calculate metrics for a scene
def identify_objects(scene_image, object_images, true_objects_in_scene, threshold=50):
    # Read the scene image
    gray_scene = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)

    # Initialize lists to store evaluation metrics
    true_positives, false_positives, true_negatives, false_negatives = [], [], [], []

    scene_with_objects = scene_image.copy()

    for object_image in object_images:
        # Detect keypoints and compute descriptors for both scene and object images
        orb = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE)
        keypoints_scene, descriptors_scene = orb.detectAndCompute(gray_scene, None)
        keypoints_object, descriptors_object = orb.detectAndCompute(object_image, None)

        # Match descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors_object, descriptors_scene, k=2)

        # Filter matches based on distance
        good_matches = [m for m, n in matches if m.distance < threshold * n.distance]

        # Evaluate metrics
        if true_objects_in_scene:
            true_positives.append(len(good_matches))
            false_negatives.append(len(matches) - len(good_matches))
            true_negatives.append(0)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_negatives.append(0)
            true_negatives.append(len(matches) - len(good_matches))
            false_positives.append(len(good_matches))

        # Draw a polygon around the object in the scene
        if len(good_matches) >= 4:
            src_pts = np.float32([keypoints_object[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find the perspective transformation using RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Apply the transformation to the object corners to get the bounding box in the scene
            h, w, _ = object_image.shape
            object_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            scene_corners = cv2.perspectiveTransform(object_corners, M)

            # Draw a polygon around the object in the scene
            cv2.polylines(scene_with_objects, [np.int32(scene_corners)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Calculate precision, recall, F1-score, and accuracy
    precision, recall, fscore, _ = precision_recall_fscore_support(
        true_objects_in_scene,
        [1 if tp > 0 else 0 for tp in true_positives],
        average='binary',
    )
    accuracy = accuracy_score(true_objects_in_scene, [1 if tp > 0 else 0 for tp in true_positives])

    metrics = {
        'True Positives': true_positives,
        'False Positives': false_positives,
        'True Negatives': true_negatives,
        'False Negatives': false_negatives,
        'Precision': precision,
        'Recall': recall,
        'F1-score': fscore,
        'Accuracy': accuracy,
        'Scene with Objects': scene_with_objects,
    }

    return metrics

# Step 1: Read Images
scene_image = cv2.imread('Scenes/scene1_1.png')
box_image = cv2.imread('Objects/box.png')
elephant_image = cv2.imread('Objects/elephant.png')
true_objects_in_scene = [1, 1, 0, 0, 0] # this mean out of all the objects only first two are present in scene

object_images = [cv2.imread('Objects/box.png'), cv2.imread('Objects/elephant.png'), cv2.imread('Objects/bag.jpg'), cv2.imread('Objects/perfume.jpg'), cv2.imread('Objects/plant.jpg')]

# Step 2: Detect Point Features for the Box with increased parameters
orb_box = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE)
keypoints_scene, descriptors_scene = orb_box.detectAndCompute(scene_image, None)
keypoints_box, descriptors_box = orb_box.detectAndCompute(box_image, None)

# Step 3: Extract Feature Descriptors (Already done in the detection step)

# Step 4: Find Putative Point Matches for the Box
bf_box = cv2.BFMatcher()
matches_box = bf_box.knnMatch(descriptors_box, descriptors_scene, k=2)

# Apply ratio test to find good matches for the Box
good_matches_box = []
for m, n in matches_box:
    if m.distance < 0.75 * n.distance:
        good_matches_box.append(m)

# Visualize Matches and Keypoints for the Box
img_matches_box = cv2.drawMatches(box_image, keypoints_box, scene_image, keypoints_scene, good_matches_box, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('Matches for Box', img_matches_box)
cv2.imwrite('Matches/box.png', img_matches_box)
# cv2.waitKey(0)

# Show Keypoints for the Box
scene_with_keypoints_box = scene_image.copy()
scene_with_keypoints_box = cv2.drawKeypoints(scene_with_keypoints_box, keypoints_box, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('Scene with Keypoints for Box', scene_with_keypoints_box)
cv2.imwrite('Keypoints/boxKp.png', scene_with_keypoints_box)
# cv2.waitKey(0)

# Step 5: Locate the Box in the Scene Using Putative Matches
if len(good_matches_box) >= 4:
    src_pts_box = np.float32([keypoints_box[m.queryIdx].pt for m in good_matches_box]).reshape(-1, 1, 2)
    dst_pts_box = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches_box]).reshape(-1, 1, 2)

    # Find the perspective transformation using RANSAC for the Box
    M_box, mask_box = cv2.findHomography(src_pts_box, dst_pts_box, cv2.RANSAC, 5.0)

    # Apply the transformation to the box corners to get the bounding box in the scene
    h_box, w_box, _ = box_image.shape
    box_corners = np.float32([[0, 0], [0, h_box - 1], [w_box - 1, h_box - 1], [w_box - 1, 0]]).reshape(-1, 1, 2)
    scene_corners_box = cv2.perspectiveTransform(box_corners, M_box)

    # Draw a polygon around the box in the scene
    scene_with_objects = scene_image.copy()
    cv2.polylines(scene_with_objects, [np.int32(scene_corners_box)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the result with the box outline
    # cv2.imshow('Object Detection Result for Box', scene_with_objects)
    cv2.imwrite('Detected_Objects/box_detection.png', scene_with_objects)
else:
    print("Not enough good matches to locate the box.")

# Step 2: Detect Point Features for the Elephant with increased parameters
orb_elephant = cv2.ORB_create(nfeatures=3000, scoreType=cv2.ORB_FAST_SCORE)
keypoints_elephant, descriptors_elephant = orb_elephant.detectAndCompute(elephant_image, None)

# Step 3: Extract Feature Descriptors (Already done in the detection step)

# Step 4: Find Putative Point Matches for the Elephant
bf_elephant = cv2.BFMatcher()
matches_elephant = bf_elephant.knnMatch(descriptors_elephant, descriptors_scene, k=2)

# Apply ratio test to find good matches for the Elephant
good_matches_elephant = []
for m, n in matches_elephant:
    if m.distance < 0.75 * n.distance:
        good_matches_elephant.append(m)

# Visualize Matches and Keypoints for the Elephant
img_matches_elephant = cv2.drawMatches(elephant_image, keypoints_elephant, scene_image, keypoints_scene, good_matches_elephant, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('Matches for Elephant', img_matches_elephant)
cv2.imwrite('Matches/elephant.png',img_matches_elephant)
# cv2.waitKey(0)

# Show Keypoints for the Elephant
scene_with_keypoints_elephant = scene_image.copy()
scene_with_keypoints_elephant = cv2.drawKeypoints(scene_with_keypoints_elephant, keypoints_elephant, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('Scene with Keypoints for Elephant', scene_with_keypoints_elephant)
cv2.imwrite('Keypoints/elephantKp.png', scene_with_keypoints_elephant)
# cv2.waitKey(0)

# Step 5: Locate the Elephant in the Scene Using Putative Matches
if len(good_matches_elephant) >= 4:
    src_pts_elephant = np.float32([keypoints_elephant[m.queryIdx].pt for m in good_matches_elephant]).reshape(-1, 1, 2)
    dst_pts_elephant = np.float32([keypoints_scene[m.trainIdx].pt for m in good_matches_elephant]).reshape(-1, 1, 2)

    # Find the perspective transformation using RANSAC for the Elephant
    M_elephant, mask_elephant = cv2.findHomography(src_pts_elephant, dst_pts_elephant, cv2.RANSAC, 5.0)

    # Apply the transformation to the elephant corners to get the bounding box in the scene
    h_elephant, w_elephant, _ = elephant_image.shape
    elephant_corners = np.float32([[0, 0], [0, h_elephant - 1], [w_elephant - 1, h_elephant - 1], [w_elephant - 1, 0]]).reshape(-1, 1, 2)
    scene_corners_elephant = cv2.perspectiveTransform(elephant_corners, M_elephant)

    # Draw a polygon around the elephant in the scene
    cv2.polylines(scene_with_objects, [np.int32(scene_corners_elephant)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Display the result with the box and elephant outlines
    # cv2.imshow('Object Detection Result for Box and Elephant', scene_with_objects)
    cv2.imwrite('Detected_Objects/elephant_detection.png',scene_with_objects)
else:
    print("Not enough good matches to locate the elephant.")



# cv2.waitKey(0)

result = identify_objects(scene_image, object_images, true_objects_in_scene,)
print(result)
precision, recall, fscore, accuracy = result['Precision'], result['Recall'], result['F1-score'], result['Accuracy']
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {fscore:.2f}')
print(f'Accuracy: {accuracy:.2f}')

