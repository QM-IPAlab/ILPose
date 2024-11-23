
import cv2
import numpy as np


def detect_keypoints(image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2):
    # Initialize Brute-Force Matcher
    bf = cv2.BFMatcher()

    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches


def compute_frame_difference(image1, image2):
    # Compute the absolute difference between two images
    abs_diff = cv2.absdiff(image1, image2)

    # Calculate the mean pixel difference
    mean_diff = np.mean(abs_diff)

    return mean_diff


def compute_mask_difference(mask1, mask2):
    # Compute the absolute difference between two masks
    abs_diff = cv2.absdiff(mask1, mask2)

    # Calculate the mean pixel difference
    mean_diff = np.mean(abs_diff)

    return mean_diff

def select_keyframes(images, masks, num_keyframes):
    keyframes = []
    keyframes_indices = []
    keyframes_scores = []

    # Compute descriptors for the first image
    keypoints_prev, descriptors_prev = detect_keypoints(images[0])

    for i in range(1, len(images)):
        keypoints_curr, descriptors_curr = detect_keypoints(images[i])

        # Match keypoints between consecutive frames
        matches = match_keypoints(descriptors_prev, descriptors_curr)

        # Compute frame difference
        frame_diff = compute_frame_difference(images[i - 1], images[i])

        # Compute mask difference
        mask_diff = compute_mask_difference(masks[i-1], masks[i])

        # Compute the score for the current frame
        score = len(matches) / (frame_diff + mask_diff)

        # Add the current frame to keyframes if it has a high enough score
        if len(keyframes) < num_keyframes:
            keyframes.append(images[i])
            keyframes_indices.append(i)
            keyframes_scores.append(score)
        else:
            min_score_index = np.argmin(keyframes_scores)
            if score > keyframes_scores[min_score_index]:
                keyframes[min_score_index] = images[i]
                keyframes_indices[min_score_index] = i
                keyframes_scores[min_score_index] = score

        # Update descriptors for the next iteration
        keypoints_prev, descriptors_prev = keypoints_curr, descriptors_curr

    return keyframes, keyframes_indices


def main():
    # Load sequential images

    object_index = 4
    keyframes_num = 30

    list_rgb = []
    list_mask = []
    item_count = 0
    input_file = open('/Linemod/Linemod_preprocessed/data/{0}/train.txt'.format('%02d' % object_index))
    while 1:
        item_count += 1
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
            list_rgb.append('/Linemod/Linemod_preprocessed/data/{0}/rgb/{1}.png'.format('%02d' % object_index, input_line))
            list_mask.append('/Linemod/Linemod_preprocessed/data/{0}/mask/{1}.png'.format('%02d' % object_index, input_line))

    frame_paths = list_rgb
    ape_masks = list_mask


    images = [cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE) for frame_path in frame_paths]
    masks = [cv2.imread(ape_mask, cv2.IMREAD_GRAYSCALE) for ape_mask in ape_masks]



    # Select keyframes
    keyframes, keyframe_indices = select_keyframes(images, masks, keyframes_num)

    output_path = '/Linemod/Linemod_preprocessed/data/{0}/selected_keyframes_{1}.txt'.format('%02d' % object_index, keyframes_num)
    with open(output_path, "w") as file:
        for i in range(len(keyframe_indices)):
            the_frame = frame_paths[keyframe_indices[i]]
            index = the_frame.split('/')[-1].replace('.png', '')
            print(the_frame, index)
            file.write(index + '\n')
    print("finshed keyframes selection, the file is saved in: ", output_path)

if __name__ == "__main__":
    main()
