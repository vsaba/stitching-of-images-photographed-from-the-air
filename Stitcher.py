import cv2
import numpy as np
import cv2 as cv

class Stitcher:
    """
    A class which has all the necessary fucntions to match 2 images together,
    expects the images to be in order, with the first image being the "query" image,
    and the second image being the "train"
    """

    def stitch(self, images, ratio, threshold):
        """
        Stitches two images together based on the provided ratio and threshold
        :param images: the provided images
        :param ratio: the provided ratio
        :param threshold: the provided threshold
        :return: returns a stitched and trimmed image
        """
        image1, image2 = images

        # SIFT
        kps1, descriptor1 = self.detect(image1)
        kps2, descriptor2 = self.detect(image2)

        # Karakteristike
        matches = self.match(descriptor1, descriptor2, ratio)

        # računanje homografije
        H, status = self.homography(kps1, kps2, matches, threshold)

        # primjena homografije
        result_img = cv.warpPerspective(image2, H,
                                        (image1.shape[1] + image2.shape[1], image2.shape[0] + image1.shape[0]))

        # spajanje slika
        result_img[0:image1.shape[0], 0:image1.shape[1]] = image1

        # podrezivanje slika zbog urednosti
        result_img = self.trim(result_img)

        return result_img

    # detects the key points of the image
    def detect(self, image):
        """
        Detects the key points of the provided image
        :param image: The provided image
        :return: Returns the key points and a descriptor of the image
        """
        sift = cv.SIFT_create()
        kps, descriptor = sift.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])

        return kps, descriptor

    def match(self, descriptor1, descriptor2, ratio):
        """
        Finds all the matches between the descriptors of the two provided images,
        filters the best matches based on the provided ratio
        :param descriptor1: The provided descriptor of the first/left image
        :param descriptor2: The provided descriptor of the second/right image
        :param ratio: The provided ratio
        :return: Returns a list of all filtered matches
        """
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        all_matches = matcher.knnMatch(descriptor1, descriptor2, 2)

        clean_matches = []

        for m in all_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                clean_matches.append((m[0].trainIdx, m[0].queryIdx))

        return clean_matches

    def homography(self, kps1, kps2, matches, threshold):
        """
        Finds the homography matrix based on the provided keypoints and matches
        :param kps1:
        :param kps2:
        :param matches:
        :param threshold:
        :return:
        """
        if len(matches) < 4:
            return None
         
        pts1 = np.float32([kps1[i] for (_, i) in matches])
        pts2 = np.float32([kps2[i] for (i, _) in matches])

        H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, threshold)

        return H, status

    def trim(self, image):
        """
        Trims the provided image, so no black spots are visible
        :param image: The provided image
        :return: Returns the trimmed/cropped image
        """
        if not np.sum(image[0]):
            return self.trim(image[1:])
        if not np.sum(image[-1]):
            return self.trim(image[:-2])
        if not np.sum(image[:, 0]):
            return self.trim(image[:, 1:])
        if not np.sum(image[:, -1]):
            return self.trim(image[:, :-2])
        return image
