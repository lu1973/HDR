import math
import random
import numpy as np
import cv2 as cv

class HdrToolbox:
    """
    A class for handling HDR imaging tasks, including sampling intensity,
    estimating the response curve, computing radiance, and converting radiance
    maps to HDR images.
    This class implements methods based on the paper:
    DEBEVEC, Paul E. et MALIK, Jitendra.
    Recovering high dynamic range radiance maps from photographs.
    In : Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 2023. p. 643-652.
    https://doi.org/10.1145/3581980.3581992

    There is some modification on the weight function. 
    """
    def __init__(self, weight_type='sin', sampling_type='grid', number_of_samples_per_dimension=20, n_sample_per_pixel=1, rand_ref_image='mean'):
        self.Zmin = 0
        self.Zmax = 255
        self.Zmid = (self.Zmax + self.Zmin) // 2
        self.num_intensities = self.Zmax - self.Zmin + 1

        self.weight_type = weight_type                                          # can be 'triangle' or 'sin'
        self.sampling_type = sampling_type                                      # can be 'rand' or 'grid'
        self.number_of_samples_per_dimension = number_of_samples_per_dimension  # for grid sampling
        self.n_sample_per_pixel = n_sample_per_pixel                            # for random extended sampling
        self.rand_ref_image = rand_ref_image                                    # 'mid' or 'mean'


    def weight(self, z) -> float:
        """
        Compute the weight for a given intensity value z based on the selected weight type.
        """
        if self.weight_type == 'triangle':
            return self.weight_triangle(z)
        if self.weight_type == 'sin':
            return self.weight_sin(z)
        raise ValueError(f"Unknown weight_type: {self.weight_type}")


    def weight_triangle(self, z):
        """
        Triangle weighting function as in Debevec & Malik.
        """
        if np.isscalar(z):
            return z - self.Zmin if z <= self.Zmid else self.Zmax - z
        z = np.array(z)
        result = np.where(z <= self.Zmid, z - self.Zmin, self.Zmax - z)
        return result
    

    def weight_sin(self, z):
        """
        Sinusoidal weighting function.
        """
        return 0.5 * np.sin(np.pi * z / self.Zmid - np.pi / 2) + 0.5


    def sample_intensity(self, stack):
        """
        Sample intensity values from the stack using the selected sampling method.
        """
        if self.sampling_type == 'rand':
            return self.sample_intensity_rand(stack)
        if self.sampling_type == 'grid':
            return self.sample_intensity_grid(stack)
        raise ValueError(f"Unknown sampling_type: {self.sampling_type}")


    def sample_intensity_rand(self, stack):
        """
        Randomly sample intensity values from the stack.
        """
        num_images = len(stack)
        random.seed()
        if self.rand_ref_image == 'mean':
            mid_img = np.array(np.mean(stack, axis=0), dtype=np.uint8)
        elif self.rand_ref_image == 'mid':
            mid_img = stack[num_images // 2]
        else:
            raise ValueError(f"Unknown rand_ref_image: {self.rand_ref_image}")

        sample = np.zeros((self.num_intensities * self.n_sample_per_pixel, num_images), dtype=np.uint8)
        k = 0
        for i in range(self.Zmin, self.Zmax + 1):
            # search for pixel position that matchs the desired pixel value
            rows, cols  = np.where(mid_img == i)
            if len(rows) >= self.n_sample_per_pixel:
                # randomly select one into the selected pixel
                idx = random.sample(range(len(rows)), self.n_sample_per_pixel)
                for p in idx:
                    for j in range(num_images):
                        # pick the pixel value in all images
                        sample[k, j] = stack[j][rows[p], cols[p]]
                    k += 1
        return sample[:k]


    def sample_intensity_grid(self, stack):
        """
        Sample pixel values from the stack using a grid sampling method.
        """
        num_images = len(stack)

        width = stack[0].shape[0]
        height = stack[0].shape[1]
        width_iteration = width / self.number_of_samples_per_dimension
        height_iteration = height / self.number_of_samples_per_dimension    
        w_iter = 0
        h_iter = 0

        sample = np.zeros((self.number_of_samples_per_dimension*self.number_of_samples_per_dimension, num_images), dtype=np.uint8)

        for img_index, img in enumerate(stack):
            h_iter = 0
            for i in range(self.number_of_samples_per_dimension):
                w_iter = 0
                for j in range(self.number_of_samples_per_dimension):
                    if math.floor(w_iter) < width and math.floor(h_iter) < height:
                        pixel = img[math.floor(w_iter), math.floor(h_iter)]
                        sample[i * self.number_of_samples_per_dimension + j, img_index] = pixel
                    w_iter += width_iteration
                h_iter += height_iteration

        return sample

    def estimate_curve(self, sample, exps, l):
        """
        Estimate the response curve g and log exposure lnE from the sampled pixel values.
        This method solves the linear system of equations derived from the pixel values
        and their corresponding exposure times.
        """
        # convert the exposure
        exp_logs = np.log2(exps)
        # number of color level
        n_level = self.Zmax - self.Zmin + 1
        # number of selected pixel
        N = sample.shape[0]
        # number of photo
        P = sample.shape[1]

        # there is an error in the paper, the number of equations should be N*P + n_level-2+1
        # it is not a big issue as it adds 0s as additional equations
        # N*P is the number of equations for each pixel in each photo
        # n_level-2 is the number of equations for smoothness 
        # +1 is for scale constraint
        A = np.zeros((N*P + n_level-2 + 1, n_level + N), dtype=np.float64)
        B = np.zeros((N*P + n_level-2 + 1, ), dtype=np.float64)

        k = 0
        # add equation w.g(Zij) - w.ln(Ei) = w.ln(deltaT)
        for i in range(N):
            for j in range(P):
                z = sample[i, j]
                w = self.weight(z)
                A[k, z] = w
                A[k, n_level + i] = -w
                B[k] = w * exp_logs[j]
                k += 1
        
        # add constraint for smoothness
        for i in range(self.Zmin + 1, self.Zmax):
            w = self.weight(i)
            A[k, i - 1] = l * w
            A[k, i] = -2 * l * w
            A[k, i + 1] = l * w
            k += 1

        # add scale constraint
        A[k, self.Zmid] = 1

        x = np.linalg.lstsq(A, B, rcond=None)

        g = x[0][:n_level]
        lnE = x[0][n_level:]

        return g, lnE
    
    def compute_radiance(self, stack, exps, curve):
        """
        Compute radiance following Eq 6 (Debevec & Malik), optimized with NumPy vectorization.
        """
        log_exps = np.log2(exps)

        # Convert stack pixel values to integers for indexing the curve
        # Ensure stack elements are suitable for indexing (e.g., within curve's bounds)
        stack_int = stack.astype(int)

        # Get g values for all pixels and all images
        g_values = curve[stack_int]

        # Get w values for all pixels and all images
        w_values = self.weight(stack_int)

        # Reshape log_exps to be broadcastable to (P, height, width)
        log_exps_reshaped = log_exps[:, np.newaxis, np.newaxis]

        # Calculate the numerator: w * (g - log_exps)
        numerator = np.sum(w_values * (g_values - log_exps_reshaped), axis=0)

        # Calculate the denominator: sum of w
        denominator = np.sum(w_values, axis=0)

        mid_img = stack.shape[0] // 2
        # Calculate the fallback value for all pixels at once.
        fallback_map = g_values[mid_img, :, :] - log_exps[mid_img]

        # Sum the numerator along the 'P' axis (axis 0)
        img_rad = np.where(denominator > 0, numerator / (denominator + 1e-6), fallback_map)

        return img_rad
    
    
    def convert_rad_to_hdr(self, rad):
        """
        Convert a radiance map to an HDR image using tone mapping.
        """
        tm = cv.createTonemapMantiuk(gamma=0.8)
        output = np.clip(255. * tm.process((rad / 255.).astype(np.float32)), 0, 255)
        output = cv.normalize(output, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # permute the channels to RGB
        output = cv.cvtColor(output.astype(np.uint8), cv.COLOR_BGR2RGB)
        return output.astype(np.uint8)
    
    
    def synthetize_ldr_image(self, hdr_radiance_map, g_curve, exposure_time):
        """
        Synthesize an LDR image from the HDR radiance map and the response curve for a given exposure time.
        """
        lnDt = math.log2(exposure_time)

        gZij_map = hdr_radiance_map + lnDt

        LDR = np.zeros(hdr_radiance_map.shape, dtype=np.uint8)

        for c in range(hdr_radiance_map.shape[2]):
            # Get the curve for the current channel
            current_g_curve = g_curve[:, c]

            # Get the gZij values for the current channel
            current_gZij = gZij_map[:, :, c]
            
            # Calculate the absolute differences for the entire channel
            diffs = np.abs(current_g_curve[:, np.newaxis, np.newaxis] - current_gZij[np.newaxis, :, :])
            
            # Find the index of the minimum absolute difference along the curve dimension (axis 0)
            # This returns an (H, W) array of indices
            # np.uint8(idx) means the indices should fit within 0-255, which is typical for 8-bit LDR.
            LDR[:, :, c] = np.argmin(diffs, axis=0).astype(np.uint8)

        LDR = cv.cvtColor(LDR.astype(np.uint8), cv.COLOR_BGR2RGB)
        return LDR
    
