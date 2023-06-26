import numpy as np
from scipy.ndimage import zoom
from skimage import exposure
from skimage import transform

class ImageTransformer:
  def __init__(self):
    pass

  def clipped_zoom(self, img, zoom_factor, **kwargs):

      h, w = img.shape[:2]

      # For multichannel images we don't want to apply the zoom factor to the RGB
      # dimension, so instead we create a tuple of zoom factors, one per array
      # dimension, with 1's for any trailing dimensions after the width and height.
      zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

      # Zooming out
      if zoom_factor < 1:

          # Bounding box of the zoomed-out image within the output array
          zh = int(np.round(h * zoom_factor))
          zw = int(np.round(w * zoom_factor))
          top = (h - zh) // 2
          left = (w - zw) // 2

          # Zero-padding
          out = np.zeros_like(img)
          out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

      # Zooming in
      elif zoom_factor > 1:

          # Bounding box of the zoomed-in region within the input array
          zh = int(np.round(h / zoom_factor))
          zw = int(np.round(w / zoom_factor))
          top = (h - zh) // 2
          left = (w - zw) // 2

          out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

          # `out` might still be slightly larger than `img` due to rounding, so
          # trim off any extra pixels at the edges
          trim_top = ((out.shape[0] - h) // 2)
          trim_left = ((out.shape[1] - w) // 2)
          out = out[trim_top:trim_top+h, trim_left:trim_left+w]

      # If zoom_factor == 1, just return the input array
      else:
          out = img
      out = (out-out.min())/(out.max()-out.min())
      out = exposure.adjust_gamma(out, 2)
      return out
  
  def rotate(self, image, angle):
    return transform.rotate(image, angle=angle)
  
  def transform_images(self, x_test):
    x_test_rotated = []
    x_test_zoomed = []
    x_test_zoomed_out = []

    for i in range(len(x_test)):
      x_test_rotated.append(self.rotate(x_test[i,:,:], angle=np.random.choice([-45,-40,40,45],1)[0])) 
      x_test_zoomed.append(self.clipped_zoom(x_test[i,:,:], 1.4))
      x_test_zoomed_out.append(self.clipped_zoom(x_test[i,:,:], 0.6))

    x_test_rotated = np.array(x_test_rotated)
    x_test_zoomed = np.array(x_test_zoomed)
    x_test_zoomed_out = np.array(x_test_zoomed_out)
    
    return x_test_rotated, x_test_zoomed, x_test_zoomed_out
      