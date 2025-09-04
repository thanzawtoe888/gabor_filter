# IMPORT module and function
from skimage.filters import sobel

#Apply edge detection filter to an image
edge_sobel = sobel('../images/frame_0015.jpg')

# show original and resulting image to compare
plot_comparison(('../images/frame_0015.jpg', edge_sobel), titles=['Original', 'Sobel Edge Detection'])

plot_comparison('../images/frame_0015.jpg', edge_sobel,"Edge with Sobel")
                 
                
                
                ##### comparing plot 
def plot_comparison (original, filtered, titile_filtered);

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6) , sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(titile_filtered)
    ax2.axis('off')