import matplotlib.pyplot as plt

def myplot(degraded, reconstruction, target):
    plt.figure(figsize=(10,5))
    plt.subplot(1,3,1); plt.imshow(degraded.permute(0,2,3,1).squeeze().cpu()); plt.title('Degraded')
    plt.subplot(1,3,2); plt.imshow(reconstruction.permute(0,2,3,1).squeeze().cpu()); plt.title('Reconstruction')
    plt.subplot(1,3,3); plt.imshow(target.permute(0,2,3,1).squeeze().cpu()); plt.title('Ground truth')
    plt.show()